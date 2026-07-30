# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from itertools import combinations
from time import perf_counter
from typing import cast

import pytest

from src.analysis.ssq_d8_b35_support import build_diversified_portfolio_v2
from src.analysis.ssq_top20_layered_coverage_v331 import (
    FIXED_TICKET_COUNT,
    build_top20_layered_coverage_v331,
)


def _probabilities() -> tuple[list[float], list[float]]:
    return (
        [float(34 - ball) for ball in range(1, 34)],
        [float(17 - ball) for ball in range(1, 17)],
    )


def _fixture() -> tuple[list[float], list[float], dict[str, object]]:
    red, blue = _probabilities()
    return red, blue, build_diversified_portfolio_v2(red, blue)


def _tickets(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document["tickets"])


def _valid_b35_with_ticket(ticket: dict[str, object]) -> dict[str, object]:
    """生成 35 张唯一合法完整票，并特意封锁给定的 Top1 蓝候选。"""

    blocked = [
        {"red": cast(list[int], ticket["red"]), "blue": cast(int, ticket["blue"])}
    ]
    for red in combinations(range(21, 34), 6):
        if len(blocked) == 35:
            break
        blocked.append({"red": list(red), "blue": 2})
    return {"groups": [{"expandedTickets": blocked}]}


def test_v331_constructs_exact_35_unique_tickets_and_exact_layered_exposure() -> None:
    red, blue, b35 = _fixture()

    document = build_top20_layered_coverage_v331(red, blue, b35)

    tickets = _tickets(document)
    top20 = cast(list[int], document["top20RedRanking"])
    targets = {
        ball: 12 if rank <= 8 else 10 if rank <= 14 else 9
        for rank, ball in enumerate(top20, start=1)
    }
    exposure = Counter(
        ball for ticket in tickets for ball in cast(list[int], ticket["red"])
    )
    ticket_set = {
        (tuple(cast(list[int], ticket["red"])), cast(int, ticket["blue"]))
        for ticket in tickets
    }
    audit = cast(dict[str, object], document["audit"])

    assert document["schemaVersion"] == "ssq_top20_layered_coverage_v331"
    assert document["fixedCostYuan"] == 70
    assert len(tickets) == len(ticket_set) == FIXED_TICKET_COUNT
    assert all(ticket["blue"] == document["blueTop1"] for ticket in tickets)
    assert exposure == Counter(targets)
    assert audit["top20Exposure"] == {str(ball): targets[ball] for ball in top20}
    assert audit["top20ExposureTargets"] == {str(ball): targets[ball] for ball in top20}
    assert audit["overlapWithB35"] == 0
    assert audit["combinedUniqueTicketCount"] == 70
    assert audit["candidateGeneration"] == "fixed_rank_rotation_remaining_exposure"
    assert cast(int, audit["candidateCountGenerated"]) <= 35 * 48
    assert audit["ticketSha256"]
    assert audit["b35TicketsSha256"]
    assert document["documentSha256"]


def test_v331_is_deterministic_and_audits_top20_b35_and_hashes() -> None:
    red, blue, b35 = _fixture()

    first = build_top20_layered_coverage_v331(red, blue, b35)
    second = build_top20_layered_coverage_v331(red, blue, b35)

    assert first == second
    assert first["top20RedRanking"] == list(range(1, 21))
    assert first["blueTop1"] == 1
    assert first["researchOnly"] is True
    assert first["predictionClaim"] is False
    assert first["fixedCliTuning"] is False
    audit = cast(dict[str, object], first["audit"])
    assert audit["selectionScoreSha256"]
    assert audit["redProbabilitiesSha256"]
    assert audit["blueProbabilitiesSha256"]
    assert audit["candidateTraceSha256"]


def test_v331_speed_smoke_constructs_100_documents_without_combination_enumeration() -> (
    None
):
    red, blue, b35 = _fixture()

    started_at = perf_counter()
    documents = [build_top20_layered_coverage_v331(red, blue, b35) for _ in range(100)]
    elapsed_seconds = perf_counter() - started_at

    assert len(documents) == 100
    # 常数规模候选生成；给 CI 普通开发机保留足够余量。
    assert elapsed_seconds < 3.0


def test_v331_skips_a_valid_complete_b35_ticket_without_relaxing_exposure() -> None:
    red, blue, b35 = _fixture()
    baseline = build_top20_layered_coverage_v331(red, blue, b35)
    blocked_b35 = _valid_b35_with_ticket(_tickets(baseline)[0])

    document = build_top20_layered_coverage_v331(red, blue, blocked_b35)

    audit = cast(dict[str, object], document["audit"])
    assert audit["overlapWithB35"] == 0
    assert cast(int, audit["candidateCountConsidered"]) > FIXED_TICKET_COUNT


def test_v331_fails_closed_for_invalid_b35() -> None:
    red, blue, _ = _fixture()

    with pytest.raises(ValueError, match="B35必须恰好35张唯一票"):
        build_top20_layered_coverage_v331(red, blue, {"groups": []})

    # 重复完整票必须在 B35 解析阶段失败关闭，不能被静默去重后继续构造。
    blocked = {
        "groups": [
            {
                "expandedTickets": [
                    {"red": [1, 2, 3, 4, 5, 6], "blue": 1} for _ in range(35)
                ]
            }
        ]
    }
    with pytest.raises(ValueError):
        build_top20_layered_coverage_v331(red, blue, blocked)
