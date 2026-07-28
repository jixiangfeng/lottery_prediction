# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import cast

import pytest

from src.analysis.ssq_8red1blue_v1_history import (
    CONTROL_COUNT,
    METRIC_NAMES,
    build_matched_random_control,
    evaluate_full_history,
    walk_forward_d8_fingerprints,
)
from src.analysis.ssq_diversified_portfolio_v2 import build_diversified_portfolio_v2
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
    validate_small_compound_8red1blue_v1,
)
from src.lotteries.ssq import SSQ_RULE


def _draws(count: int) -> list[SSQDraw]:
    draws: list[SSQDraw] = []
    for index in range(count):
        red = tuple(sorted(((index + offset * 5) % 33) + 1 for offset in range(6)))
        normalized_red, blue = SSQ_RULE.validate_draw(red, index % 16 + 1)
        draws.append(
            SSQDraw(
                issue=str(2024001 + index),
                draw_date=f"2024-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
                red=normalized_red,
                blue=blue,
                source_url="fixture",
                raw_hash=f"{index:064x}",
                raw={},
            )
        )
    return draws


def test_builder_is_deterministic_exact_cost_and_zero_overlap_with_b() -> None:
    red = [float(34 - ball) for ball in range(1, 34)]
    blue = [float(17 - ball) for ball in range(1, 17)]
    b = build_diversified_portfolio_v2(red, blue)

    first = build_small_compound_8red1blue_v1(red, blue, b)
    second = build_small_compound_8red1blue_v1(red, blue, b)

    assert first == second
    assert len(first["red"]) == len(set(cast(list[int], first["red"]))) == 8
    assert first["blue"] == 1
    assert len(first["expandedTickets"]) == 28
    assert first["fixedCostMultiplier"] == 28
    assert first["audit"]["overlapWithB"] == 0
    assert first["audit"]["combinedNominalTicketCount"] == 63
    assert first["audit"]["combinedUniqueTicketCount"] == 63
    validate_small_compound_8red1blue_v1(
        first,
        red_probabilities=red,
        blue_probabilities=blue,
        diversified_portfolio=b,
    )


def test_builder_fails_closed_when_b_covers_every_candidate_ticket() -> None:
    red = [float(34 - ball) for ball in range(1, 34)]
    blue = [float(17 - ball) for ball in range(1, 17)]
    b = build_diversified_portfolio_v2(red, blue)
    groups = cast(list[dict[str, object]], b["groups"])
    all_candidate_tickets = []
    for red8 in __import__("itertools").combinations(range(1, 13), 8):
        for red6 in __import__("itertools").combinations(red8, 6):
            all_candidate_tickets.append({"red": list(red6), "blue": 1})
    groups[0]["expandedTickets"] = all_candidate_tickets
    with pytest.raises(ValueError, match="B组必须展开为7注"):
        build_small_compound_8red1blue_v1(red, blue, b)


def test_random_controls_are_deterministic_legal_and_matched_cost() -> None:
    first = build_matched_random_control("2026086", 0)
    assert first == build_matched_random_control("2026086", 0)
    assert first != build_matched_random_control("2026086", 1)
    assert len(first.red8) == len(set(first.red8)) == 8
    assert len(first.tickets) == len(set(first.tickets)) == 28
    for red6, blue in first.tickets:
        SSQ_RULE.validate_draw(red6, blue)


def test_history_is_predict_before_update_and_records_full_contract() -> None:
    report = evaluate_full_history(_draws(123))
    history = cast(dict[str, object], report["history"])
    summary = cast(dict[str, dict[str, object]], report["summary"])
    per_issue = cast(list[dict[str, object]], report["perIssue"])

    assert history["warmupPeriods"] == 120
    assert history["evaluatedPeriods"] == 3
    assert len(per_issue) == 3
    assert summary["D8"]["observations"] == 3
    assert summary["C32"]["observations"] == 3 * CONTROL_COUNT
    assert set(cast(dict[str, object], report["D8MinusC32Mean"])) == set(METRIC_NAMES)
    for item in per_issue:
        d8 = cast(dict[str, object], item["D8"])
        audit = cast(dict[str, object], d8["audit"])
        assert audit == {
            "overlapWithB": 0,
            "combinedNominalTicketCount": 63,
            "combinedUniqueTicketCount": 63,
        }


def test_history_prefix_invariance() -> None:
    prefix = _draws(124)
    future = _draws(2)
    extended = [
        *prefix,
        *[
            SSQDraw(
                issue=str(2030000 + index),
                draw_date=draw.draw_date,
                red=draw.red,
                blue=draw.blue,
                source_url=draw.source_url,
                raw_hash=draw.raw_hash,
                raw=draw.raw,
            )
            for index, draw in enumerate(future)
        ],
    ]
    assert walk_forward_d8_fingerprints(extended)[: len(prefix)] == (
        walk_forward_d8_fingerprints(prefix)
    )
