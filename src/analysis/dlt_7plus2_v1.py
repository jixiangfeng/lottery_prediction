# -*- coding: utf-8 -*-
"""大乐透固定 7+2、21 注研究构造 v1。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Real
from typing import cast

from src.lotteries.dlt import DLT_RULE

SCHEMA_VERSION = "dlt_7plus2_v1"
FRONT_UNIVERSE_SIZE = 35
BACK_UNIVERSE_SIZE = 12
FRONT_POOL_SIZE = 7
BACK_POOL_SIZE = 2
EXPANDED_TICKET_COUNT = 21
FIXED_COST_MULTIPLIER = 21
FIXED_BASIC_COST_YUAN = 42
SPLIT_COUNTS: dict[str, int] = {
    "warmup": 600,
    "search": 1301,
    "validation": 500,
    "frozen": 500,
}
CANDIDATE_IDS = (
    "C1_LONG_RIDGE",
    "C2_MULTISCALE_RIDGE",
    "C3_PAIR_GRAPH_RIDGE",
    "C4_EQUAL_LOGPOOL",
)

PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "game": {"front": "35_choose_5", "back": "12_choose_2"},
    "splits": {
        "counts": SPLIT_COUNTS,
        "total": 2901,
        "rowRangesInclusive": {
            "warmup": [1, 600],
            "search": [601, 1901],
            "validation": [1902, 2401],
            "frozen": [2402, 2901],
        },
    },
    "candidateSet": {"count": 4, "ids": list(CANDIDATE_IDS)},
    "distribution": {
        "family": "fixed_cardinality_additive_score",
        "temperatureGrid": [0.5, 0.75, 1.0, 1.5, 2.0],
        "uniformMixtureGrid": [0.0, 0.05, 0.1, 0.2],
        "normalizer": "elementary_symmetric_log_dp",
        "zonesIndependentProduct": True,
    },
    "construction": {
        "front": "top7_marginals",
        "back": "top2_marginals",
        "rankingTieBreak": "score_descending_then_number_ascending",
        "expansion": "lexicographic_all_C(7,5)_with_shared_back2",
        "ticketCount": 21,
        "basicCostYuan": 42,
        "additionalBet": False,
    },
    "claims": {
        "researchOnly": True,
        "predictionClaim": False,
        "equalChanceNoEdge": True,
        "formalRecommendationStatus": "uniform_abstain",
    },
}

# 由上述规范 JSON（sort_keys、紧凑分隔符、UTF-8）冻结；修改协议必须新建版本。
FROZEN_PROTOCOL_SHA256 = (
    "8d4fd071f6036bfb298332f90ae2413a5349a8520a46b146587bafb13ecada4e"
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def protocol_sha256() -> str:
    """返回当前协议规范内容的 SHA-256。"""

    return _sha256_payload(PROTOCOL)


def _validate_marginals(
    values: Sequence[float], expected_length: int, expected_sum: int, zone: str
) -> tuple[float, ...]:
    if len(values) != expected_length:
        raise ValueError(f"{zone}边际概率必须恰含{expected_length}项")
    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{zone}边际概率第{index + 1}项必须为实数")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{zone}边际概率必须为[0, 1]内有限实数")
        normalized.append(probability)
    if not math.isclose(sum(normalized), expected_sum, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{zone}边际概率之和必须为{expected_sum}")
    return tuple(normalized)


def _top_numbers(probabilities: Sequence[float], count: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(1, len(probabilities) + 1),
            key=lambda number: (-probabilities[number - 1], number),
        )[:count]
    )


def _selection_payload(front: Sequence[int], back: Sequence[int]) -> dict[str, object]:
    return {"front": list(front), "back": list(back)}


def _expanded_tickets(
    front: Sequence[int], back: Sequence[int]
) -> list[dict[str, list[int]]]:
    return [
        {"front": list(front5), "back": list(back)}
        for front5 in combinations(sorted(front), DLT_RULE.front_count)
    ]


def _build_unvalidated(
    front_marginals: Sequence[float], back_marginals: Sequence[float]
) -> dict[str, object]:
    front_probabilities = _validate_marginals(
        front_marginals, FRONT_UNIVERSE_SIZE, DLT_RULE.front_count, "前区"
    )
    back_probabilities = _validate_marginals(
        back_marginals, BACK_UNIVERSE_SIZE, DLT_RULE.back_count, "后区"
    )
    front_ranking = _top_numbers(front_probabilities, FRONT_POOL_SIZE)
    back_ranking = _top_numbers(back_probabilities, BACK_POOL_SIZE)
    front = tuple(sorted(front_ranking))
    back = tuple(sorted(back_ranking))
    tickets = _expanded_tickets(front, back)
    selection = _selection_payload(front, back)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "protocolSha256": protocol_sha256(),
        "researchOnly": True,
        "predictionClaim": False,
        "equalChanceNoEdge": True,
        "formalRecommendationStatus": "uniform_abstain",
        "fixedCostMultiplier": FIXED_COST_MULTIPLIER,
        "fixedBasicCostYuan": FIXED_BASIC_COST_YUAN,
        "frontRanking": list(front_ranking),
        "backRanking": list(back_ranking),
        "front": list(front),
        "back": list(back),
        "expandedTickets": tickets,
        "audit": {
            "distinctFrontPoolCount": len(set(front)),
            "distinctBackPoolCount": len(set(back)),
            "expandedNominalTicketCount": len(tickets),
            "expandedUniqueTicketCount": len(
                {(tuple(ticket["front"]), tuple(ticket["back"])) for ticket in tickets}
            ),
            "frontMarginalsSha256": _sha256_payload(list(front_probabilities)),
            "backMarginalsSha256": _sha256_payload(list(back_probabilities)),
            "selectionSha256": _sha256_payload(selection),
            "expandedTicketsSha256": _sha256_payload(tickets),
        },
    }


def build_dlt_7plus2_v1(
    front_marginals: Sequence[float], back_marginals: Sequence[float]
) -> dict[str, object]:
    """按边际概率确定性构造前区 Top7、后区 Top2 及全部21注。"""

    document = _build_unvalidated(front_marginals, back_marginals)
    validate_dlt_7plus2_v1(document)
    return document


def _strict_int_list(value: object, length: int, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field}必须恰含{length}个号码")
    if any(isinstance(number, bool) or not isinstance(number, int) for number in value):
        raise ValueError(f"{field}号码必须是整数且不得为bool")
    return tuple(cast(list[int], value))


def validate_dlt_7plus2_v1(
    document: Mapping[str, object],
    *,
    front_marginals: Sequence[float] | None = None,
    back_marginals: Sequence[float] | None = None,
) -> None:
    """严格验证研究边界、合法性、完整展开、成本与稳定哈希。"""

    if protocol_sha256() != FROZEN_PROTOCOL_SHA256:
        raise ValueError("运行时代码中的大乐透协议已偏离冻结哈希")
    if document.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("DLT 7+2 schemaVersion不匹配")
    if document.get("protocolSha256") != FROZEN_PROTOCOL_SHA256:
        raise ValueError("DLT 7+2 protocolSha256不匹配")
    if (
        document.get("researchOnly") is not True
        or document.get("predictionClaim") is not False
        or document.get("equalChanceNoEdge") is not True
        or document.get("formalRecommendationStatus") != "uniform_abstain"
    ):
        raise ValueError("DLT 7+2研究边界不匹配")
    if (
        document.get("fixedCostMultiplier") != FIXED_COST_MULTIPLIER
        or document.get("fixedBasicCostYuan") != FIXED_BASIC_COST_YUAN
    ):
        raise ValueError("DLT 7+2固定成本不匹配")

    front_ranking = _strict_int_list(
        document.get("frontRanking"), FRONT_POOL_SIZE, "frontRanking"
    )
    back_ranking = _strict_int_list(
        document.get("backRanking"), BACK_POOL_SIZE, "backRanking"
    )
    front = _strict_int_list(document.get("front"), FRONT_POOL_SIZE, "front")
    back = _strict_int_list(document.get("back"), BACK_POOL_SIZE, "back")
    if (
        len(set(front_ranking)) != FRONT_POOL_SIZE
        or tuple(sorted(front_ranking)) != front
    ):
        raise ValueError("frontRanking与升序唯一前区7号不一致")
    if len(set(back_ranking)) != BACK_POOL_SIZE or tuple(sorted(back_ranking)) != back:
        raise ValueError("backRanking与升序唯一后区2号不一致")

    expected_tickets = _expanded_tickets(front, back)
    tickets_raw = document.get("expandedTickets")
    if not isinstance(tickets_raw, list) or len(tickets_raw) != EXPANDED_TICKET_COUNT:
        raise ValueError("DLT 7+2必须恰好展开21注")
    actual_tickets: list[dict[str, list[int]]] = []
    unique_tickets: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for raw_ticket in tickets_raw:
        if not isinstance(raw_ticket, Mapping):
            raise ValueError("DLT展开票格式非法")
        ticket_front = _strict_int_list(
            raw_ticket.get("front"), DLT_RULE.front_count, "ticket.front"
        )
        ticket_back = _strict_int_list(
            raw_ticket.get("back"), DLT_RULE.back_count, "ticket.back"
        )
        normalized_front, normalized_back = DLT_RULE.validate_draw(
            ticket_front, ticket_back
        )
        actual_tickets.append(
            {"front": list(normalized_front), "back": list(normalized_back)}
        )
        unique_tickets.add((normalized_front, normalized_back))
    if (
        actual_tickets != expected_tickets
        or len(unique_tickets) != EXPANDED_TICKET_COUNT
    ):
        raise ValueError("DLT 7+2展开票必须按字典序完整且唯一")

    audit = document.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("DLT 7+2缺少审计字段")
    required_counts = {
        "distinctFrontPoolCount": FRONT_POOL_SIZE,
        "distinctBackPoolCount": BACK_POOL_SIZE,
        "expandedNominalTicketCount": EXPANDED_TICKET_COUNT,
        "expandedUniqueTicketCount": EXPANDED_TICKET_COUNT,
    }
    for key, expected in required_counts.items():
        if audit.get(key) != expected:
            raise ValueError(f"DLT 7+2审计字段不匹配：{key}")
    if audit.get("selectionSha256") != _sha256_payload(_selection_payload(front, back)):
        raise ValueError("DLT 7+2 selectionSha256不匹配")
    if audit.get("expandedTicketsSha256") != _sha256_payload(expected_tickets):
        raise ValueError("DLT 7+2 expandedTicketsSha256不匹配")

    if (front_marginals is None) != (back_marginals is None):
        raise ValueError("前后区边际概率绑定参数必须同时提供")
    if front_marginals is not None and back_marginals is not None:
        rebuilt = _build_unvalidated(front_marginals, back_marginals)
        if dict(document) != rebuilt:
            raise ValueError("DLT 7+2与绑定边际概率的确定性构造不一致")


__all__ = [
    "BACK_POOL_SIZE",
    "CANDIDATE_IDS",
    "EXPANDED_TICKET_COUNT",
    "FIXED_BASIC_COST_YUAN",
    "FIXED_COST_MULTIPLIER",
    "FRONT_POOL_SIZE",
    "FROZEN_PROTOCOL_SHA256",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "SPLIT_COUNTS",
    "build_dlt_7plus2_v1",
    "protocol_sha256",
    "validate_dlt_7plus2_v1",
]
