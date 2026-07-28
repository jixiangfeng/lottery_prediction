# -*- coding: utf-8 -*-
"""双色球独立8红+1蓝研究影子 v1 的固定构造与审计。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import cast

from src.analysis.ssq_diversified_portfolio_v2 import ranked_balls
from src.lotteries.ssq import SSQ_RULE

SCHEMA_VERSION = "ssq_small_compound_8red1blue_v1"
TOP_RED_POOL_SIZE = 12
RED_GROUP_SIZE = 8
EXPANDED_TICKET_COUNT = 28
FIXED_COST_MULTIPLIER = 28
COMBINED_WITH_B_COUNT = 63
EPSILON = 1e-12

PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "independent_research_shadow_only",
    "probabilityInput": {
        "red": "current_ssq_ensemble_v1_red_marginal_probabilities",
        "blue": "current_ssq_ensemble_v1_blue_probabilities",
        "ranking": "probability_descending_then_ball_ascending",
    },
    "candidateConstruction": {
        "pool": "top12 ranked reds",
        "candidates": "all C(12,8)",
        "score": "sum(log(clipped red marginal probability))",
        "order": "score descending then lexicographic red tuple ascending",
        "selection": "first candidate with zero full-ticket overlap against B35",
        "failure": "fail_closed_if_none",
    },
    "blue": "model_top1_probability_descending_then_ball_ascending",
    "expansion": "all C(8,6)=28 red6 tickets with the shared Top1 blue",
    "claims": {
        "researchOnly": True,
        "predictionClaim": False,
        "equalChanceNoEdge": True,
        "formalRecommendationStatus": "uniform_abstain",
    },
    "fixed": {"cliTuning": False, "fixedCostMultiplier": 28},
}


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
    """返回固定构造协议摘要。"""

    return _sha256_payload(PROTOCOL)


def _expanded_b_tickets(
    diversified_portfolio: Mapping[str, object],
) -> set[tuple[tuple[int, ...], int]]:
    groups = diversified_portfolio.get("groups")
    if not isinstance(groups, list) or len(groups) != 5:
        raise ValueError("D8构造要求已审计的既有B35组合")
    tickets: set[tuple[tuple[int, ...], int]] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("B组格式非法")
        expanded = group.get("expandedTickets")
        if not isinstance(expanded, list) or len(expanded) != 7:
            raise ValueError("B组必须展开为7注")
        for ticket in expanded:
            if not isinstance(ticket, Mapping):
                raise ValueError("B展开票格式非法")
            red_raw = ticket.get("red")
            blue = ticket.get("blue")
            if not isinstance(red_raw, list) or not isinstance(blue, int):
                raise ValueError("B展开票红蓝字段非法")
            red, normalized_blue = SSQ_RULE.validate_draw(red_raw, blue)
            tickets.add((red, normalized_blue))
    if len(tickets) != 35:
        raise ValueError("B组合必须恰好形成35注唯一票")
    return tickets


def _candidate_order(
    red_probabilities: Sequence[float],
) -> tuple[tuple[int, ...], list[tuple[tuple[int, ...], float]]]:
    ranking = ranked_balls(red_probabilities, 33)
    top12 = tuple(ranking[:TOP_RED_POOL_SIZE])
    candidates = [
        (
            tuple(sorted(candidate)),
            sum(
                math.log(max(EPSILON, float(red_probabilities[ball - 1])))
                for ball in candidate
            ),
        )
        for candidate in combinations(top12, RED_GROUP_SIZE)
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return top12, candidates


def _build_unvalidated(
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    diversified_portfolio: Mapping[str, object],
) -> dict[str, object]:
    if len(red_probabilities) != 33 or len(blue_probabilities) != 16:
        raise ValueError("D8概率向量长度非法")
    if any(
        not math.isfinite(float(value)) or float(value) < 0
        for value in red_probabilities
    ):
        raise ValueError("D8红球概率必须为有限非负数")
    if any(
        not math.isfinite(float(value)) or float(value) < 0
        for value in blue_probabilities
    ):
        raise ValueError("D8蓝球概率必须为有限非负数")
    b_tickets = _expanded_b_tickets(diversified_portfolio)
    blue = ranked_balls(blue_probabilities, 16)[0]
    top12, candidates = _candidate_order(red_probabilities)
    selected: tuple[int, ...] | None = None
    selected_score = 0.0
    selected_rank = 0
    selected_tickets: tuple[tuple[tuple[int, ...], int], ...] = ()
    for rank, (candidate, score) in enumerate(candidates, start=1):
        tickets = tuple((tuple(red6), blue) for red6 in combinations(candidate, 6))
        if not set(tickets).intersection(b_tickets):
            selected = candidate
            selected_score = score
            selected_rank = rank
            selected_tickets = tickets
            break
    if selected is None:
        raise ValueError("D8候选全集均与B35重叠，失败关闭")
    document: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "protocolSha256": protocol_sha256(),
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": "uniform_abstain",
        "equalChanceNoEdge": True,
        "fixedCostMultiplier": FIXED_COST_MULTIPLIER,
        "top12RedRanking": list(top12),
        "candidateCount": len(candidates),
        "selectedCandidateRank": selected_rank,
        "selectedCandidateLogScore": selected_score,
        "red": list(selected),
        "blue": blue,
        "expandedTickets": [
            {"red": list(red6), "blue": ticket_blue}
            for red6, ticket_blue in selected_tickets
        ],
        "audit": {
            "distinctRedCount": len(set(selected)),
            "distinctBlueCount": 1,
            "expandedNominalTicketCount": len(selected_tickets),
            "expandedUniqueTicketCount": len(set(selected_tickets)),
            "overlapWithB": len(set(selected_tickets).intersection(b_tickets)),
            "combinedNominalTicketCount": COMBINED_WITH_B_COUNT,
            "combinedUniqueTicketCount": len(set(selected_tickets).union(b_tickets)),
            "redProbabilitiesSha256": _sha256_payload(list(red_probabilities)),
            "blueProbabilitiesSha256": _sha256_payload(list(blue_probabilities)),
            "diversifiedPortfolioSha256": _sha256_payload(diversified_portfolio),
        },
    }
    return document


def build_small_compound_8red1blue_v1(
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    diversified_portfolio: Mapping[str, object],
) -> dict[str, object]:
    """确定性选择首个与B35零重叠的8红+Top1蓝研究影子。"""

    document = _build_unvalidated(
        red_probabilities, blue_probabilities, diversified_portfolio
    )
    validate_small_compound_8red1blue_v1(document)
    return document


def validate_small_compound_8red1blue_v1(
    document: Mapping[str, object],
    *,
    red_probabilities: Sequence[float] | None = None,
    blue_probabilities: Sequence[float] | None = None,
    diversified_portfolio: Mapping[str, object] | None = None,
) -> None:
    """严格验证D8结构、成本、零重叠与可选概率重建绑定。"""

    if document.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("D8 schemaVersion不匹配")
    if document.get("protocolSha256") != protocol_sha256():
        raise ValueError("D8 protocolSha256不匹配")
    if document.get("fixedCostMultiplier") != FIXED_COST_MULTIPLIER:
        raise ValueError("D8固定成本倍数不匹配")
    if (
        document.get("researchOnly") is not True
        or document.get("predictionClaim") is not False
        or document.get("equalChanceNoEdge") is not True
        or document.get("formalRecommendationStatus") != "uniform_abstain"
    ):
        raise ValueError("D8研究边界不匹配")
    red_raw = document.get("red")
    blue = document.get("blue")
    tickets_raw = document.get("expandedTickets")
    if not isinstance(red_raw, list) or not isinstance(blue, int):
        raise ValueError("D8红蓝字段非法")
    red8 = tuple(cast(list[int], red_raw))
    if (
        len(red8) != RED_GROUP_SIZE
        or tuple(sorted(red8)) != red8
        or len(set(red8)) != RED_GROUP_SIZE
    ):
        raise ValueError("D8必须包含恰好8个升序唯一红球")
    expected = {(tuple(red6), blue) for red6 in combinations(red8, 6)}
    if not isinstance(tickets_raw, list) or len(tickets_raw) != EXPANDED_TICKET_COUNT:
        raise ValueError("D8必须展开为28注")
    actual: set[tuple[tuple[int, ...], int]] = set()
    for ticket in tickets_raw:
        if not isinstance(ticket, Mapping):
            raise ValueError("D8展开票格式非法")
        ticket_red = ticket.get("red")
        ticket_blue = ticket.get("blue")
        if not isinstance(ticket_red, list) or not isinstance(ticket_blue, int):
            raise ValueError("D8展开票红蓝字段非法")
        normalized = SSQ_RULE.validate_draw(ticket_red, ticket_blue)
        actual.add(normalized)
    if actual != expected or len(actual) != EXPANDED_TICKET_COUNT:
        raise ValueError("D8展开票不完整或不唯一")
    audit = document.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("D8缺少审计")
    required_audit = {
        "distinctRedCount": 8,
        "distinctBlueCount": 1,
        "expandedNominalTicketCount": 28,
        "expandedUniqueTicketCount": 28,
        "overlapWithB": 0,
        "combinedNominalTicketCount": 63,
        "combinedUniqueTicketCount": 63,
    }
    for key, value in required_audit.items():
        if audit.get(key) != value:
            raise ValueError(f"D8审计字段不匹配：{key}")
    provided = (red_probabilities, blue_probabilities, diversified_portfolio)
    if any(value is not None for value in provided):
        if any(value is None for value in provided):
            raise ValueError("D8概率与B绑定参数必须同时提供")
        rebuilt = _build_unvalidated(
            cast(Sequence[float], red_probabilities),
            cast(Sequence[float], blue_probabilities),
            cast(Mapping[str, object], diversified_portfolio),
        )
        if dict(document) != rebuilt:
            raise ValueError("D8与当前概率及B组合的固定构造不一致")


__all__ = [
    "COMBINED_WITH_B_COUNT",
    "EXPANDED_TICKET_COUNT",
    "FIXED_COST_MULTIPLIER",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "build_small_compound_8red1blue_v1",
    "protocol_sha256",
    "validate_small_compound_8red1blue_v1",
]
