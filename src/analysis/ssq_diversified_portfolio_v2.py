# -*- coding: utf-8 -*-
"""双色球覆盖优先分散组合挑战器 v2 的固定构造与审计。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import cast

from src.lotteries.ssq import SSQ_RULE

SCHEMA_VERSION = "ssq_diversified_portfolio_v2"
GROUP_COUNT = 5
RED_PER_GROUP = 7
BLUE_COUNT = 5
EXPANDED_TICKET_COUNT = 35
FIXED_COST_MULTIPLIER = 35

PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "research_only_coverage_first_diversified_portfolio_challenger",
    "probabilityInput": {
        "red": "current_ssq_ensemble_v1_red_marginal_probabilities",
        "blue": "current_ssq_ensemble_v1_blue_probabilities",
        "ranking": "probability_descending_then_ball_ascending",
    },
    "redConstruction": {
        "orderedSlots": (
            "all 33 legal red balls exactly once in model rank order, followed by "
            "one extra copy of rank1 and one extra copy of rank2"
        ),
        "layout": (
            "fixed seven-row five-column snake: even rows groups 1..5, odd rows "
            "groups 5..1; each ordered slot is appended to its addressed group"
        ),
        "localOptimization": False,
        "coveragePriority": "lexicographic_hard_constraints_before_any_model_score",
        "hardConstraints": {
            "distinctRed7Groups": 5,
            "redUnion": 33,
            "maximumRedExposure": 2,
            "maximumPairwiseIntersection": 3,
            "uniqueLegalRedsPerGroup": 7,
        },
    },
    "blueConstruction": {
        "selection": "top5_distinct",
        "assignment": "blue probability rank i is assigned to red group i",
    },
    "expansion": {
        "method": "each red7 group expands to its seven red6 subsets with assigned blue",
        "uniqueTickets": EXPANDED_TICKET_COUNT,
        "fixedCostMultiplier": FIXED_COST_MULTIPLIER,
    },
    "claims": {
        "researchOnly": True,
        "equalChanceNoEdge": True,
        "formalRecommendation": False,
    },
    "fixed": {
        "cliTuning": False,
        "parameterSearch": False,
        "iterationOrder": "single deterministic construction; no swaps",
    },
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def protocol_sha256() -> str:
    """返回固定挑战协议的稳定 SHA-256。"""

    return _sha256_payload(PROTOCOL)


def ranked_balls(probabilities: Sequence[float], upper: int) -> tuple[int, ...]:
    """按概率降序、球号升序返回完整合法排名。"""

    if len(probabilities) != upper:
        raise ValueError(f"概率数量必须恰好为{upper}")
    if any(not isinstance(value, (int, float)) for value in probabilities):
        raise ValueError("概率必须全部为数值")
    return tuple(
        sorted(range(1, upper + 1), key=lambda ball: (-probabilities[ball - 1], ball))
    )


def _snake_group_indexes() -> tuple[int, ...]:
    indexes: list[int] = []
    for row in range(RED_PER_GROUP):
        row_indexes = (
            range(GROUP_COUNT) if row % 2 == 0 else range(GROUP_COUNT - 1, -1, -1)
        )
        indexes.extend(row_indexes)
    return tuple(indexes)


SNAKE_GROUP_INDEXES = _snake_group_indexes()


def _expand_groups(
    red_groups: Sequence[Sequence[int]], blues: Sequence[int]
) -> tuple[tuple[tuple[int, ...], int], ...]:
    expanded = tuple(
        (tuple(red6), blue)
        for red7, blue in zip(red_groups, blues)
        for red6 in combinations(red7, 6)
    )
    if len(expanded) != EXPANDED_TICKET_COUNT or len(set(expanded)) != len(expanded):
        raise ValueError("分散组合必须形成35注全局唯一红6+蓝票")
    return expanded


def build_diversified_portfolio_v2(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> dict[str, object]:
    """由当前前序概率确定性构造覆盖优先 B 组合并立即严格审计。"""

    red_ranking = ranked_balls(red_probabilities, 33)
    blue_ranking = ranked_balls(blue_probabilities, 16)
    ordered_slots = (*red_ranking, red_ranking[0], red_ranking[1])
    mutable_groups: list[list[int]] = [[] for _ in range(GROUP_COUNT)]
    for ball, group_index in zip(ordered_slots, SNAKE_GROUP_INDEXES):
        mutable_groups[group_index].append(ball)
    red_groups = tuple(tuple(sorted(group)) for group in mutable_groups)
    blues = blue_ranking[:BLUE_COUNT]
    expanded = _expand_groups(red_groups, blues)
    exposures = Counter(ball for group in red_groups for ball in group)
    pairwise = [
        len(set(red_groups[left]).intersection(red_groups[right]))
        for left in range(GROUP_COUNT)
        for right in range(left + 1, GROUP_COUNT)
    ]
    portfolio: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "protocolSha256": protocol_sha256(),
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": "uniform_abstain",
        "equalChanceNoEdge": True,
        "fixedCostMultiplier": FIXED_COST_MULTIPLIER,
        "redRanking": list(red_ranking),
        "blueTop5Ranking": list(blues),
        "orderedRedSlots": list(ordered_slots),
        "snakeGroupIndexesZeroBased": list(SNAKE_GROUP_INDEXES),
        "groups": [
            {
                "group": index,
                "red": list(red7),
                "blue": blue,
                "expandedTickets": [
                    {"red": list(red6), "blue": ticket_blue}
                    for red6, ticket_blue in expanded
                    if ticket_blue == blue
                ],
            }
            for index, (red7, blue) in enumerate(zip(red_groups, blues), start=1)
        ],
        "audit": {
            "redUnionCount": len(exposures),
            "maximumRedExposure": max(exposures.values()),
            "duplicatedReds": sorted(
                ball for ball, exposure in exposures.items() if exposure == 2
            ),
            "maximumPairwiseIntersection": max(pairwise),
            "pairwiseIntersections": pairwise,
            "distinctRed7Groups": len(set(red_groups)),
            "distinctBlues": len(set(blues)),
            "expandedUniqueTicketCount": len(set(expanded)),
            "redProbabilitiesSha256": _sha256_payload(list(red_probabilities)),
            "blueProbabilitiesSha256": _sha256_payload(list(blue_probabilities)),
        },
    }
    validate_diversified_portfolio_v2(
        portfolio,
        red_probabilities=red_probabilities,
        blue_probabilities=blue_probabilities,
    )
    return portfolio


def validate_diversified_portfolio_v2(
    portfolio: Mapping[str, object],
    *,
    red_probabilities: Sequence[float] | None = None,
    blue_probabilities: Sequence[float] | None = None,
) -> None:
    """失败即关闭：验证协议、排名、结构、展开票与可选概率绑定。"""

    if portfolio.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("分散组合schemaVersion不匹配")
    if portfolio.get("protocolSha256") != protocol_sha256():
        raise ValueError("分散组合protocolSha256不匹配")
    if portfolio.get("fixedCostMultiplier") != FIXED_COST_MULTIPLIER:
        raise ValueError("分散组合固定成本倍数不匹配")
    if (
        portfolio.get("researchOnly") is not True
        or portfolio.get("predictionClaim") is not False
    ):
        raise ValueError("分散组合研究边界不匹配")
    groups_raw = portfolio.get("groups")
    if not isinstance(groups_raw, list) or len(groups_raw) != GROUP_COUNT:
        raise ValueError("分散组合必须恰好包含5组")
    red_groups: list[tuple[int, ...]] = []
    blues: list[int] = []
    expanded: list[tuple[tuple[int, ...], int]] = []
    for expected_index, group_raw in enumerate(groups_raw, start=1):
        if (
            not isinstance(group_raw, Mapping)
            or group_raw.get("group") != expected_index
        ):
            raise ValueError("分散组合组号非法")
        red_raw = group_raw.get("red")
        blue_raw = group_raw.get("blue")
        tickets_raw = group_raw.get("expandedTickets")
        if not isinstance(red_raw, list) or not isinstance(blue_raw, int):
            raise ValueError("分散组合红蓝字段类型非法")
        red7 = tuple(cast(list[int], red_raw))
        if (
            len(red7) != RED_PER_GROUP
            or tuple(sorted(red7)) != red7
            or len(set(red7)) != RED_PER_GROUP
        ):
            raise ValueError("每组必须包含7个合法升序互异红球")
        if not all(1 <= ball <= 33 for ball in red7) or not 1 <= blue_raw <= 16:
            raise ValueError("分散组合号码越界")
        if not isinstance(tickets_raw, list) or len(tickets_raw) != RED_PER_GROUP:
            raise ValueError("每组必须恰好展开7注")
        expected_tickets = {(tuple(red6), blue_raw) for red6 in combinations(red7, 6)}
        actual_tickets: set[tuple[tuple[int, ...], int]] = set()
        for ticket_raw in tickets_raw:
            if not isinstance(ticket_raw, Mapping):
                raise ValueError("展开票必须是对象")
            ticket_red_raw = ticket_raw.get("red")
            ticket_blue = ticket_raw.get("blue")
            if not isinstance(ticket_red_raw, list) or not isinstance(ticket_blue, int):
                raise ValueError("展开票红蓝字段类型非法")
            ticket_red = tuple(cast(list[int], ticket_red_raw))
            SSQ_RULE.validate_draw(ticket_red, ticket_blue)
            actual_tickets.add((ticket_red, ticket_blue))
        if actual_tickets != expected_tickets:
            raise ValueError("展开票与所属红7及蓝球不一致")
        red_groups.append(red7)
        blues.append(blue_raw)
        expanded.extend(sorted(actual_tickets))
    if len(set(red_groups)) != GROUP_COUNT:
        raise ValueError("分散组合红7组必须互异")
    if len(set(blues)) != BLUE_COUNT:
        raise ValueError("分散组合蓝球必须为5个互异号码")
    exposures = Counter(ball for group in red_groups for ball in group)
    if set(exposures) != set(range(1, 34)) or len(exposures) != 33:
        raise ValueError("分散组合红球并集必须恰好覆盖33球")
    if sorted(exposures.values()).count(2) != 2 or max(exposures.values()) != 2:
        raise ValueError("分散组合必须且只能重复两个红球一次")
    pairwise = [
        len(set(red_groups[left]).intersection(red_groups[right]))
        for left in range(GROUP_COUNT)
        for right in range(left + 1, GROUP_COUNT)
    ]
    if max(pairwise) > 3:
        raise ValueError("分散组合任意两组红球交集不得超过3")
    if (
        len(expanded) != EXPANDED_TICKET_COUNT
        or len(set(expanded)) != EXPANDED_TICKET_COUNT
    ):
        raise ValueError("分散组合必须恰好形成35注唯一票")
    if red_probabilities is not None or blue_probabilities is not None:
        if red_probabilities is None or blue_probabilities is None:
            raise ValueError("概率绑定必须同时提供红球与蓝球概率")
        expected = build_diversified_portfolio_v2_unchecked(
            red_probabilities, blue_probabilities
        )
        if dict(portfolio) != expected:
            raise ValueError("分散组合与当前概率的固定构造不一致")


def build_diversified_portfolio_v2_unchecked(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> dict[str, object]:
    """仅供审计重建，避免公开构造器递归调用验证。"""

    red_ranking = ranked_balls(red_probabilities, 33)
    blue_ranking = ranked_balls(blue_probabilities, 16)
    ordered_slots = (*red_ranking, red_ranking[0], red_ranking[1])
    mutable_groups: list[list[int]] = [[] for _ in range(GROUP_COUNT)]
    for ball, group_index in zip(ordered_slots, SNAKE_GROUP_INDEXES):
        mutable_groups[group_index].append(ball)
    red_groups = tuple(tuple(sorted(group)) for group in mutable_groups)
    blues = blue_ranking[:BLUE_COUNT]
    expanded = _expand_groups(red_groups, blues)
    exposures = Counter(ball for group in red_groups for ball in group)
    pairwise = [
        len(set(red_groups[left]).intersection(red_groups[right]))
        for left in range(GROUP_COUNT)
        for right in range(left + 1, GROUP_COUNT)
    ]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "protocolSha256": protocol_sha256(),
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": "uniform_abstain",
        "equalChanceNoEdge": True,
        "fixedCostMultiplier": FIXED_COST_MULTIPLIER,
        "redRanking": list(red_ranking),
        "blueTop5Ranking": list(blues),
        "orderedRedSlots": list(ordered_slots),
        "snakeGroupIndexesZeroBased": list(SNAKE_GROUP_INDEXES),
        "groups": [
            {
                "group": index,
                "red": list(red7),
                "blue": blue,
                "expandedTickets": [
                    {"red": list(red6), "blue": ticket_blue}
                    for red6, ticket_blue in expanded
                    if ticket_blue == blue
                ],
            }
            for index, (red7, blue) in enumerate(zip(red_groups, blues), start=1)
        ],
        "audit": {
            "redUnionCount": len(exposures),
            "maximumRedExposure": max(exposures.values()),
            "duplicatedReds": sorted(
                ball for ball, exposure in exposures.items() if exposure == 2
            ),
            "maximumPairwiseIntersection": max(pairwise),
            "pairwiseIntersections": pairwise,
            "distinctRed7Groups": len(set(red_groups)),
            "distinctBlues": len(set(blues)),
            "expandedUniqueTicketCount": len(set(expanded)),
            "redProbabilitiesSha256": _sha256_payload(list(red_probabilities)),
            "blueProbabilitiesSha256": _sha256_payload(list(blue_probabilities)),
        },
    }


__all__ = [
    "EXPANDED_TICKET_COUNT",
    "FIXED_COST_MULTIPLIER",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "SNAKE_GROUP_INDEXES",
    "build_diversified_portfolio_v2",
    "protocol_sha256",
    "ranked_balls",
    "validate_diversified_portfolio_v2",
]
