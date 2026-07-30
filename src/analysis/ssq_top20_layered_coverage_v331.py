# -*- coding: utf-8 -*-
"""独立 v3.3.1 Top20 快速分层循环区组构造。

本模块刻意不枚举 ``C(20, 6)``。每轮只生成固定的 48 个“剩余曝光优先、
排名轮转”候选，候选被 B35、重复票或余额约束拒绝即跳过；固定候选预算耗尽
即失败关闭，而不扩大搜索空间或创建任何状态。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from src.analysis.ssq_d8_b35_support import ranked_balls
from src.lotteries.ssq import SSQ_RULE

SCHEMA_VERSION = "ssq_top20_layered_coverage_v331"
TOP20_SIZE = 20
B35_TICKET_COUNT = 35
FIXED_TICKET_COUNT = 35
TICKET_PRICE_YUAN = 2
FIXED_COST_YUAN = FIXED_TICKET_COUNT * TICKET_PRICE_YUAN
RED_PER_TICKET = 6
MAX_CANDIDATES_PER_ROUND = 48

Ticket = tuple[tuple[int, ...], int]


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_probabilities(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> None:
    if len(red_probabilities) != 33 or len(blue_probabilities) != 16:
        raise ValueError("概率向量维度非法")
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in (*red_probabilities, *blue_probabilities)
    ):
        raise ValueError("概率必须是有限非负数")


def _b35_tickets(portfolio: Mapping[str, object]) -> set[Ticket]:
    """严格抽取恰好 35 张唯一合法的完整 B35 票。"""

    groups = portfolio.get("groups")
    if not isinstance(groups, list):
        raise ValueError("B35组合结构非法")
    tickets: set[Ticket] = set()
    raw_ticket_count = 0
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("B35组非法")
        raw_tickets = group.get("expandedTickets")
        if not isinstance(raw_tickets, list):
            raise ValueError("B35展开票非法")
        for raw_ticket in raw_tickets:
            raw_ticket_count += 1
            if not isinstance(raw_ticket, Mapping):
                raise ValueError("B35票非法")
            raw_red = raw_ticket.get("red")
            blue = raw_ticket.get("blue")
            if not isinstance(raw_red, list) or not isinstance(blue, int):
                raise ValueError("B35票红蓝字段非法")
            red = tuple(cast(list[int], raw_red))
            SSQ_RULE.validate_draw(red, blue)
            tickets.add((red, blue))
    if raw_ticket_count != B35_TICKET_COUNT or len(tickets) != B35_TICKET_COUNT:
        raise ValueError("B35必须恰好35张唯一票")
    return tickets


def _exposure_targets(top20_ranking: Sequence[int]) -> dict[int, int]:
    if len(top20_ranking) != TOP20_SIZE or len(set(top20_ranking)) != TOP20_SIZE:
        raise ValueError("Top20排名非法")
    targets = {
        ball: 12 if rank <= 8 else 10 if rank <= 14 else 9
        for rank, ball in enumerate(top20_ranking, start=1)
    }
    if sum(targets.values()) != FIXED_TICKET_COUNT * RED_PER_TICKET:
        raise ValueError("Top20分层曝光目标红球位总数非法")
    return targets


def _candidate_score(
    red: tuple[int, ...],
    rank_by_ball: Mapping[int, int],
    probabilities: Sequence[float],
) -> float:
    return sum(
        (TOP20_SIZE - rank_by_ball[ball] + 1) * float(probabilities[ball - 1])
        for ball in red
    )


def _rotating_candidate(
    top20_ranking: tuple[int, ...],
    remaining: Mapping[int, int],
    offset: int,
    stride: int,
) -> tuple[int, ...]:
    """从余额最高层取六球；仅在同余额层按固定排名轮转。

    这相当于一个有界 Havel--Hakimi 步骤：每次只消耗当前余额最大的六个
    红球，保证尚余总槽位与每球余额的基本可行性；``offset`` 只负责消除同层
    的固定偏置，绝不把低余额球提到高余额球之前。
    """

    rank_by_ball = {ball: rank for rank, ball in enumerate(top20_ranking)}
    ordered = sorted(
        top20_ranking,
        key=lambda ball: (
            -remaining[ball],
            (rank_by_ball[ball] * stride - offset) % TOP20_SIZE,
        ),
    )
    candidate = tuple(sorted(ordered[:RED_PER_TICKET]))
    if len(set(candidate)) != RED_PER_TICKET or any(
        remaining[ball] <= 0 for ball in candidate
    ):
        raise ValueError("固定轮转候选无法保持正剩余曝光")
    return candidate


def _select_tickets(
    top20_ranking: tuple[int, ...],
    red_probabilities: Sequence[float],
    blue_top1: int,
    blocked: set[Ticket],
) -> tuple[list[tuple[Ticket, float]], dict[int, int], list[dict[str, object]]]:
    """使用常数规模候选和固定回退构造精确曝光票集。"""

    targets = _exposure_targets(top20_ranking)
    rank_by_ball = {ball: rank for rank, ball in enumerate(top20_ranking, start=1)}
    remaining = dict(targets)
    selected: list[tuple[Ticket, float]] = []
    selected_tickets: set[Ticket] = set()
    trace: list[dict[str, object]] = []

    for round_index in range(FIXED_TICKET_COUNT):
        chosen: tuple[Ticket, float] | None = None
        rejected_blocked = 0
        rejected_duplicate = 0
        for candidate_index in range(MAX_CANDIDATES_PER_ROUND):
            # 47 个排名轮转候选；第 48 个是固定的零偏移/stride=1 回退。
            # 互素 stride 令同余额层产生多样的确定性区组，仍不改变
            # “剩余曝光优先”的主排序。
            if candidate_index == MAX_CANDIDATES_PER_ROUND - 1:
                candidate_kind = "fixedFallback"
                stride = 1
                offset = 0
            else:
                candidate_kind = "rankRotation"
                strides = (1, 3, 7, 9, 11, 13, 17, 19)
                stride = strides[candidate_index % len(strides)]
                phase = candidate_index // len(strides)
                offset = (round_index * RED_PER_TICKET + phase * 3) % TOP20_SIZE
            red = _rotating_candidate(top20_ranking, remaining, offset, stride)
            ticket = (red, blue_top1)
            if ticket in blocked:
                rejected_blocked += 1
                continue
            if ticket in selected_tickets:
                rejected_duplicate += 1
                continue
            chosen = (ticket, _candidate_score(red, rank_by_ball, red_probabilities))
            trace.append(
                {
                    "round": round_index + 1,
                    "candidateIndex": candidate_index + 1,
                    "candidateKind": candidate_kind,
                    "offset": offset,
                    "stride": stride,
                    "red": list(red),
                    "rejectedBlocked": rejected_blocked,
                    "rejectedDuplicate": rejected_duplicate,
                }
            )
            break
        if chosen is None:
            raise ValueError("固定候选预算耗尽，无法满足B35/唯一票约束")
        ticket, score = chosen
        if any(remaining[ball] <= 0 for ball in ticket[0]):
            raise ValueError("候选超过Top20剩余曝光目标")
        for ball in ticket[0]:
            remaining[ball] -= 1
        selected.append((ticket, score))
        selected_tickets.add(ticket)

    if any(remaining.values()):
        raise ValueError("固定轮转未能精确满足Top20分层曝光目标")
    if len(selected_tickets) != FIXED_TICKET_COUNT:
        raise ValueError("固定轮转未能形成35张唯一票")
    if selected_tickets.intersection(blocked):
        raise ValueError("固定轮转票与B35完整票重叠")
    return selected, targets, trace


def _audit(
    tickets_with_scores: Sequence[tuple[Ticket, float]],
    top20_ranking: tuple[int, ...],
    targets: Mapping[int, int],
    blocked: set[Ticket],
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    tickets = [ticket for ticket, _ in tickets_with_scores]
    ticket_set = set(tickets)
    exposure = Counter(ball for red, _ in tickets for ball in red)
    scores = [score for _, score in tickets_with_scores]
    candidates_considered = sum(cast(int, entry["candidateIndex"]) for entry in trace)
    return {
        "candidateGeneration": "fixed_rank_rotation_remaining_exposure",
        "maxCandidatesPerRound": MAX_CANDIDATES_PER_ROUND,
        "candidateCountBudget": FIXED_TICKET_COUNT * MAX_CANDIDATES_PER_ROUND,
        "candidateCountGenerated": candidates_considered,
        "candidateCountConsidered": candidates_considered,
        "candidateTraceSha256": _hash(list(trace)),
        "top20ExposureTargets": {str(ball): targets[ball] for ball in top20_ranking},
        "top20Exposure": {str(ball): exposure[ball] for ball in top20_ranking},
        "redExposureSlotCount": sum(exposure.values()),
        "expectedRedExposureSlotCount": FIXED_TICKET_COUNT * RED_PER_TICKET,
        "uniqueTicketCount": len(ticket_set),
        "ticketCount": len(tickets),
        "ticketPriceYuan": TICKET_PRICE_YUAN,
        "fixedCostYuan": FIXED_COST_YUAN,
        "overlapWithB35": len(ticket_set.intersection(blocked)),
        "combinedUniqueTicketCount": len(ticket_set.union(blocked)),
        "ticketSha256": _hash(sorted((list(red), blue) for red, blue in ticket_set)),
        "b35TicketsSha256": _hash(sorted((list(red), blue) for red, blue in blocked)),
        "redProbabilitiesSha256": _hash(list(red_probabilities)),
        "blueProbabilitiesSha256": _hash(list(blue_probabilities)),
        "selectionScoreSha256": _hash(scores),
    }


def build_top20_layered_coverage_v331(
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    b35: Mapping[str, object],
) -> dict[str, object]:
    """构造独立 v3.3.1 的 35 注/70 元快速 Top20 分层覆盖研究文档。"""

    _validate_probabilities(red_probabilities, blue_probabilities)
    top20_ranking = tuple(ranked_balls(red_probabilities, 33)[:TOP20_SIZE])
    blue_top1 = ranked_balls(blue_probabilities, 16)[0]
    blocked = _b35_tickets(b35)
    tickets_with_scores, targets, trace = _select_tickets(
        top20_ranking, red_probabilities, blue_top1, blocked
    )
    audit = _audit(
        tickets_with_scores,
        top20_ranking,
        targets,
        blocked,
        red_probabilities,
        blue_probabilities,
        trace,
    )
    if (
        audit["top20Exposure"] != audit["top20ExposureTargets"]
        or audit["uniqueTicketCount"] != FIXED_TICKET_COUNT
        or audit["ticketCount"] != FIXED_TICKET_COUNT
        or audit["fixedCostYuan"] != FIXED_COST_YUAN
        or audit["overlapWithB35"] != 0
        or audit["redExposureSlotCount"] != FIXED_TICKET_COUNT * RED_PER_TICKET
    ):
        raise ValueError("Top20快速分层覆盖审计失败")

    document: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": "uniform_abstain",
        "fixedCliTuning": False,
        "construction": "fixed_top20_rank_rotation_remaining_exposure_35x6plus1_blue_top1_bounded_candidates",
        "fixedCostMultiplier": FIXED_TICKET_COUNT,
        "ticketPriceYuan": TICKET_PRICE_YUAN,
        "fixedCostYuan": FIXED_COST_YUAN,
        "top20RedRanking": list(top20_ranking),
        "blueTop1": blue_top1,
        "top20": [
            {
                "rank": rank,
                "ball": ball,
                "probability": float(red_probabilities[ball - 1]),
                "exposureTarget": targets[ball],
            }
            for rank, ball in enumerate(top20_ranking, start=1)
        ],
        "tickets": [
            {
                "ticketIndex": index,
                "red": list(red),
                "blue": blue,
                "fixedRankProbabilityScore": score,
            }
            for index, ((red, blue), score) in enumerate(tickets_with_scores, start=1)
        ],
        "audit": audit,
    }
    document["documentSha256"] = _hash(document)
    return document


__all__ = [
    "B35_TICKET_COUNT",
    "FIXED_COST_YUAN",
    "FIXED_TICKET_COUNT",
    "MAX_CANDIDATES_PER_ROUND",
    "SCHEMA_VERSION",
    "TOP20_SIZE",
    "build_top20_layered_coverage_v331",
]
