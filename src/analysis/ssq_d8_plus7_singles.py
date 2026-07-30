# -*- coding: utf-8 -*-
"""保留方案：D8 8+1 加7张核心/边界单式（35注/70元）。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import cast

from src.analysis.ssq_d8_b35_support import build_diversified_portfolio_v2, ranked_balls
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)
from src.lotteries.ssq import SSQ_RULE

TOP14_SIZE = 14
CORE_SIZE = 8
SUPPLEMENT_COUNT = 7
TICKET_COUNT = 35
TICKET_PRICE_YUAN = 2
EPSILON = 1e-12
Ticket = tuple[tuple[int, ...], int]


def _expanded(document: Mapping[str, object]) -> set[Ticket]:
    raw = document.get("expandedTickets")
    if not isinstance(raw, list) or len(raw) != 28:
        raise ValueError("D8必须展开为28张票")
    tickets: set[Ticket] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("D8展开票非法")
        red, blue = item.get("red"), item.get("blue")
        if not isinstance(red, list) or not isinstance(blue, int):
            raise ValueError("D8展开票字段非法")
        tickets.add(SSQ_RULE.validate_draw(cast(list[int], red), blue))
    if len(tickets) != 28:
        raise ValueError("D8展开票必须唯一")
    return tickets


def _blocked(portfolio: Mapping[str, object]) -> set[Ticket]:
    groups = portfolio.get("groups")
    if not isinstance(groups, list) or len(groups) != 5:
        raise ValueError("B35结构非法")
    tickets: set[Ticket] = set()
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(
            group.get("expandedTickets"), list
        ):
            raise ValueError("B35组非法")
        for item in cast(list[object], group["expandedTickets"]):
            if not isinstance(item, Mapping):
                raise ValueError("B35票非法")
            red, blue = item.get("red"), item.get("blue")
            if not isinstance(red, list) or not isinstance(blue, int):
                raise ValueError("B35票字段非法")
            tickets.add(SSQ_RULE.validate_draw(cast(list[int], red), blue))
    if len(tickets) != 35:
        raise ValueError("B35必须为35张唯一票")
    return tickets


def build_d8_plus7_singles(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> dict[str, object]:
    """严格确定性地构造D8 28注加7张Top14边界单式。"""
    if len(red_probabilities) != 33 or len(blue_probabilities) != 16:
        raise ValueError("概率向量维度非法")
    b35 = build_diversified_portfolio_v2(red_probabilities, blue_probabilities)
    d8 = build_small_compound_8red1blue_v1(red_probabilities, blue_probabilities, b35)
    core_raw, blue = d8.get("red"), d8.get("blue")
    if not isinstance(core_raw, list) or not isinstance(blue, int):
        raise ValueError("D8核心或蓝球非法")
    core = tuple(cast(list[int], core_raw))
    top14 = tuple(ranked_balls(red_probabilities, 33)[:TOP14_SIZE])
    boundary = tuple(ball for ball in top14 if ball not in set(core))
    if len(core) != CORE_SIZE or not set(core).issubset(top14) or len(boundary) != 6:
        raise ValueError("D8核心与Top14边界关系非法")
    d8_tickets, blocked = _expanded(d8), _blocked(b35)
    candidates = []
    for core5 in combinations(core, 5):
        for edge in boundary:
            red = tuple(sorted((*core5, edge)))
            score = sum(
                math.log(max(EPSILON, float(red_probabilities[ball - 1])))
                for ball in red
            )
            candidates.append(((red, blue), score))
    candidates.sort(key=lambda item: (-item[1], item[0][0]))
    supplements: list[Ticket] = []
    for ticket, _ in candidates:
        if ticket not in d8_tickets and ticket not in blocked:
            supplements.append(ticket)
        if len(supplements) == SUPPLEMENT_COUNT:
            break
    all_tickets = [*sorted(d8_tickets), *supplements]
    if len(all_tickets) != TICKET_COUNT or len(set(all_tickets)) != TICKET_COUNT:
        raise ValueError("D8+7票数或唯一性失败")
    if set(all_tickets).intersection(blocked):
        raise ValueError("D8+7与B35重叠")
    return {
        "schemaVersion": "ssq_d8_plus7_singles_v1",
        "researchOnly": True,
        "predictionClaim": False,
        "red": list(core),
        "blue": blue,
        "top14RedRanking": list(top14),
        "supplementTickets": [
            {"red": list(red), "blue": ticket_blue} for red, ticket_blue in supplements
        ],
        "expandedTickets": [
            {"red": list(red), "blue": ticket_blue} for red, ticket_blue in all_tickets
        ],
        "audit": {
            "tickets": TICKET_COUNT,
            "costYuan": TICKET_COUNT * TICKET_PRICE_YUAN,
            "b35Overlap": 0,
        },
    }
