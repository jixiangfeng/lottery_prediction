# -*- coding: utf-8 -*-
"""双色球研究报告使用的确定性复式构造辅助函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations


def _red_tuple(entry: Mapping[str, object]) -> tuple[int, ...]:
    raw = entry.get("red")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("红6候选缺少red数组")
    red = tuple(int(value) for value in raw)
    if len(red) != 6 or len(set(red)) != 6 or tuple(sorted(red)) != red:
        raise ValueError("红6候选必须是6个升序互异号码")
    if not all(1 <= value <= 33 for value in red):
        raise ValueError("红6候选号码越界")
    return red


def _rank(entry: Mapping[str, object], fallback: int) -> int:
    value = entry.get("rank", fallback)
    if not isinstance(value, (int, str)):
        raise ValueError("候选rank必须是整数")
    return int(value)


def _small_compound_7_red_1_blue_top5(
    ranked_entries: Sequence[Mapping[str, object]], shared_blue: str
) -> dict[str, object]:
    """从排名红6中确定性构造5组互不重复的7红1蓝小复式。

    对每个基础候选只取紧邻下一名候选中号码最小的新红球；若合并后的
    7张红6展开票与已接受组合重叠，则直接放弃该基础候选。该规则保持
    旧版五组小复式的字节级确定性，同时与已删除的跨彩种汇总模块解耦。
    """

    blue = int(shared_blue)
    if not 1 <= blue <= 16:
        raise ValueError("共享蓝球必须在01至16之间")
    parsed = [(_red_tuple(entry), entry) for entry in ranked_entries]
    used_tickets: set[tuple[int, ...]] = set()
    compounds: list[dict[str, object]] = []
    for index, (base, base_entry) in enumerate(parsed):
        selected: (
            tuple[tuple[tuple[int, ...], ...], Mapping[str, object], int] | None
        ) = None
        if index + 1 >= len(parsed):
            continue
        later, later_entry = parsed[index + 1]
        new_balls = sorted(set(later).difference(base))
        if not new_balls:
            continue
        added_ball = new_balls[0]
        red7 = tuple(sorted((*base, added_ball)))
        expanded = tuple(combinations(red7, 6))
        if not any(ticket in used_tickets for ticket in expanded):
            selected = (expanded, later_entry, added_ball)
        if selected is None:
            continue
        expanded, partner_entry, added_ball = selected
        red7 = tuple(sorted((*base, added_ball)))
        used_tickets.update(expanded)
        compounds.append(
            {
                "baseRank": _rank(base_entry, index + 1),
                "pairedLaterRank": _rank(partner_entry, index + 2),
                "addedRed": f"{added_ball:02d}",
                "red": [f"{value:02d}" for value in red7],
                "blue": f"{blue:02d}",
                "expandedTickets": [
                    {
                        "red": [f"{value:02d}" for value in ticket],
                        "blue": f"{blue:02d}",
                    }
                    for ticket in expanded
                ],
            }
        )
        if len(compounds) == 5:
            break
    if len(compounds) != 5:
        raise ValueError("排名候选不足以构造5组无重复小复式")
    return {
        "construction": "five_disjoint_7red_compounds_from_ranked_red6_v1",
        "sharedBlue": f"{blue:02d}",
        "compounds": compounds,
        "expandedNominalTicketCount": 35,
        "expandedUniqueTicketCount": len(used_tickets),
    }


__all__ = ["_small_compound_7_red_1_blue_top5"]
