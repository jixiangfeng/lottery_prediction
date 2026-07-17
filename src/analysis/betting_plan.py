# -*- coding: utf-8 -*-
"""投注结构/复式/胆拖建议。

该模块只做成本、覆盖结构和历史候选压缩，不承诺提高中奖概率。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BetCost:
    """投注注数与成本。"""

    bet_count: int
    cost: int
    risk_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "betCount": self.bet_count,
            "cost": self.cost,
            "riskLevel": self.risk_level,
        }


@dataclass(frozen=True)
class BettingPlanItem:
    """单个投注方案。"""

    kind: str
    title: str
    numbers: list[int] | None = None
    number_count: int | None = None
    banker_numbers: list[int] | None = None
    drag_numbers: list[int] | None = None
    position_pools: dict[str, list[int]] | None = None
    bet_count: int = 0
    cost: int = 0
    risk_level: str = "低"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "numbers": self.numbers,
            "numberCount": self.number_count,
            "bankerNumbers": self.banker_numbers,
            "dragNumbers": self.drag_numbers,
            "positionPools": self.position_pools,
            "betCount": self.bet_count,
            "cost": self.cost,
            "riskLevel": self.risk_level,
            "note": self.note,
        }


@dataclass(frozen=True)
class BettingPlan:
    """某玩法的一组投注结构建议。"""

    play: str
    display_name: str
    core_numbers: list[int]
    assist_numbers: list[int]
    defensive_numbers: list[int]
    budgets: list[int]
    plans: list[BettingPlanItem]
    disclaimer: str = (
        "复式/胆拖只是在成本可控前提下扩大覆盖，不保证中奖，不建议追损或倍投。"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "play": self.play,
            "displayName": self.display_name,
            "coreNumbers": self.core_numbers,
            "assistNumbers": self.assist_numbers,
            "defensiveNumbers": self.defensive_numbers,
            "budgets": self.budgets,
            "plans": [item.to_dict() for item in self.plans],
            "disclaimer": self.disclaimer,
        }


def _risk_level(cost: int) -> str:
    if cost <= 100:
        return "低"
    if cost <= 300:
        return "中"
    return "高"


def _position_pools(
    candidates: Sequence[Any], width: int | None = None
) -> dict[str, list[int]]:
    if not candidates:
        return {}
    draw_count = len(candidates[0].numbers)
    limit = width or (4 if draw_count == 3 else 3)
    pools: dict[str, list[int]] = {}
    for index in range(draw_count):
        counter: Counter[int] = Counter(
            int(candidate.numbers[index]) for candidate in candidates
        )
        pools[f"pos{index + 1}"] = [
            digit
            for digit, _ in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
        ]
    return pools


def _position_cost(pools: dict[str, list[int]], *, ticket_price: int = 2) -> BetCost:
    bet_count = 1
    for digits in pools.values():
        bet_count *= max(1, len(digits))
    cost = bet_count * ticket_price
    return BetCost(bet_count=bet_count, cost=cost, risk_level=_risk_level(cost))


def _group_compound_pool(candidates: Sequence[Any], limit: int = 6) -> list[int]:
    counter: Counter[int] = Counter()
    for candidate in candidates:
        for digit in set(int(number) for number in candidate.numbers):
            counter[digit] += 1
    return [
        digit
        for digit, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def build_digit_betting_plan(
    candidate_result: Any, *, ticket_price: int = 2
) -> BettingPlan:
    """从数字彩候选生成直选/组选/定位复式建议。"""

    candidates = list(candidate_result.candidates)
    pools = _position_pools(candidates)
    cost = _position_cost(pools, ticket_price=ticket_price)
    plans = [
        BettingPlanItem(
            kind="定位复式" if len(pools) == 5 else "直选复式",
            title="按位置候选池投注",
            position_pools=pools,
            bet_count=cost.bet_count,
            cost=cost.cost,
            risk_level=cost.risk_level,
            note="位置池来自系统候选聚合，适合按预算删减每位数字。",
        )
    ]
    core_numbers: list[int] = []
    assist_numbers: list[int] = []
    if len(pools) == 3:
        group_pool = _group_compound_pool(candidates, limit=6)
        group_bets = math.comb(len(group_pool), 3) if len(group_pool) >= 3 else 0
        group_cost = group_bets * ticket_price
        plans.append(
            BettingPlanItem(
                kind="组选复式",
                title="三位数字组选池",
                numbers=group_pool,
                number_count=len(group_pool),
                bet_count=group_bets,
                cost=group_cost,
                risk_level=_risk_level(group_cost),
                note="只适用于福彩3D/排列三组选参考；不覆盖豹子。",
            )
        )
        core_numbers = group_pool[:4]
        assist_numbers = group_pool[4:]
    else:
        flattened = [digit for digits in pools.values() for digit in digits]
        core_numbers = sorted(set(flattened))[:6]
        assist_numbers = sorted(set(flattened))[6:]
    return BettingPlan(
        play=str(candidate_result.rule_code),
        display_name=str(candidate_result.display_name),
        core_numbers=core_numbers,
        assist_numbers=assist_numbers,
        defensive_numbers=[],
        budgets=[cost.cost],
        plans=plans,
    )


def _format_numbers(numbers: Sequence[int] | None) -> str:
    if not numbers:
        return "--"
    return " ".join(str(int(number)).zfill(2) for number in numbers)


def build_betting_plan_markdown(plan: BettingPlan) -> str:
    """投注方案 Markdown。"""

    lines = [
        f"## {plan.display_name} 复式/胆拖建议",
        "",
        f"- 核心池：`{_format_numbers(plan.core_numbers)}`",
        f"- 辅助池：`{_format_numbers(plan.assist_numbers)}`",
    ]
    if plan.defensive_numbers:
        lines.append(f"- 防守池：`{_format_numbers(plan.defensive_numbers)}`")
    lines.extend(
        [
            "",
            "| 类型 | 标题 | 号码/位置池 | 注数 | 成本 | 风险 | 说明 |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for item in plan.plans:
        if item.position_pools:
            pool_text = "；".join(
                f"{pos}:{','.join(str(d) for d in digits)}"
                for pos, digits in item.position_pools.items()
            )
        elif item.banker_numbers or item.drag_numbers:
            pool_text = f"胆:{_format_numbers(item.banker_numbers)}；拖:{_format_numbers(item.drag_numbers)}"
        else:
            pool_text = _format_numbers(item.numbers)
        cost = f"{item.cost}元" if item.cost else "需复核"
        bets = str(item.bet_count) if item.bet_count else "--"
        lines.append(
            f"| {item.kind} | {item.title} | `{pool_text}` | {bets} | {cost} | {item.risk_level} | {item.note} |"
        )
    lines.extend(["", f"说明：{plan.disclaimer}", ""])
    return "\n".join(lines)
