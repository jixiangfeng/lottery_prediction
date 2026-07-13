# -*- coding: utf-8 -*-
"""快乐8固定候选组回测工具。

当前模块用于回答一个明确问题：如果把今天生成的候选组固定下来，放到最近 N 期历史开奖里，
每注能命中几个、成本多少、按快乐8选十奖表能返多少。它不是未来预测，只是策略体检。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

KL8_SELECT10_PRIZE_TABLE: dict[int, int] = {
    10: 5_000_000,
    9: 8_000,
    8: 800,
    7: 80,
    6: 5,
    5: 3,
    0: 2,
}


@dataclass(frozen=True)
class BacktestSummary:
    """固定候选组在历史开奖上的回测摘要。"""

    draw_count: int
    group_count: int
    total_bets: int
    ticket_price: int
    total_cost: int
    total_prize: int
    roi: float
    average_hit: float
    hit_distribution: Counter[int]
    max_miss_streak: int


def _number_columns(df: pd.DataFrame) -> list[str]:
    columns = [f"红球_{idx}" for idx in range(1, 21)]
    missing = [column for column in columns if column not in df.columns]
    if "期数" not in df.columns or missing:
        raise ValueError("历史数据必须包含【期数】和【红球_1】到【红球_20】列")
    return columns


def _draw_sets(df: pd.DataFrame, window: int | None = None) -> list[set[int]]:
    columns = _number_columns(df)
    sorted_df = df.copy()
    sorted_df["期数"] = sorted_df["期数"].astype(str)
    sorted_df = sorted_df.sort_values("期数", ascending=False)
    if window is not None:
        sorted_df = sorted_df.head(window)
    return [set(int(row[column]) for column in columns) for _, row in sorted_df.iterrows()]


def _normalize_groups(groups: Sequence[Sequence[int]]) -> list[set[int]]:
    normalized: list[set[int]] = []
    for group in groups:
        numbers = [int(number) for number in group]
        if len(numbers) != 10 or len(set(numbers)) != 10:
            raise ValueError("快乐8选十回测要求每组恰好 10 个不重复号码")
        if any(number < 1 or number > 80 for number in numbers):
            raise ValueError("快乐8号码必须位于 1-80")
        normalized.append(set(numbers))
    if not normalized:
        raise ValueError("至少需要一组候选号码")
    return normalized


def run_fixed_candidate_backtest(
    history: pd.DataFrame,
    groups: Sequence[Sequence[int]],
    *,
    ticket_price: int = 2,
    window: int | None = None,
    prize_table: dict[int, int] | None = None,
) -> BacktestSummary:
    """把固定候选组选十放到最近历史开奖中回测。

    Args:
        history: 包含 `期数` 和 `红球_1`...`红球_20` 的历史开奖表。
        groups: 候选组选十，每组 10 个号码。
        ticket_price: 每注成本，默认 2 元。
        window: 仅回测最近 N 期；默认使用全部历史。
        prize_table: 命中奖表，默认快乐8选十奖表。
    """

    if ticket_price <= 0:
        raise ValueError("ticket_price 必须为正整数")
    draw_sets = _draw_sets(history, window=window)
    if not draw_sets:
        raise ValueError("没有可用于回测的历史开奖")
    normalized_groups = _normalize_groups(groups)
    prizes = prize_table or KL8_SELECT10_PRIZE_TABLE

    hit_distribution: Counter[int] = Counter()
    total_prize = 0
    total_hits = 0
    current_miss_streak = 0
    max_miss_streak = 0

    for draw in draw_sets:
        for group in normalized_groups:
            hit = len(group & draw)
            hit_distribution[hit] += 1
            total_hits += hit
            prize = int(prizes.get(hit, 0))
            total_prize += prize
            if prize > 0:
                current_miss_streak = 0
            else:
                current_miss_streak += 1
                max_miss_streak = max(max_miss_streak, current_miss_streak)

    total_bets = len(draw_sets) * len(normalized_groups)
    total_cost = total_bets * ticket_price
    roi = round((total_prize - total_cost) / total_cost, 4) if total_cost else 0.0
    average_hit = round(total_hits / total_bets, 4) if total_bets else 0.0

    return BacktestSummary(
        draw_count=len(draw_sets),
        group_count=len(normalized_groups),
        total_bets=total_bets,
        ticket_price=ticket_price,
        total_cost=total_cost,
        total_prize=total_prize,
        roi=roi,
        average_hit=average_hit,
        hit_distribution=hit_distribution,
        max_miss_streak=max_miss_streak,
    )


def build_backtest_markdown(summary: BacktestSummary) -> str:
    """生成回测 Markdown 片段。"""

    lines = [
        "## 最近历史固定候选回测",
        "",
        f"- 回测期数：`{summary.draw_count}`",
        f"- 候选组数：`{summary.group_count}`",
        f"- 总注数：`{summary.total_bets}`",
        f"- 投入：`{summary.total_cost}` 元",
        f"- 返奖：`{summary.total_prize}` 元",
        f"- 收益率：`{summary.roi:.2%}`",
        f"- 平均命中：`{summary.average_hit}` 个/注",
        f"- 最大连续未中奖：`{summary.max_miss_streak}` 注",
        "",
        "| 命中数 | 次数 |",
        "|---:|---:|",
    ]
    for hit in range(10, -1, -1):
        lines.append(f"| 中{hit} | {summary.hit_distribution.get(hit, 0)} |")
    lines.extend(
        [
            "",
            "说明：这是把当前候选组固定后回放到历史开奖上的体检，不代表未来收益。",
            "",
        ]
    )
    return "\n".join(lines)
