# -*- coding: utf-8 -*-
"""快乐8策略横向对比。

对比目标不是证明某策略能预测未来，而是让候选生成策略与随机/热号/冷号/均衡基线放在同一
回测口径下，避免只看单一算法输出。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.analysis.backtest import BacktestSummary, run_fixed_candidate_backtest

REQUIRED_STRATEGIES = ("random", "hot", "cold", "balanced", "hybrid")
STRATEGY_NAMES = {
    "random": "随机基线",
    "hot": "热号优先",
    "cold": "冷号回补",
    "balanced": "结构均衡",
    "hybrid": "当前综合",
}


@dataclass(frozen=True)
class StrategyComparisonResult:
    """单个策略的候选组和回测摘要。"""

    strategy: str
    groups: list[list[int]]
    summary: BacktestSummary


def _dedupe_group(numbers: Sequence[int], group_size: int) -> list[int]:
    result: list[int] = []
    for number in numbers:
        value = int(number)
        if 1 <= value <= 80 and value not in result:
            result.append(value)
        if len(result) == group_size:
            break
    return sorted(result)


def _fill_group(numbers: Sequence[int], group_size: int, rng: random.Random) -> list[int]:
    result = _dedupe_group(numbers, group_size)
    pool = [number for number in range(1, 81) if number not in result]
    rng.shuffle(pool)
    result.extend(pool[: max(0, group_size - len(result))])
    return sorted(result[:group_size])


def _append_unique(groups: list[list[int]], candidate: list[int], count: int) -> None:
    if len(groups) >= count:
        return
    key = tuple(candidate)
    if len(candidate) == 10 and key not in {tuple(group) for group in groups}:
        groups.append(candidate)


def _random_groups(count: int, group_size: int, rng: random.Random) -> list[list[int]]:
    groups: list[list[int]] = []
    while len(groups) < count:
        _append_unique(groups, sorted(rng.sample(range(1, 81), group_size)), count)
    return groups


def _ranked_groups(ranked_numbers: Sequence[int], count: int, group_size: int, rng: random.Random) -> list[list[int]]:
    groups: list[list[int]] = []
    ranked = list(ranked_numbers)
    cursor = 0
    while len(groups) < count and cursor < len(ranked) + count * group_size:
        base = ranked[cursor : cursor + group_size]
        if len(base) < group_size:
            base = base + ranked[: group_size - len(base)]
        _append_unique(groups, _fill_group(base, group_size, rng), count)
        cursor += max(1, group_size // 2)
    while len(groups) < count:
        _append_unique(groups, _fill_group(rng.sample(ranked or list(range(1, 81)), min(group_size, len(ranked or []))), group_size, rng), count)
    return groups


def _balanced_groups(stats: Any, count: int, group_size: int, rng: random.Random) -> list[list[int]]:
    groups: list[list[int]] = []
    zone_pools = {
        zone: list(range(zone * 10 + 1, zone * 10 + 11))
        for zone in range(8)
    }
    hot_set = set(stats.hot_numbers[:30])
    cold_set = set(stats.cold_numbers[:30])
    while len(groups) < count:
        candidate: list[int] = []
        zones = list(range(8))
        rng.shuffle(zones)
        for zone in zones:
            pool = zone_pools[zone]
            weighted = [3 if number in hot_set else 2 if number in cold_set else 1 for number in pool]
            chosen = rng.choices(pool, weights=weighted, k=1)[0]
            if chosen not in candidate:
                candidate.append(chosen)
        remaining_pool = [number for number in range(1, 81) if number not in candidate]
        rng.shuffle(remaining_pool)
        candidate.extend(remaining_pool[: max(0, group_size - len(candidate))])
        candidate = sorted(candidate[:group_size])
        odd = sum(number % 2 for number in candidate)
        big = sum(number >= 41 for number in candidate)
        if 3 <= odd <= 7 and 3 <= big <= 7:
            _append_unique(groups, candidate, count)
    return groups


def generate_strategy_groups(
    stats: Any,
    hybrid_groups: Sequence[Any],
    *,
    count: int = 10,
    group_size: int = 10,
    seed: int | None = None,
) -> dict[str, list[list[int]]]:
    """生成各策略候选组。"""

    rng = random.Random(seed if seed is not None else int(stats.latest_issue))
    primary_window = min(stats.frequency_by_window)
    primary_frequency = stats.frequency_by_window[primary_window]
    hot_ranked = sorted(range(1, 81), key=lambda n: (-primary_frequency[n], n))
    cold_ranked = sorted(range(1, 81), key=lambda n: (primary_frequency[n], -stats.current_omission[n], n))
    hybrid = [list(group.numbers) for group in hybrid_groups[:count]]
    while len(hybrid) < count:
        _append_unique(hybrid, sorted(rng.sample(range(1, 81), group_size)), count)

    return {
        "random": _random_groups(count, group_size, random.Random(rng.randint(1, 10_000_000))),
        "hot": _ranked_groups(hot_ranked, count, group_size, random.Random(rng.randint(1, 10_000_000))),
        "cold": _ranked_groups(cold_ranked, count, group_size, random.Random(rng.randint(1, 10_000_000))),
        "balanced": _balanced_groups(stats, count, group_size, random.Random(rng.randint(1, 10_000_000))),
        "hybrid": [sorted(group) for group in hybrid[:count]],
    }


def compare_strategies(
    history: pd.DataFrame,
    stats: Any,
    hybrid_groups: Sequence[Any],
    *,
    count: int = 10,
    group_size: int = 10,
    seed: int | None = None,
    window: int = 100,
) -> dict[str, StrategyComparisonResult]:
    """生成策略候选组并逐一回测。"""

    groups_by_strategy = generate_strategy_groups(
        stats,
        hybrid_groups,
        count=count,
        group_size=group_size,
        seed=seed,
    )
    return {
        strategy: StrategyComparisonResult(
            strategy=strategy,
            groups=groups,
            summary=run_fixed_candidate_backtest(history, groups, window=window),
        )
        for strategy, groups in groups_by_strategy.items()
    }


def build_strategy_comparison_markdown(comparison: dict[str, StrategyComparisonResult]) -> str:
    """生成策略横向对比 Markdown 表格。"""

    lines = [
        "## 策略横向对比",
        "",
        "| 策略 | 说明 | 平均命中 | 中5+ | 投入 | 返奖 | 收益率 | 最大连续未中奖 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in REQUIRED_STRATEGIES:
        result = comparison[strategy]
        summary = result.summary
        hit5_plus = sum(summary.hit_distribution.get(hit, 0) for hit in range(5, 11))
        lines.append(
            f"| {strategy} | {STRATEGY_NAMES.get(strategy, strategy)} | {summary.average_hit:.3f} | "
            f"{hit5_plus} | {summary.total_cost} | {summary.total_prize} | {summary.roi:.2%} | {summary.max_miss_streak} |"
        )
    lines.extend(
        [
            "",
            "说明：策略对比使用同一期历史窗口、同样注数和同一奖表，仅用于比较历史表现。",
            "",
        ]
    )
    return "\n".join(lines)
