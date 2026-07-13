# -*- coding: utf-8 -*-
"""快乐8候选参数自动搜索。

搜索目标是把若干候选生成参数放到同一滑动窗口回测口径下排序，避免凭感觉调参。
评分仍然只代表历史体检，不代表未来预测能力。
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.analysis.backtest import BacktestSummary, run_fixed_candidate_backtest
from src.analysis.sliding_window import SlidingStrategySummary, SlidingWindowResult


@dataclass(frozen=True)
class ParameterConfig:
    """候选生成参数组合。"""

    name: str
    hot_weight: float
    cold_weight: float
    omission_weight: float
    random_weight: float
    repeat_last_weight: float = 0.0
    max_repeat_last: int = 4


@dataclass(frozen=True)
class ParameterSearchResult:
    """单组参数的搜索结果。"""

    config: ParameterConfig
    groups: list[list[int]]
    sliding_summary: SlidingStrategySummary
    score: float


def default_parameter_grid() -> list[ParameterConfig]:
    """默认搜索网格。"""

    return [
        ParameterConfig("hot_heavy", hot_weight=0.70, cold_weight=0.05, omission_weight=0.10, random_weight=0.15, max_repeat_last=4),
        ParameterConfig("balanced_mix", hot_weight=0.35, cold_weight=0.20, omission_weight=0.20, random_weight=0.25, max_repeat_last=4),
        ParameterConfig("omission_mix", hot_weight=0.25, cold_weight=0.15, omission_weight=0.45, random_weight=0.15, max_repeat_last=3),
        ParameterConfig("cold_rebound", hot_weight=0.10, cold_weight=0.45, omission_weight=0.30, random_weight=0.15, max_repeat_last=3),
        ParameterConfig("wide_random", hot_weight=0.20, cold_weight=0.20, omission_weight=0.10, random_weight=0.50, max_repeat_last=5),
        ParameterConfig("hot_omission", hot_weight=0.45, cold_weight=0.05, omission_weight=0.35, random_weight=0.15, max_repeat_last=4),
        # 复盘发现相邻期可能出现 5 个左右重号；增加重号桥接参数，避免候选全部压低上期重号。
        ParameterConfig(
            "repeat_bridge",
            hot_weight=0.32,
            cold_weight=0.08,
            omission_weight=0.25,
            random_weight=0.15,
            repeat_last_weight=0.20,
            max_repeat_last=6,
        ),
        ParameterConfig(
            "repeat_hot_mix",
            hot_weight=0.42,
            cold_weight=0.03,
            omission_weight=0.18,
            random_weight=0.12,
            repeat_last_weight=0.25,
            max_repeat_last=6,
        ),
    ]


def _frequency_score(stats: Any, number: int) -> float:
    primary_window = min(stats.frequency_by_window)
    frequency = stats.frequency_by_window[primary_window]
    max_freq = max(frequency.values()) or 1
    return frequency[number] / max_freq


def _cold_score(stats: Any, number: int) -> float:
    cold_rank = {num: idx for idx, num in enumerate(stats.cold_numbers)}
    return 1.0 - cold_rank.get(number, 79) / 79.0


def _omission_score(stats: Any, number: int) -> float:
    max_omission = max(stats.current_omission.values()) or 1
    return stats.current_omission[number] / max_omission


def _weighted_pool(stats: Any, config: ParameterConfig, rng: random.Random) -> list[tuple[int, float]]:
    pool = []
    latest_set = set(stats.latest_numbers)
    for number in range(1, 81):
        score = (
            config.hot_weight * _frequency_score(stats, number)
            + config.cold_weight * _cold_score(stats, number)
            + config.omission_weight * _omission_score(stats, number)
            + config.repeat_last_weight * (1.0 if number in latest_set else 0.0)
            + config.random_weight * rng.random()
        )
        pool.append((number, max(score, 0.0001)))
    return pool


def _zone_distribution(numbers: Sequence[int]) -> list[int]:
    zones = [0] * 8
    for number in numbers:
        zones[(int(number) - 1) // 10] += 1
    return zones


def _sample_group(stats: Any, config: ParameterConfig, group_size: int, rng: random.Random) -> list[int]:
    pool = _weighted_pool(stats, config, rng)
    selected: list[int] = []
    attempts = 0
    while len(selected) < group_size and attempts < 300:
        attempts += 1
        candidates = [number for number, _ in pool if number not in selected]
        weights = [weight for number, weight in pool if number not in selected]
        chosen = rng.choices(candidates, weights=weights, k=1)[0]
        draft = selected + [chosen]
        repeat_last = len(set(draft) & set(stats.latest_numbers))
        zones = _zone_distribution(draft)
        # 逐步控制上期重号和单区间过度集中，避免极端组合。
        if repeat_last <= config.max_repeat_last and max(zones) <= 4:
            selected.append(chosen)
    if len(selected) < group_size:
        fallback = [number for number in range(1, 81) if number not in selected]
        rng.shuffle(fallback)
        selected.extend(fallback[: group_size - len(selected)])
    return sorted(selected[:group_size])


def generate_parameter_groups(
    stats: Any,
    config: ParameterConfig,
    *,
    count: int = 10,
    group_size: int = 10,
    seed: int | None = None,
) -> list[list[int]]:
    """按一组参数生成候选组。"""

    rng = random.Random(seed if seed is not None else f"{stats.latest_issue}:{config.name}")
    groups: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(groups) < count and attempts < count * 80:
        attempts += 1
        group = _sample_group(stats, config, group_size, rng)
        key = tuple(group)
        if key not in seen:
            seen.add(key)
            groups.append(group)
    while len(groups) < count:
        group = sorted(rng.sample(range(1, 81), group_size))
        key = tuple(group)
        if key not in seen:
            seen.add(key)
            groups.append(group)
    return groups


def _sorted_history(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df["期数"] = df["期数"].astype(str)
    return df.sort_values("期数", ascending=False).reset_index(drop=True)


def _windows(history: pd.DataFrame, window_size: int, step: int, max_windows: int | None) -> list[tuple[str, pd.DataFrame]]:
    sorted_df = _sorted_history(history)
    output = []
    start = 0
    while start + window_size <= len(sorted_df):
        end = start + window_size
        output.append((f"{start + 1}-{end}", sorted_df.iloc[start:end].copy()))
        if max_windows is not None and len(output) >= max_windows:
            break
        start += step
    if not output:
        raise ValueError("历史数据不足以构造参数搜索窗口")
    return output


def _hit5_plus(summary: BacktestSummary) -> int:
    return sum(summary.hit_distribution.get(hit, 0) for hit in range(5, 11))


def _sliding_summary_for_groups(
    history: pd.DataFrame,
    strategy: str,
    groups: Sequence[Sequence[int]],
    *,
    window_size: int,
    step: int,
    max_windows: int | None,
) -> SlidingStrategySummary:
    window_results: list[SlidingWindowResult] = []
    for label, window_df in _windows(history, window_size, step, max_windows):
        summary = run_fixed_candidate_backtest(window_df, groups)
        window_results.append(
            SlidingWindowResult(
                label=label,
                start_issue=str(window_df.iloc[0]["期数"]),
                end_issue=str(window_df.iloc[-1]["期数"]),
                summary=summary,
            )
        )
    rois = [item.summary.roi for item in window_results]
    mean_hits = [item.summary.average_hit for item in window_results]
    hit5_values = [_hit5_plus(item.summary) for item in window_results]
    return SlidingStrategySummary(
        strategy=strategy,
        windows=window_results,
        mean_roi=round(statistics.fmean(rois), 4),
        best_roi=max(rois),
        worst_roi=min(rois),
        roi_std=round(statistics.pstdev(rois), 4) if len(rois) > 1 else 0.0,
        mean_hit=round(statistics.fmean(mean_hits), 4),
        mean_hit5_plus=round(statistics.fmean(hit5_values), 4),
    )


def _score(summary: SlidingStrategySummary) -> float:
    # 返奖表现优先，其次奖励中5+次数，惩罚波动；分数只用于排序。
    return round(summary.mean_roi - 0.35 * summary.roi_std + summary.mean_hit5_plus / 1000.0, 6)


def search_parameter_grid(
    history: pd.DataFrame,
    stats: Any,
    *,
    configs: Sequence[ParameterConfig] | None = None,
    count: int = 10,
    group_size: int = 10,
    seed: int | None = None,
    window_size: int = 100,
    step: int = 100,
    max_windows: int | None = 6,
) -> list[ParameterSearchResult]:
    """搜索参数网格并按综合评分降序返回。"""

    grid = list(configs) if configs is not None else default_parameter_grid()
    results: list[ParameterSearchResult] = []
    for index, config in enumerate(grid):
        config_seed = None if seed is None else seed + index * 1009
        groups = generate_parameter_groups(stats, config, count=count, group_size=group_size, seed=config_seed)
        summary = _sliding_summary_for_groups(
            history,
            config.name,
            groups,
            window_size=window_size,
            step=step,
            max_windows=max_windows,
        )
        results.append(ParameterSearchResult(config=config, groups=groups, sliding_summary=summary, score=_score(summary)))
    return sorted(results, key=lambda item: item.score, reverse=True)


def build_parameter_search_markdown(results: Sequence[ParameterSearchResult], top_n: int = 5) -> str:
    """生成参数搜索 Markdown 表格。"""

    lines = [
        "## 参数自动搜索",
        "",
        "| 排名 | 参数 | 综合评分 | 平均命中 | 平均中5+ | 平均收益率 | 收益率波动 | 最好收益率 | 最差收益率 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(results[:top_n], 1):
        summary = result.sliding_summary
        lines.append(
            f"| {rank} | {result.config.name} | {result.score:.4f} | {summary.mean_hit:.3f} | "
            f"{summary.mean_hit5_plus:.1f} | {summary.mean_roi:.2%} | {summary.roi_std:.2%} | "
            f"{summary.best_roi:.2%} | {summary.worst_roi:.2%} |"
        )
    if results:
        best = results[0]
        lines.extend(
            [
                "",
                f"当前搜索最佳参数：`{best.config.name}`；权重 hot/cold/omission/random = "
                f"`{best.config.hot_weight}/{best.config.cold_weight}/{best.config.omission_weight}/{best.config.random_weight}`，"
                f"最大上期重号 `{best.config.max_repeat_last}`。",
                "说明：参数搜索根据历史滑动窗口排序，可能过拟合，不能代表未来收益。",
                "",
            ]
        )
    return "\n".join(lines)
