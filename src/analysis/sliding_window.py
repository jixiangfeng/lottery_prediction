# -*- coding: utf-8 -*-
"""快乐8策略滑动窗口稳定性回测。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.analysis.backtest import BacktestSummary, run_fixed_candidate_backtest
from src.analysis.strategy_compare import REQUIRED_STRATEGIES, STRATEGY_NAMES


@dataclass(frozen=True)
class SlidingWindowResult:
    """单个历史窗口的回测结果。"""

    label: str
    start_issue: str
    end_issue: str
    summary: BacktestSummary


@dataclass(frozen=True)
class SlidingStrategySummary:
    """某个策略在多个滑动窗口上的稳定性摘要。"""

    strategy: str
    windows: list[SlidingWindowResult]
    mean_roi: float
    best_roi: float
    worst_roi: float
    roi_std: float
    mean_hit: float
    mean_hit5_plus: float

    @property
    def window_count(self) -> int:
        return len(self.windows)


def _sorted_history(history: pd.DataFrame) -> pd.DataFrame:
    if "期数" not in history.columns:
        raise ValueError("历史数据必须包含【期数】列")
    sorted_df = history.copy()
    sorted_df["期数"] = sorted_df["期数"].astype(str)
    return sorted_df.sort_values("期数", ascending=False).reset_index(drop=True)


def _history_windows(history: pd.DataFrame, window_size: int, step: int, max_windows: int | None) -> list[tuple[str, pd.DataFrame]]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size 和 step 必须为正整数")
    sorted_df = _sorted_history(history)
    windows: list[tuple[str, pd.DataFrame]] = []
    start = 0
    while start + window_size <= len(sorted_df):
        end = start + window_size
        label = f"{start + 1}-{end}"
        windows.append((label, sorted_df.iloc[start:end].copy()))
        if max_windows is not None and len(windows) >= max_windows:
            break
        start += step
    if not windows:
        raise ValueError("历史数据不足以构造滑动窗口")
    return windows


def _hit5_plus(summary: BacktestSummary) -> int:
    return sum(summary.hit_distribution.get(hit, 0) for hit in range(5, 11))


def run_sliding_window_comparison(
    history: pd.DataFrame,
    groups_by_strategy: dict[str, Sequence[Sequence[int]]],
    *,
    window_size: int = 100,
    step: int = 100,
    max_windows: int | None = 6,
) -> dict[str, SlidingStrategySummary]:
    """对每个策略在多个历史窗口上做固定候选回测。"""

    windows = _history_windows(history, window_size, step, max_windows)
    results: dict[str, SlidingStrategySummary] = {}
    for strategy in REQUIRED_STRATEGIES:
        groups = groups_by_strategy[strategy]
        window_results: list[SlidingWindowResult] = []
        for label, window_df in windows:
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
        results[strategy] = SlidingStrategySummary(
            strategy=strategy,
            windows=window_results,
            mean_roi=round(statistics.fmean(rois), 4),
            best_roi=max(rois),
            worst_roi=min(rois),
            roi_std=round(statistics.pstdev(rois), 4) if len(rois) > 1 else 0.0,
            mean_hit=round(statistics.fmean(mean_hits), 4),
            mean_hit5_plus=round(statistics.fmean(hit5_values), 4),
        )
    return results


def build_sliding_window_markdown(results: dict[str, SlidingStrategySummary]) -> str:
    """生成滑动窗口稳定性 Markdown 表格。"""

    lines = [
        "## 滑动窗口稳定性回测",
        "",
        "| 策略 | 说明 | 窗口数 | 平均命中 | 平均中5+ | 平均收益率 | 最好收益率 | 最差收益率 | 收益率波动 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in REQUIRED_STRATEGIES:
        result = results[strategy]
        lines.append(
            f"| {strategy} | {STRATEGY_NAMES.get(strategy, strategy)} | {result.window_count} | "
            f"{result.mean_hit:.3f} | {result.mean_hit5_plus:.1f} | {result.mean_roi:.2%} | "
            f"{result.best_roi:.2%} | {result.worst_roi:.2%} | {result.roi_std:.2%} |"
        )
    lines.extend(
        [
            "",
            "说明：滑动窗口把历史分段回放，收益率波动越低代表该策略历史表现越稳定；但仍不代表未来收益。",
            "",
        ]
    )
    return "\n".join(lines)
