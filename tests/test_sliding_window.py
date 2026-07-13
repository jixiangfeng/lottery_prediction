# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.daily_report import compute_basic_stats, generate_candidate_groups
from src.analysis.strategy_compare import REQUIRED_STRATEGIES, generate_strategy_groups
from src.analysis.sliding_window import (
    build_sliding_window_markdown,
    run_sliding_window_comparison,
)


def _history(rows: int = 60) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        start = (idx * 5) % 60 + 1
        nums = sorted(((start + offset - 1) % 80) + 1 for offset in range(20))
        data.append({"期数": 2025000 + rows - idx, **{f"红球_{i+1}": nums[i] for i in range(20)}})
    return pd.DataFrame(data)


def test_run_sliding_window_comparison_returns_stability_metrics():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    hybrid_groups = generate_candidate_groups(stats, count=3, group_size=10, seed=2025061)
    groups_by_strategy = generate_strategy_groups(stats, hybrid_groups, count=3, group_size=10, seed=2025061)

    results = run_sliding_window_comparison(history, groups_by_strategy, window_size=20, step=20, max_windows=3)

    assert set(REQUIRED_STRATEGIES).issubset(results)
    for strategy, summary in results.items():
        assert summary.strategy == strategy
        assert summary.window_count == 3
        assert len(summary.windows) == 3
        assert summary.mean_roi <= summary.best_roi
        assert summary.worst_roi <= summary.mean_roi
        assert summary.roi_std >= 0
        assert summary.mean_hit >= 0


def test_build_sliding_window_markdown_contains_strategy_table():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    hybrid_groups = generate_candidate_groups(stats, count=3, group_size=10, seed=2025061)
    groups_by_strategy = generate_strategy_groups(stats, hybrid_groups, count=3, group_size=10, seed=2025061)
    results = run_sliding_window_comparison(history, groups_by_strategy, window_size=20, step=20, max_windows=3)

    markdown = build_sliding_window_markdown(results)

    assert "## 滑动窗口稳定性回测" in markdown
    assert "平均收益率" in markdown
    assert "收益率波动" in markdown
    assert "窗口数" in markdown
    for strategy in REQUIRED_STRATEGIES:
        assert strategy in markdown
