# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.daily_report import compute_basic_stats, generate_candidate_groups
from src.analysis.strategy_compare import (
    REQUIRED_STRATEGIES,
    build_strategy_comparison_markdown,
    compare_strategies,
    generate_strategy_groups,
)


def _history() -> pd.DataFrame:
    rows = []
    for idx in range(30):
        start = (idx * 3) % 60 + 1
        nums = sorted(((start + offset - 1) % 80) + 1 for offset in range(20))
        rows.append({"期数": 2024030 - idx, **{f"红球_{i+1}": nums[i] for i in range(20)}})
    return pd.DataFrame(rows)


def test_generate_strategy_groups_returns_required_unique_groups():
    stats = compute_basic_stats(_history(), windows=(10, 20))
    hybrid_groups = generate_candidate_groups(stats, count=4, group_size=10, seed=2024031)

    groups_by_strategy = generate_strategy_groups(stats, hybrid_groups, count=4, group_size=10, seed=2024031)

    assert set(REQUIRED_STRATEGIES).issubset(groups_by_strategy)
    for strategy, groups in groups_by_strategy.items():
        assert len(groups) == 4, strategy
        assert len({tuple(group) for group in groups}) == 4
        for group in groups:
            assert len(group) == 10
            assert group == sorted(group)
            assert all(1 <= number <= 80 for number in group)


def test_compare_strategies_runs_backtests_for_each_strategy():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    hybrid_groups = generate_candidate_groups(stats, count=4, group_size=10, seed=2024031)

    comparison = compare_strategies(history, stats, hybrid_groups, count=4, group_size=10, seed=2024031, window=20)

    assert set(REQUIRED_STRATEGIES).issubset(comparison)
    for strategy, result in comparison.items():
        assert result.strategy == strategy
        assert result.summary.draw_count == 20
        assert result.summary.group_count == 4
        assert result.summary.total_bets == 80
        assert result.groups


def test_build_strategy_comparison_markdown_contains_all_metrics():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    hybrid_groups = generate_candidate_groups(stats, count=3, group_size=10, seed=2024031)
    comparison = compare_strategies(history, stats, hybrid_groups, count=3, group_size=10, seed=2024031, window=20)

    markdown = build_strategy_comparison_markdown(comparison)

    assert "## 策略横向对比" in markdown
    for strategy in REQUIRED_STRATEGIES:
        assert strategy in markdown
    assert "平均命中" in markdown
    assert "收益率" in markdown
    assert "中5+" in markdown
