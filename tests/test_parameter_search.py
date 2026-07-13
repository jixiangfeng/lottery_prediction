# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.daily_report import compute_basic_stats
from src.analysis.parameter_search import (
    ParameterConfig,
    build_parameter_search_markdown,
    default_parameter_grid,
    search_parameter_grid,
)


def _history(rows: int = 80) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        start = (idx * 7) % 60 + 1
        nums = sorted(((start + offset - 1) % 80) + 1 for offset in range(20))
        data.append({"期数": 2026000 + rows - idx, **{f"红球_{i+1}": nums[i] for i in range(20)}})
    return pd.DataFrame(data)


def test_default_parameter_grid_has_named_weight_profiles():
    grid = default_parameter_grid()

    assert len(grid) >= 4
    assert all(isinstance(item, ParameterConfig) for item in grid)
    assert {item.name for item in grid} >= {"hot_heavy", "balanced_mix", "omission_mix", "repeat_bridge"}
    for item in grid:
        assert round(item.hot_weight + item.cold_weight + item.omission_weight + item.random_weight + item.repeat_last_weight, 6) > 0
        assert 0 <= item.max_repeat_last <= 10
    repeat_bridge = next(item for item in grid if item.name == "repeat_bridge")
    assert repeat_bridge.repeat_last_weight > 0
    assert repeat_bridge.max_repeat_last >= 5


def test_search_parameter_grid_returns_sorted_results_with_groups():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    configs = default_parameter_grid()[:4]

    results = search_parameter_grid(
        history,
        stats,
        configs=configs,
        count=4,
        group_size=10,
        seed=2026081,
        window_size=20,
        step=20,
        max_windows=3,
    )

    assert len(results) == len(configs)
    assert results == sorted(results, key=lambda item: item.score, reverse=True)
    for result in results:
        assert result.config in configs
        assert len(result.groups) == 4
        assert result.sliding_summary.window_count == 3
        assert isinstance(result.score, float)
        for group in result.groups:
            assert len(group) == 10
            assert group == sorted(group)
            assert all(1 <= number <= 80 for number in group)


def test_build_parameter_search_markdown_contains_ranked_configs():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    results = search_parameter_grid(
        history,
        stats,
        configs=default_parameter_grid()[:3],
        count=3,
        group_size=10,
        seed=2026081,
        window_size=20,
        step=20,
        max_windows=3,
    )

    markdown = build_parameter_search_markdown(results, top_n=3)

    assert "## 参数自动搜索" in markdown
    assert "综合评分" in markdown
    assert "平均收益率" in markdown
    assert "收益率波动" in markdown
    assert results[0].config.name in markdown
