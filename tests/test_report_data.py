# -*- coding: utf-8 -*-

import json

import pandas as pd

from src.analysis.daily_report import compute_basic_stats, generate_candidate_groups
from src.analysis.backtest import run_fixed_candidate_backtest
from src.analysis.strategy_compare import compare_strategies
from src.analysis.sliding_window import run_sliding_window_comparison
from src.analysis.parameter_search import search_parameter_grid
from src.analysis.report_data import build_report_data, write_report_data


def _history(rows: int = 120) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        start = (idx * 7) % 60 + 1
        nums = sorted(((start + offset - 1) % 80) + 1 for offset in range(20))
        data.append({"期数": 2027000 + rows - idx, **{f"红球_{i+1}": nums[i] for i in range(20)}})
    return pd.DataFrame(data)


def test_build_report_data_contains_vue_ready_sections():
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    groups = generate_candidate_groups(stats, count=3, group_size=10, seed=2027121)
    backtest = run_fixed_candidate_backtest(history, [group.numbers for group in groups], window=20)
    strategy = compare_strategies(history, stats, groups, count=3, group_size=10, seed=2027121, window=20)
    sliding = run_sliding_window_comparison(history, {k: v.groups for k, v in strategy.items()}, window_size=20, step=20, max_windows=3)
    params = search_parameter_grid(history, stats, count=3, group_size=10, seed=2027121, window_size=20, step=20, max_windows=3)

    data = build_report_data(
        stats=stats,
        groups=groups,
        parameter_name="test_param",
        backtest_summary=backtest,
        strategy_comparison=strategy,
        sliding_window_summary=sliding,
        parameter_search_results=params,
        pick_snapshot_path="reports/picks/kl8_2027121.json",
        html_path="reports/html/kl8_daily_2027120.html",
        walk_forward_parameter_weights={"enabled": True, "bestStrategy": "omission_mix"},
    )

    assert data["lottery"] == "kl8"
    assert data["play"] == "select10"
    assert data["issue"] == stats.latest_issue
    assert data["parameterName"] == "test_param"
    assert len(data["latestNumbers"]) == 20
    assert len(data["candidateGroups"]) == 3
    assert "totalCost" in data["backtest"]
    assert "hot" in data["strategyComparison"]
    assert "hybrid" in data["slidingWindow"]
    assert data["parameterSearch"][0]["rank"] == 1
    assert data["walkForwardParameterWeights"]["bestStrategy"] == "omission_mix"
    assert data["artifacts"]["html"].endswith(".html")


def test_write_report_data_creates_json_file(tmp_path):
    history = _history()
    stats = compute_basic_stats(history, windows=(10, 20))
    groups = generate_candidate_groups(stats, count=2, group_size=10, seed=2027121)
    backtest = run_fixed_candidate_backtest(history, [group.numbers for group in groups], window=20)

    output = write_report_data(
        stats=stats,
        groups=groups,
        parameter_name="test_param",
        backtest_summary=backtest,
        output_dir=tmp_path,
    )

    assert output == tmp_path / "data" / f"kl8_daily_{stats.latest_issue}.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["issue"] == stats.latest_issue
    assert payload["candidateGroups"][0]["numbers"] == groups[0].numbers
