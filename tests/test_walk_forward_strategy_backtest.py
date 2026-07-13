# -*- coding: utf-8 -*-

import json

import pandas as pd

from src.analysis.parameter_search import ParameterConfig
from src.analysis.walk_forward_strategy_backtest import (
    build_walk_forward_strategy_markdown,
    run_walk_forward_strategy_backtest,
    write_walk_forward_strategy_reports,
)


def _history(rows: int = 50) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        start = (idx * 7) % 60 + 1
        nums = sorted(((start + offset - 1) % 80) + 1 for offset in range(20))
        data.append({"期数": 2026000 + rows - idx, **{f"红球_{i+1}": nums[i] for i in range(20)}})
    return pd.DataFrame(data)


def _configs() -> list[ParameterConfig]:
    return [
        ParameterConfig("hot_heavy", hot_weight=0.7, cold_weight=0.05, omission_weight=0.1, random_weight=0.15),
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


def test_run_walk_forward_strategy_backtest_uses_prior_history_only():
    report = run_walk_forward_strategy_backtest(
        _history(),
        configs=_configs(),
        periods=8,
        min_train_size=20,
        group_count=3,
        group_size=10,
    )

    assert report.period_count == 8
    assert report.best_strategy in {"hot_heavy", "repeat_hot_mix"}
    assert len(report.summaries) == 2
    for summary in report.summaries:
        assert summary.period_count == 8
        assert summary.group_count == 3
        assert summary.total_cost == 8 * 3 * 2
        assert summary.issue_results[0].issue == "2026050"
        assert len(summary.issue_results[0].groups) == 3
        assert len(summary.issue_results[0].groups[0]) == 10


def test_walk_forward_markdown_and_json_are_written(tmp_path):
    report = run_walk_forward_strategy_backtest(
        _history(),
        configs=_configs(),
        periods=5,
        min_train_size=20,
        group_count=2,
        group_size=10,
    )

    markdown = build_walk_forward_strategy_markdown(report)
    markdown_path, json_path = write_walk_forward_strategy_reports(report, tmp_path)

    assert "快乐8逐期前推策略回测" in markdown
    assert "策略汇总" in markdown
    assert markdown_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["bestStrategy"] == report.best_strategy
    assert payload["summaries"][0]["issues"]
