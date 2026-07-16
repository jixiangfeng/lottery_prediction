# -*- coding: utf-8 -*-

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.digit_candidates import DigitCandidateConfig
from src.analysis.digit_walk_forward import (
    _select_nested_config,
    build_digit_walk_forward_markdown,
    run_digit_walk_forward_backtest,
    write_digit_walk_forward_reports,
)
from src.lotteries import get_lottery_rule


def _history(rows: int = 18) -> pd.DataFrame:
    data = []
    for index in range(rows):
        issue = 2026001 + index
        data.append(
            {
                "期数": str(issue),
                "百位": index % 10,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
        )
    return pd.DataFrame(data)


def test_digit_walk_forward_uses_prior_history_only():
    rule = get_lottery_rule("fc3d")
    original = _history()
    changed_future = original.copy()
    changed_future.loc[
        changed_future["期数"] == "2026018", ["百位", "十位", "个位"]
    ] = [9, 9, 9]

    first = run_digit_walk_forward_backtest(
        original, rule, periods=6, min_train_size=8, candidate_count=12
    )
    second = run_digit_walk_forward_backtest(
        changed_future, rule, periods=6, min_train_size=8, candidate_count=12
    )

    first_issue = next(
        item for item in first.strategy_summaries[0].issues if item.issue == "2026017"
    )
    second_issue = next(
        item for item in second.strategy_summaries[0].issues if item.issue == "2026017"
    )
    assert first_issue.candidate_texts == second_issue.candidate_texts
    assert first_issue.train_end_issue == "2026016"
    assert first_issue.train_size == 16


def test_digit_walk_forward_contains_baseline_and_required_metrics(tmp_path):
    rule = get_lottery_rule("fc3d")
    report = run_digit_walk_forward_backtest(
        _history(), rule, periods=5, min_train_size=8, candidate_count=10
    )

    assert report.period_count == 5
    assert {summary.strategy for summary in report.strategy_summaries} == {
        "current_statistics",
        "ensemble_voting",
        "uniform_random",
    }
    for summary in report.strategy_summaries:
        assert summary.target_periods == 5
        assert summary.candidate_count == 10
        assert len(summary.position_hit_coverage) == 3
        assert summary.max_direct_miss_streak >= 0
        assert summary.group_hits is not None
    current = next(
        summary
        for summary in report.strategy_summaries
        if summary.strategy == "current_statistics"
    )
    ensemble = next(
        summary
        for summary in report.strategy_summaries
        if summary.strategy == "ensemble_voting"
    )
    assert current.relative_to_baseline is not None
    assert ensemble.relative_to_baseline is not None
    assert "directHitRateDiff" in current.relative_to_baseline

    markdown = build_digit_walk_forward_markdown(report)
    markdown_path, json_path = write_digit_walk_forward_reports(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "严格逐期前推" in markdown
    assert "uniform_random" in markdown
    assert "ensemble_voting" in markdown
    assert markdown_path.exists()
    assert payload["periodCount"] == 5
    assert payload["strategies"][0]["issues"]
    assert payload["scoreBucketDistribution"]
    assert payload["strategies"][0]["issues"][0]["eligibleCount"] >= 1
    assert 0.0 <= payload["strategies"][0]["issues"][0]["actualRankPercentile"] <= 1.0


def test_pl5_walk_forward_disables_group_metric():
    rule = get_lottery_rule("pl5")
    rows = []
    for index in range(14):
        rows.append(
            {
                "期数": str(2026001 + index),
                "万位": index % 10,
                "千位": (index + 1) % 10,
                "百位": (index + 2) % 10,
                "十位": (index + 3) % 10,
                "个位": (index + 4) % 10,
            }
        )

    report = run_digit_walk_forward_backtest(
        pd.DataFrame(rows),
        rule,
        periods=3,
        min_train_size=8,
        candidate_count=5,
    )

    assert all(summary.group_hits is None for summary in report.strategy_summaries)


def test_digit_walk_forward_orders_non_padded_issues_without_future_leakage():
    rule = get_lottery_rule("fc3d")
    rows = [
        {
            "期数": str(issue),
            "百位": issue % 10,
            "十位": (issue + 1) % 10,
            "个位": (issue + 2) % 10,
        }
        for issue in range(1, 12)
    ]
    report = run_digit_walk_forward_backtest(
        pd.DataFrame(reversed(rows)),
        rule,
        periods=2,
        min_train_size=5,
        candidate_count=10,
    )

    issues = report.strategy_summaries[0].issues
    assert [item.issue for item in issues] == ["10", "11"]
    assert issues[0].train_end_issue == "9"
    assert issues[1].train_end_issue == "10"


def test_walk_forward_reports_multi_random_distribution_and_percentile():
    rule = get_lottery_rule("fc3d")
    report = run_digit_walk_forward_backtest(
        _history(50),
        rule,
        periods=8,
        min_train_size=10,
        candidate_count=8,
        baseline_seed=1,
        baseline_runs=5,
    )

    distribution = report.random_baseline_distribution
    assert distribution["runs"] == 5
    assert distribution["directHits"]["q05"] <= distribution["directHits"]["mean"]
    assert distribution["directHits"]["mean"] <= distribution["directHits"]["q95"]
    assert 0.0 <= distribution["currentStrategyPercentile"] <= 100.0
    assert 0.0 <= distribution["candidateScorePercentile"] <= 100.0
    assert 0.0 <= distribution["groupHitPercentile"] <= 100.0
    baseline = next(
        summary
        for summary in report.strategy_summaries
        if summary.strategy == "uniform_random"
    )
    current = next(
        summary
        for summary in report.strategy_summaries
        if summary.strategy == "current_statistics"
    )
    assert baseline.direct_hits == pytest.approx(distribution["directHits"]["mean"])
    assert baseline.group_hits == pytest.approx(distribution["groupHits"]["mean"])
    assert current.relative_to_baseline["directHitRateDiff"] == pytest.approx(
        current.direct_hit_rate - baseline.direct_hit_rate
    )
    assert any(
        summary.strategy == "uniform_random" for summary in report.strategy_summaries
    )

    markdown = build_digit_walk_forward_markdown(report)
    assert "选择器内部诊断" in markdown
    assert "mid-rank 50%" in markdown
    assert "分位桶诊断" in markdown
    assert "候选联合概率质量" not in markdown


def test_nested_tuning_pl5_uses_final_direct_candidates_not_rank_or_raw_score(
    monkeypatch,
):
    import src.analysis.digit_walk_forward as module

    rule = get_lottery_rule("pl5")
    train = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "万位": 1,
                "千位": 2,
                "百位": 3,
                "十位": 4,
                "个位": 5,
            }
            for index in range(6)
        ]
    )
    generated_profiles = []

    monkeypatch.setattr(
        module, "analyze_digit_history", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        module,
        "rank_digit_numbers_with_eligible_count",
        lambda *args, **kwargs: (1, 9999.0, 1000),
        raising=False,
    )

    def fake_generate(*args, config, **kwargs):
        generated_profiles.append(config.pair_weight)
        texts = ["12345"] if config.pair_weight == 0.0 else ["54321"]
        return SimpleNamespace(
            direct_candidates=[SimpleNamespace(text=text) for text in texts],
            group_candidates=[],
        )

    monkeypatch.setattr(module, "generate_digit_betting_candidates", fake_generate)

    selected, _ = _select_nested_config(train, rule, DigitCandidateConfig(count=1), 2)

    assert selected == "marginal_only"
    assert generated_profiles


def test_nested_tuning_three_digit_uses_final_group_candidates(monkeypatch):
    import src.analysis.digit_walk_forward as module

    rule = get_lottery_rule("fc3d")
    train = _history(6)
    monkeypatch.setattr(
        module, "analyze_digit_history", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        module,
        "rank_digit_numbers_with_eligible_count",
        lambda *args, **kwargs: (10, -9999.0, 1000),
        raising=False,
    )

    def fake_generate(*args, config, **kwargs):
        group_keys = ["034", "567"] if config.pair_weight == 0.0 else ["789"]
        return SimpleNamespace(
            direct_candidates=[SimpleNamespace(text="999")],
            group_candidates=[SimpleNamespace(group_key=key) for key in group_keys],
        )

    monkeypatch.setattr(module, "generate_digit_betting_candidates", fake_generate)

    selected, _ = _select_nested_config(train, rule, DigitCandidateConfig(count=1), 2)

    assert selected == "marginal_only"


def test_nested_tuning_records_training_cutoff_before_outer_target():
    rule = get_lottery_rule("fc3d")
    report = run_digit_walk_forward_backtest(
        _history(28),
        rule,
        periods=3,
        min_train_size=12,
        candidate_count=8,
        baseline_runs=2,
        nested_tuning=True,
        inner_validation_periods=4,
    )

    current = next(
        summary
        for summary in report.strategy_summaries
        if summary.strategy == "current_statistics"
    )
    assert {issue.selected_config for issue in current.issues} <= {
        "marginal_only",
        "joint_balanced",
        "joint_heavy",
        "window_30",
        "window_50",
        "window_100",
        "window_300",
        "window_all",
    }
    for issue in current.issues:
        assert issue.selected_config_train_end_issue < issue.issue
        assert issue.train_end_issue < issue.issue
        assert issue.actual_rank >= 1


def test_walk_forward_keeps_legacy_json_fields_while_adding_quality_metrics():
    rule = get_lottery_rule("fc3d")
    payload = run_digit_walk_forward_backtest(
        _history(20),
        rule,
        periods=2,
        min_train_size=10,
        candidate_count=6,
        baseline_runs=2,
    ).to_dict()

    assert payload["schemaVersion"] == 4
    assert payload["strategies"][0]["issues"][0]["candidateTexts"]
    assert "positionHitCoverage" in payload["strategies"][0]
    assert "randomBaselineDistribution" in payload
    assert "strategyBaselineDistributions" in payload
    assert "ensemble_voting" in payload["strategyBaselineDistributions"]
    assert "scoreBucketDistribution" in payload
    assert "meanCandidateScore" in payload["strategies"][0]


def test_walk_forward_can_enable_monte_carlo_and_ml_voters_without_future_data():
    rule = get_lottery_rule("fc3d")
    report = run_digit_walk_forward_backtest(
        _history(16),
        rule,
        periods=2,
        min_train_size=10,
        candidate_count=5,
        baseline_runs=1,
        advanced_models=True,
        monte_carlo_simulations=500,
        ml_training_periods=3,
        ml_negative_samples=2,
    )

    assert report.advanced_models is True
    assert report.to_dict()["advancedModels"] is True
    assert report.to_dict()["modelPerformance"]
    ensemble = next(
        summary
        for summary in report.strategy_summaries
        if summary.strategy == "ensemble_voting"
    )
    assert all(issue.train_end_issue < issue.issue for issue in ensemble.issues)


def test_walk_forward_compares_requested_independent_windows():
    rule = get_lottery_rule("fc3d")
    report = run_digit_walk_forward_backtest(
        _history(18),
        rule,
        periods=2,
        min_train_size=10,
        candidate_count=5,
        baseline_runs=1,
        compare_windows=True,
    )

    assert {item["window"] for item in report.window_comparison} == {
        "30",
        "50",
        "100",
        "300",
        "all",
    }
    assert all(item["targetPeriods"] == 2 for item in report.window_comparison)
    assert "独立窗口稳定性比较" in build_digit_walk_forward_markdown(report)
