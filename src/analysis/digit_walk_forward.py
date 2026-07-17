# -*- coding: utf-8 -*-
"""数字彩严格逐期前推回测。

每个目标期只使用该期开奖之前的历史生成候选，并同时评估复合统计策略、
集成投票策略与 ``uniform_random`` 随机基线。该模块用于验证流程与历史表现，
不保证提高中奖概率。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.analysis.digit_advanced_models import build_advanced_model_scores
from src.analysis.digit_candidates import (
    ENSEMBLE_MODEL_NAMES,
    DigitCandidateConfig,
    DigitCandidateResult,
    generate_digit_betting_candidates,
    generate_uniform_digit_betting_candidates,
    rank_digit_numbers_with_eligible_count,
    with_all_history_window,
)
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_statistics import analyze_digit_history
from src.analysis.prediction_viability import (
    PredictionViabilityReport,
    build_prediction_viability_report,
    calculate_group_random_probability,
)
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitWalkForwardIssue:
    """单个目标期的严格前推结果。"""

    issue: str
    train_start_issue: str
    train_end_issue: str
    train_size: int
    actual_text: str
    candidate_texts: list[str]
    direct_hit: bool
    group_hit: bool | None
    position_hits: list[bool]
    group_candidate_keys: list[str]
    selected_config: str
    selected_config_train_end_issue: str
    ranking_mode: str
    candidate_mean_score: float
    actual_score: float
    actual_rank: int
    eligible_count: int
    actual_rank_percentile: float
    direct_random_probability: float
    group_random_probability: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "trainStartIssue": self.train_start_issue,
            "trainEndIssue": self.train_end_issue,
            "trainSize": self.train_size,
            "actualText": self.actual_text,
            "candidateTexts": self.candidate_texts,
            "directHit": self.direct_hit,
            "groupHit": self.group_hit,
            "positionHits": self.position_hits,
            "groupCandidateKeys": self.group_candidate_keys,
            "selectedConfig": self.selected_config,
            "selectedConfigTrainEndIssue": self.selected_config_train_end_issue,
            "rankingMode": self.ranking_mode,
            "candidateMeanScore": self.candidate_mean_score,
            "actualScore": self.actual_score,
            "actualRank": self.actual_rank,
            "eligibleCount": self.eligible_count,
            "actualRankPercentile": self.actual_rank_percentile,
            "directRandomProbability": self.direct_random_probability,
            "groupRandomProbability": self.group_random_probability,
        }


@dataclass(frozen=True)
class DigitWalkForwardStrategySummary:
    """单个策略的逐期前推汇总。"""

    strategy: str
    target_periods: int
    candidate_count: int
    direct_hits: float
    direct_hit_rate: float
    group_hits: float | None
    group_hit_rate: float | None
    position_hit_coverage: dict[str, dict[str, float | int]]
    max_direct_miss_streak: float
    relative_to_baseline: dict[str, float] | None
    issues: list[DigitWalkForwardIssue]
    mean_candidate_score: float
    mean_actual_score: float
    mean_actual_rank: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "targetPeriods": self.target_periods,
            "candidateCount": self.candidate_count,
            "directHits": self.direct_hits,
            "directHitRate": self.direct_hit_rate,
            "groupHits": self.group_hits,
            "groupHitRate": self.group_hit_rate,
            "positionHitCoverage": self.position_hit_coverage,
            "maxDirectMissStreak": self.max_direct_miss_streak,
            "relativeToBaseline": self.relative_to_baseline,
            "issues": [issue.to_dict() for issue in self.issues],
            "meanCandidateScore": self.mean_candidate_score,
            "meanActualScore": self.mean_actual_score,
            "meanActualRank": self.mean_actual_rank,
        }


@dataclass(frozen=True)
class DigitWalkForwardReport:
    """数字彩严格逐期前推报告。"""

    rule_code: str
    display_name: str
    period_count: int
    min_train_size: int
    candidate_count: int
    strategy_summaries: list[DigitWalkForwardStrategySummary]
    baseline_runs: int
    random_baseline_distribution: dict[str, Any]
    strategy_baseline_distributions: dict[str, dict[str, Any]]
    score_bucket_distribution: list[dict[str, Any]]
    strategy_score_bucket_distributions: dict[str, list[dict[str, Any]]]
    nested_tuning: bool
    inner_validation_periods: int
    advanced_models: bool
    window_comparison: list[dict[str, Any]]
    model_performance: dict[str, dict[str, float | int]]
    active_model_names: tuple[str, ...]
    available_model_names: tuple[str, ...]
    model_activation_counts: dict[str, int]
    strategy_viability: dict[str, PredictionViabilityReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 5,
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "periodCount": self.period_count,
            "minTrainSize": self.min_train_size,
            "candidateCount": self.candidate_count,
            "strategies": [summary.to_dict() for summary in self.strategy_summaries],
            "baselineRuns": self.baseline_runs,
            "randomBaselineDistribution": self.random_baseline_distribution,
            "strategyBaselineDistributions": self.strategy_baseline_distributions,
            "scoreBucketDistribution": self.score_bucket_distribution,
            "strategyScoreBucketDistributions": self.strategy_score_bucket_distributions,
            "nestedTuning": self.nested_tuning,
            "innerValidationPeriods": self.inner_validation_periods,
            "advancedModels": self.advanced_models,
            "windowComparison": self.window_comparison,
            "modelPerformance": self.model_performance,
            "activeModelNames": list(self.active_model_names),
            "activeModelCount": len(self.active_model_names),
            "availableModelNames": list(self.available_model_names),
            "availableModelCount": len(self.available_model_names),
            "modelActivationCounts": self.model_activation_counts,
            "strategyViability": {
                name: viability.to_dict()
                for name, viability in self.strategy_viability.items()
            },
            "disclaimer": "严格逐期前推仅用于历史验证，随机基线和统计策略都不能保证提高中奖概率。",
        }


def _candidate_text(numbers: Sequence[int]) -> str:
    return "".join(str(int(number)) for number in numbers)


def _group_key(text: str) -> str:
    return "".join(sorted(text))


def _evaluate_issue(
    target: pd.Series,
    train: pd.DataFrame,
    rule: LotteryRule,
    candidate_result: DigitCandidateResult,
    *,
    group_candidate_keys: Sequence[str] | None = None,
    selected_config: str = "joint_balanced",
    selected_config_train_end_issue: str = "",
    actual_rank: int | None = None,
    actual_score: float | None = None,
    eligible_count: int | None = None,
    ranking_mode: str = "composite",
) -> DigitWalkForwardIssue:
    actual_numbers = [int(target[column]) for column in rule.number_columns]
    actual_text = _candidate_text(actual_numbers)
    candidates = candidate_result.candidates
    candidate_texts = [candidate.text for candidate in candidates]
    direct_hit = actual_text in candidate_texts
    group_keys = list(group_candidate_keys or [])
    if rule.draw_count == 3 and not group_keys:
        group_keys = sorted({_group_key(candidate.text) for candidate in candidates})
    group_hit = _group_key(actual_text) in group_keys if rule.draw_count == 3 else None
    direct_random_probability = len(set(candidate_texts)) / (10**rule.draw_count)
    group_random_probability = (
        calculate_group_random_probability(group_keys, draw_count=rule.draw_count)
        if rule.draw_count == 3
        else None
    )
    position_hits = [
        any(
            candidate.numbers[index] == actual_numbers[index]
            for candidate in candidates
        )
        for index in range(rule.draw_count)
    ]
    effective_rank = len(candidates) + 1 if actual_rank is None else actual_rank
    effective_eligible_count = (
        len(candidates) if eligible_count is None else eligible_count
    )
    return DigitWalkForwardIssue(
        issue=str(target["期数"]),
        train_start_issue=str(train.iloc[0]["期数"]),
        train_end_issue=str(train.iloc[-1]["期数"]),
        train_size=len(train),
        actual_text=actual_text,
        candidate_texts=candidate_texts,
        direct_hit=direct_hit,
        group_hit=group_hit,
        position_hits=position_hits,
        group_candidate_keys=group_keys,
        selected_config=selected_config,
        selected_config_train_end_issue=selected_config_train_end_issue
        or str(train.iloc[-1]["期数"]),
        ranking_mode=ranking_mode,
        candidate_mean_score=(
            sum(
                (
                    candidate.ensemble_score
                    if ranking_mode == "ensemble"
                    else candidate.score
                )
                for candidate in candidates
            )
            / len(candidates)
            if candidates
            else 0.0
        ),
        actual_score=0.0 if actual_score is None else actual_score,
        actual_rank=effective_rank,
        eligible_count=effective_eligible_count,
        actual_rank_percentile=min(
            1.0, effective_rank / max(1, effective_eligible_count)
        ),
        direct_random_probability=direct_random_probability,
        group_random_probability=group_random_probability,
    )


def _max_miss_streak(issues: Sequence[DigitWalkForwardIssue]) -> int:
    maximum = 0
    current = 0
    for issue in issues:
        if issue.direct_hit:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _summarize(
    strategy: str,
    issues: list[DigitWalkForwardIssue],
    rule: LotteryRule,
    candidate_count: int,
) -> DigitWalkForwardStrategySummary:
    target_periods = len(issues)
    direct_hits = sum(issue.direct_hit for issue in issues)
    group_hits = (
        sum(bool(issue.group_hit) for issue in issues) if rule.draw_count == 3 else None
    )
    coverage = {}
    for index, position in enumerate(rule.number_columns):
        hits = sum(issue.position_hits[index] for issue in issues)
        coverage[position] = {
            "hits": hits,
            "rate": hits / target_periods if target_periods else 0.0,
        }
    return DigitWalkForwardStrategySummary(
        strategy=strategy,
        target_periods=target_periods,
        candidate_count=candidate_count,
        direct_hits=direct_hits,
        direct_hit_rate=direct_hits / target_periods if target_periods else 0.0,
        group_hits=group_hits,
        group_hit_rate=(
            (group_hits / target_periods if target_periods else 0.0)
            if group_hits is not None
            else None
        ),
        position_hit_coverage=coverage,
        max_direct_miss_streak=_max_miss_streak(issues),
        relative_to_baseline=None,
        issues=issues,
        mean_candidate_score=(
            sum(issue.candidate_mean_score for issue in issues) / target_periods
            if target_periods
            else 0.0
        ),
        mean_actual_score=(
            sum(issue.actual_score for issue in issues) / target_periods
            if target_periods
            else 0.0
        ),
        mean_actual_rank=(
            sum(issue.actual_rank for issue in issues) / target_periods
            if target_periods
            else 0.0
        ),
    )


def _build_strategy_viability(
    summary: DigitWalkForwardStrategySummary,
) -> PredictionViabilityReport:
    """使用每期真实候选覆盖率构建策略可行性闸门。"""

    direct_hits = [issue.direct_hit for issue in summary.issues]
    direct_probabilities = [issue.direct_random_probability for issue in summary.issues]
    group_hits = (
        [bool(issue.group_hit) for issue in summary.issues]
        if all(issue.group_hit is not None for issue in summary.issues)
        else None
    )
    group_probabilities = (
        [
            float(issue.group_random_probability)
            for issue in summary.issues
            if issue.group_random_probability is not None
        ]
        if group_hits is not None
        else None
    )
    return build_prediction_viability_report(
        direct_hits,
        direct_probabilities,
        group_hits=group_hits,
        group_random_probabilities=group_probabilities,
    )


def _relative_metrics(
    summary: DigitWalkForwardStrategySummary,
    baseline: DigitWalkForwardStrategySummary,
) -> dict[str, float]:
    current_position_rate = sum(
        float(value["rate"]) for value in summary.position_hit_coverage.values()
    ) / len(summary.position_hit_coverage)
    baseline_position_rate = sum(
        float(value["rate"]) for value in baseline.position_hit_coverage.values()
    ) / len(baseline.position_hit_coverage)
    metrics = {
        "directHitRateDiff": summary.direct_hit_rate - baseline.direct_hit_rate,
        "positionCoverageRateDiff": current_position_rate - baseline_position_rate,
        "maxDirectMissStreakDiff": float(
            summary.max_direct_miss_streak - baseline.max_direct_miss_streak
        ),
    }
    if summary.group_hit_rate is not None and baseline.group_hit_rate is not None:
        metrics["groupHitRateDiff"] = summary.group_hit_rate - baseline.group_hit_rate
    return metrics


def _tuning_profiles(base: DigitCandidateConfig) -> dict[str, DigitCandidateConfig]:
    """返回保守且固定的内层候选配置，禁止根据外层结果临时扩展。"""

    profiles = {
        "marginal_only": replace(
            base,
            marginal_weight=1.0,
            pair_weight=0.0,
            shape_weight=0.0,
            sum_weight=0.0,
            span_weight=0.0,
            omission_weight=min(base.omission_weight, 0.03),
        ),
        "joint_balanced": replace(
            base,
            marginal_weight=1.0,
            pair_weight=1.0,
            shape_weight=0.2,
            sum_weight=0.15,
            span_weight=0.1,
            omission_weight=min(base.omission_weight, 0.05),
        ),
        "joint_heavy": replace(
            base,
            marginal_weight=0.8,
            pair_weight=2.0,
            shape_weight=0.25,
            sum_weight=0.2,
            span_weight=0.15,
            omission_weight=min(base.omission_weight, 0.02),
        ),
    }
    for window in (30, 50, 100, 300):
        profiles[f"window_{window}"] = replace(
            base,
            frequency_windows=(window,),
            frequency_window_weights=(1.0,),
        )
    return profiles


def _window_profiles(
    base: DigitCandidateConfig,
    train_size: int,
) -> dict[str, DigitCandidateConfig]:
    profiles = {
        f"{window}": replace(
            base,
            frequency_windows=(window,),
            frequency_window_weights=(1.0,),
            ranking_mode="composite",
        )
        for window in (30, 50, 100, 300)
    }
    profiles["all"] = replace(
        base,
        frequency_windows=(max(1, train_size),),
        frequency_window_weights=(1.0,),
        ranking_mode="composite",
    )
    return profiles


def _select_nested_config(
    train: pd.DataFrame,
    rule: LotteryRule,
    base_config: DigitCandidateConfig,
    inner_validation_periods: int,
) -> tuple[str, DigitCandidateConfig]:
    """仅使用外层训练集执行内层尾部验证。"""

    profiles = _tuning_profiles(base_config)
    profiles["window_all"] = replace(
        base_config,
        frequency_windows=(len(train),),
        frequency_window_weights=(1.0,),
    )
    if len(train) < 3:
        return "joint_balanced", profiles["joint_balanced"]
    validation_count = min(max(1, inner_validation_periods), len(train) - 1)
    validation_start = len(train) - validation_count
    scores: dict[str, tuple[int, int, float]] = {}
    for name, profile in profiles.items():
        direct_hits = 0
        group_hits = 0
        rank_percentiles: list[float] = []
        for target_index in range(validation_start, len(train)):
            inner_train = train.iloc[:target_index].copy()
            target = train.iloc[target_index]
            stats = analyze_digit_history(
                inner_train, rule, frequency_windows=profile.frequency_windows
            )
            actual_numbers = [int(target[column]) for column in rule.number_columns]
            actual_text = _candidate_text(actual_numbers)
            rank, _, eligible_count = rank_digit_numbers_with_eligible_count(
                stats, rule, actual_numbers, profile
            )
            rank_percentiles.append(min(1.0, rank / max(1, eligible_count)))
            plan = generate_digit_betting_candidates(
                stats,
                rule,
                config=profile,
                group_count=profile.count,
            )
            direct_hits += actual_text in {
                candidate.text for candidate in plan.direct_candidates
            }
            if rule.draw_count == 3:
                group_hits += _group_key(actual_text) in {
                    candidate.group_key for candidate in plan.group_candidates
                }
        mean_rank_percentile = (
            sum(rank_percentiles) / len(rank_percentiles)
            if rank_percentiles
            else math.inf
        )
        scores[name] = (direct_hits, group_hits, -mean_rank_percentile)
    selected_name = max(
        scores, key=lambda name: (scores[name], name == "joint_balanced", name)
    )
    return selected_name, profiles[selected_name]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "q05": _quantile(values, 0.05),
        "q95": _quantile(values, 0.95),
    }


def _score_bucket_distribution(
    issues: Sequence[DigitWalkForwardIssue],
) -> list[dict[str, Any]]:
    """按模型排名分位统计真实开奖号落点。"""

    buckets = [
        ("top_1pct", 0.00, 0.01),
        ("top_5pct", 0.01, 0.05),
        ("top_10pct", 0.05, 0.10),
        ("middle_40pct", 0.10, 0.50),
        ("bottom_50pct", 0.50, 1.00),
    ]
    total = len(issues)
    rows: list[dict[str, Any]] = []
    for label, lower, upper in buckets:
        if label == "top_1pct":
            matched = [
                issue for issue in issues if issue.actual_rank_percentile <= upper
            ]
        else:
            matched = [
                issue
                for issue in issues
                if issue.actual_rank_percentile > lower
                and issue.actual_rank_percentile <= upper
            ]
        rows.append(
            {
                "bucket": label,
                "rankPercentileRange": [lower, upper],
                "hits": len(matched),
                "rate": len(matched) / total if total else 0.0,
                "expectedRate": upper - lower,
            }
        )
    return rows


def _random_baseline_distribution(
    current: DigitWalkForwardStrategySummary,
    baselines: Sequence[DigitWalkForwardStrategySummary],
    *,
    include_candidate_score: bool = True,
) -> dict[str, Any]:
    direct_hits = [float(summary.direct_hits) for summary in baselines]
    group_hits = [float(summary.group_hits or 0) for summary in baselines]
    candidate_scores = [summary.mean_candidate_score for summary in baselines]

    def percentile(value: float, samples: Sequence[float]) -> float:
        less = sum(sample < value for sample in samples)
        equal = sum(sample == value for sample in samples)
        return 100.0 * (less + 0.5 * equal) / len(samples) if samples else 0.0

    payload: dict[str, Any] = {
        "runs": len(baselines),
        "directHits": _distribution(direct_hits),
        "currentStrategyPercentile": percentile(
            float(current.direct_hits), direct_hits
        ),
    }
    if include_candidate_score:
        payload["meanCandidateScore"] = _distribution(candidate_scores)
        payload["candidateScorePercentile"] = percentile(
            current.mean_candidate_score, candidate_scores
        )
    if current.group_hits is not None:
        payload["groupHits"] = _distribution(group_hits)
        payload["groupHitPercentile"] = percentile(
            float(current.group_hits), group_hits
        )
    return payload


def _aggregate_random_baselines(
    baselines: Sequence[DigitWalkForwardStrategySummary],
) -> DigitWalkForwardStrategySummary:
    """把多次随机运行汇总成报告表中的均值基线。"""

    if not baselines:
        raise ValueError("随机基线汇总不能为空")
    first = baselines[0]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    coverage = {
        position: {
            "hits": mean(
                [
                    float(summary.position_hit_coverage[position]["hits"])
                    for summary in baselines
                ]
            ),
            "rate": mean(
                [
                    float(summary.position_hit_coverage[position]["rate"])
                    for summary in baselines
                ]
            ),
        }
        for position in first.position_hit_coverage
    }
    group_hits = None
    group_hit_rate = None
    if first.group_hits is not None:
        group_hits = mean([float(summary.group_hits or 0.0) for summary in baselines])
        group_hit_rate = mean(
            [float(summary.group_hit_rate or 0.0) for summary in baselines]
        )
    return replace(
        first,
        direct_hits=mean([float(summary.direct_hits) for summary in baselines]),
        direct_hit_rate=mean([summary.direct_hit_rate for summary in baselines]),
        group_hits=group_hits,
        group_hit_rate=group_hit_rate,
        position_hit_coverage=coverage,
        max_direct_miss_streak=mean(
            [float(summary.max_direct_miss_streak) for summary in baselines]
        ),
        relative_to_baseline=None,
        mean_candidate_score=mean(
            [summary.mean_candidate_score for summary in baselines]
        ),
        mean_actual_score=first.mean_actual_score,
        mean_actual_rank=first.mean_actual_rank,
    )


def run_digit_walk_forward_backtest(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    periods: int = 50,
    min_train_size: int = 100,
    candidate_count: int = 10,
    config: DigitCandidateConfig | None = None,
    baseline_seed: int = 20260715,
    baseline_runs: int = 20,
    nested_tuning: bool = False,
    inner_validation_periods: int = 10,
    advanced_models: bool = False,
    monte_carlo_simulations: int = 5_000,
    ml_training_periods: int = 30,
    ml_negative_samples: int = 5,
    compare_windows: bool = False,
) -> DigitWalkForwardReport:
    """执行数字彩严格逐期前推回测。

    示例：``run_digit_walk_forward_backtest(df, rule, periods=30, min_train_size=100)``。
    """

    if periods <= 0 or min_train_size <= 0 or candidate_count <= 0:
        raise ValueError("periods、min_train_size、candidate_count 必须为正整数")
    if baseline_runs <= 0 or inner_validation_periods <= 0:
        raise ValueError("baseline_runs、inner_validation_periods 必须为正整数")
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    available_targets = len(chronological) - min_train_size
    if available_targets <= 0:
        raise ValueError(f"历史数据不足：至少需要 {min_train_size + 1} 期")
    selected_periods = min(periods, available_targets)
    target_indexes = list(
        range(len(chronological) - selected_periods, len(chronological))
    )
    effective_config = replace(
        config or DigitCandidateConfig(),
        count=candidate_count,
        ranking_mode="composite",
    )
    current_issues: list[DigitWalkForwardIssue] = []
    ensemble_issues: list[DigitWalkForwardIssue] = []
    window_evidence: dict[str, list[tuple[bool, float]]] = {}
    model_evidence: dict[str, list[bool]] = {}
    model_activation_counts: Counter[str] = Counter()
    baseline_issue_runs: list[list[DigitWalkForwardIssue]] = [
        [] for _ in range(baseline_runs)
    ]

    for target_index in target_indexes:
        train = chronological.iloc[:target_index].copy()
        target = chronological.iloc[target_index]
        selected_name = "configured"
        selected_config = effective_config
        if nested_tuning:
            selected_name, selected_config = _select_nested_config(
                train,
                rule,
                effective_config,
                inner_validation_periods,
            )
        else:
            selected_config = with_all_history_window(selected_config, len(train))
        stats = analyze_digit_history(
            train, rule, frequency_windows=selected_config.frequency_windows
        )
        current_plan = generate_digit_betting_candidates(
            stats,
            rule,
            config=selected_config,
            group_count=candidate_count,
        )
        actual_numbers = [int(target[column]) for column in rule.number_columns]
        if compare_windows:
            for label, window_config in _window_profiles(
                effective_config,
                len(train),
            ).items():
                window_stats = analyze_digit_history(
                    train,
                    rule,
                    frequency_windows=window_config.frequency_windows,
                )
                window_rank, _, window_eligible = (
                    rank_digit_numbers_with_eligible_count(
                        window_stats,
                        rule,
                        actual_numbers,
                        window_config,
                    )
                )
                window_evidence.setdefault(label, []).append(
                    (
                        window_rank <= candidate_count,
                        min(1.0, window_rank / max(1, window_eligible)),
                    )
                )
        actual_rank, actual_score, eligible_count = (
            rank_digit_numbers_with_eligible_count(
                stats,
                rule,
                actual_numbers,
                selected_config,
            )
        )
        current_issues.append(
            _evaluate_issue(
                target,
                train,
                rule,
                current_plan,  # type: ignore[arg-type]
                group_candidate_keys=[
                    candidate.group_key for candidate in current_plan.group_candidates
                ],
                selected_config=selected_name,
                selected_config_train_end_issue=str(train.iloc[-1]["期数"]),
                actual_rank=actual_rank,
                actual_score=actual_score,
                eligible_count=eligible_count,
                ranking_mode="composite",
            )
        )
        ensemble_config = replace(selected_config, ranking_mode="ensemble")
        external_scores = None
        if advanced_models:
            external_scores, _ = build_advanced_model_scores(
                train,
                stats,
                rule,
                ensemble_config,
                monte_carlo_simulations=monte_carlo_simulations,
                ml_training_periods=ml_training_periods,
                ml_negative_samples=ml_negative_samples,
                seed=int(str(target["期数"])),
            )
        ensemble_plan = generate_digit_betting_candidates(
            stats,
            rule,
            config=ensemble_config,
            group_count=candidate_count,
            external_scores=external_scores,
        )
        model_activation_counts.update(ensemble_plan.model_candidates.keys())
        actual_text = _candidate_text(actual_numbers)
        for model_name, model_candidates in ensemble_plan.model_candidates.items():
            model_evidence.setdefault(model_name, []).append(
                actual_text in model_candidates
            )
        ensemble_rank, ensemble_score, ensemble_eligible_count = (
            rank_digit_numbers_with_eligible_count(
                stats,
                rule,
                actual_numbers,
                ensemble_config,
                external_scores,
            )
        )
        ensemble_issues.append(
            _evaluate_issue(
                target,
                train,
                rule,
                ensemble_plan,  # type: ignore[arg-type]
                group_candidate_keys=[
                    candidate.group_key for candidate in ensemble_plan.group_candidates
                ],
                selected_config="ensemble_voting",
                selected_config_train_end_issue=str(train.iloc[-1]["期数"]),
                actual_rank=ensemble_rank,
                actual_score=ensemble_score,
                eligible_count=ensemble_eligible_count,
                ranking_mode="ensemble",
            )
        )
        for run_index in range(baseline_runs):
            baseline_plan = generate_uniform_digit_betting_candidates(
                stats,
                rule,
                config=selected_config,
                group_count=candidate_count,
                seed=f"{baseline_seed}:{run_index}:{rule.code}:{target['期数']}",
            )
            baseline_issue_runs[run_index].append(
                _evaluate_issue(
                    target,
                    train,
                    rule,
                    baseline_plan,  # type: ignore[arg-type]
                    group_candidate_keys=[
                        candidate.group_key
                        for candidate in baseline_plan.group_candidates
                    ],
                    selected_config="uniform_random",
                    selected_config_train_end_issue=str(train.iloc[-1]["期数"]),
                    actual_rank=actual_rank,
                    actual_score=actual_score,
                    eligible_count=eligible_count,
                    ranking_mode="composite",
                )
            )

    current = _summarize("current_statistics", current_issues, rule, candidate_count)
    ensemble = _summarize("ensemble_voting", ensemble_issues, rule, candidate_count)
    baseline_summaries = [
        _summarize("uniform_random", issues, rule, candidate_count)
        for issues in baseline_issue_runs
    ]
    baseline = _aggregate_random_baselines(baseline_summaries)
    current = replace(
        current, relative_to_baseline=_relative_metrics(current, baseline)
    )
    ensemble = replace(
        ensemble, relative_to_baseline=_relative_metrics(ensemble, baseline)
    )
    baseline = replace(
        baseline, relative_to_baseline=_relative_metrics(baseline, baseline)
    )
    distribution = _random_baseline_distribution(current, baseline_summaries)
    strategy_distributions = {
        "current_statistics": distribution,
        "ensemble_voting": _random_baseline_distribution(
            ensemble,
            baseline_summaries,
            include_candidate_score=False,
        ),
    }
    score_buckets = {
        "current_statistics": _score_bucket_distribution(current.issues),
        "ensemble_voting": _score_bucket_distribution(ensemble.issues),
    }
    window_comparison: list[dict[str, Any]] = []
    for label, evidence in window_evidence.items():
        target_count = len(evidence)
        top_hits = sum(hit for hit, _ in evidence)
        mean_rank_percentile = sum(rank for _, rank in evidence) / target_count
        window_comparison.append(
            {
                "window": label,
                "targetPeriods": target_count,
                "topCandidateHits": top_hits,
                "topCandidateHitRate": top_hits / target_count,
                "meanRankPercentile": mean_rank_percentile,
                "stabilityScore": top_hits / target_count - mean_rank_percentile,
            }
        )
    window_comparison.sort(
        key=lambda item: (-float(item["stabilityScore"]), str(item["window"]))
    )
    model_performance = {
        name: {
            "sampleCount": len(values),
            "hits": sum(values),
            "hitRate": sum(values) / len(values),
        }
        for name, values in model_evidence.items()
        if values
    }
    strategy_viability = {
        "current_statistics": _build_strategy_viability(current),
        "ensemble_voting": _build_strategy_viability(ensemble),
    }
    return DigitWalkForwardReport(
        rule_code=rule.code,
        display_name=rule.display_name,
        period_count=selected_periods,
        min_train_size=min_train_size,
        candidate_count=candidate_count,
        strategy_summaries=[current, ensemble, baseline],
        baseline_runs=baseline_runs,
        random_baseline_distribution=distribution,
        strategy_baseline_distributions=strategy_distributions,
        score_bucket_distribution=score_buckets["current_statistics"],
        strategy_score_bucket_distributions=score_buckets,
        nested_tuning=nested_tuning,
        inner_validation_periods=inner_validation_periods,
        advanced_models=advanced_models,
        window_comparison=window_comparison,
        model_performance=model_performance,
        active_model_names=tuple(
            name for name in ENSEMBLE_MODEL_NAMES if model_activation_counts[name] > 0
        ),
        available_model_names=ENSEMBLE_MODEL_NAMES,
        model_activation_counts={
            name: model_activation_counts[name] for name in ENSEMBLE_MODEL_NAMES
        },
        strategy_viability=strategy_viability,
    )


def build_digit_walk_forward_markdown(report: DigitWalkForwardReport) -> str:
    """生成数字彩严格逐期前推 Markdown。"""

    lines = [
        f"# {report.display_name}严格逐期前推回测",
        "",
        f"- 目标期数：`{report.period_count}`",
        f"- 最小训练期数：`{report.min_train_size}`",
        f"- 每期候选数：`{report.candidate_count}`",
        f"- 随机基线独立运行：`{report.baseline_runs}` 次",
        f"- 嵌套调参：`{'开启' if report.nested_tuning else '关闭'}`（内层验证 {report.inner_validation_periods} 期）",
        f"- 蒙特卡洛/机器学习外部投票：`{'开启' if report.advanced_models else '关闭'}`",
        f"- 实际启用模型（{len(report.active_model_names)}/{len(report.available_model_names)}）："
        f"`{', '.join(report.active_model_names)}`",
        "- 对比策略：`current_statistics`、`ensemble_voting` 与 `uniform_random`",
        "- 主指标：直选/组选命中与命中随机基线百分位；位置覆盖和候选分位仅作诊断。",
        "",
        "| 策略 | 直选命中 | 三位彩组选命中 | 候选平均排序分 | 目标平均排名 | 最大连续未中 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in report.strategy_summaries:
        group_text = (
            "-"
            if summary.group_hits is None
            else f"{summary.group_hits}/{summary.target_periods} ({summary.group_hit_rate:.2%})"
        )
        rank_text = (
            f"{summary.mean_actual_rank:.1f}"
            if summary.strategy != "uniform_random"
            else "-"
        )
        lines.append(
            f"| {summary.strategy} | {summary.direct_hits}/{summary.target_periods} "
            f"({summary.direct_hit_rate:.2%}) | {group_text} | "
            f"{summary.mean_candidate_score:.4f} | {rank_text} | "
            f"{summary.max_direct_miss_streak} |"
        )
    lines.extend(
        [
            "",
            "## 统计可行性闸门",
            "",
            "只有同时满足至少 500 期、单侧 p<0.01、相对随机提升至少 25%、"
            "99% Wilson 下界高于随机基准、3 个非重叠时间块均不低于随机时才通过。"
            "三位彩还要求直选和组选分别通过。",
            "",
            "| 策略 | 整体 | 直选 | 组选 | 判定原因 |",
            "|---|---|---|---|---|",
        ]
    )
    for strategy, viability in report.strategy_viability.items():
        group_status = (
            "-"
            if viability.group_gate is None
            else ("通过" if viability.group_gate.viable else "不通过")
        )
        lines.append(
            f"| {strategy} | {'通过' if viability.viable else '不通过'} | "
            f"{'通过' if viability.direct_gate.viable else '不通过'} | "
            f"{group_status} | {viability.reason} |"
        )
    for strategy, viability in report.strategy_viability.items():
        lines.extend(["", f"### `{strategy}` 明细", ""])
        for label, metric in (
            ("直选", viability.direct_gate),
            ("组选", viability.group_gate),
        ):
            if metric is None:
                continue
            relative_lift = (
                "不可计算"
                if metric.relative_lift is None
                else f"{metric.relative_lift:.2%}"
            )
            lines.append(
                f"- {label}：实际 `{metric.hits}/{metric.periods}`，随机期望 "
                f"`{metric.expected_random_hits:.2f}`，相对提升 `{relative_lift}`，"
                f"单侧 p 值 `{metric.p_value:.6f}`，99% Wilson 下界 "
                f"`{metric.wilson_lower_bound_99:.4%}`。"
            )
            block_text = "；".join(
                f"块{block.index} {block.hits}/{block.periods} "
                f"(随机期望{block.expected_random_hits:.2f}, "
                f"{'达标' if block.meets_random_baseline else '未达标'})"
                for block in metric.blocks
            )
            lines.append(f"- {label}时间稳定性：{block_text}。")
    distribution = report.random_baseline_distribution
    direct_distribution = distribution.get("directHits", {})
    lines.extend(
        [
            "",
            "## 多随机基线分布",
            "",
            f"- 直选命中均值 / 5% / 95%："
            f"`{direct_distribution.get('mean', 0.0):.2f}` / "
            f"`{direct_distribution.get('q05', 0.0):.2f}` / "
            f"`{direct_distribution.get('q95', 0.0):.2f}`",
            f"- 直选命中随机百分位：`{distribution.get('currentStrategyPercentile', 0.0):.1f}%`",
            f"- 选择器内部诊断 candidateScorePercentile：`{distribution.get('candidateScorePercentile', 0.0):.1f}%`",
            "- 诊断解释：使用 mid-rank；当所有随机运行的候选均分打平时，mid-rank 50% 是正常结果，不代表预测优势。",
        ]
    )
    if "groupHitPercentile" in distribution:
        lines.append(
            f"- 组选命中随机百分位：`{distribution['groupHitPercentile']:.1f}%`"
        )
    ensemble_distribution = report.strategy_baseline_distributions.get(
        "ensemble_voting", {}
    )
    if ensemble_distribution:
        lines.append(
            f"- 集成投票直选命中随机百分位：`{ensemble_distribution.get('currentStrategyPercentile', 0.0):.1f}%`"
        )
        if "groupHitPercentile" in ensemble_distribution:
            lines.append(
                f"- 集成投票组选命中随机百分位：`{ensemble_distribution['groupHitPercentile']:.1f}%`"
            )
    if report.strategy_score_bucket_distributions:
        lines.extend(
            [
                "",
                "## 分位桶诊断",
                "",
                "真实开奖号如果经常落在 Top 1% / Top 5% 分位，才说明模型排序有可验证信号；否则高分候选只是历史拟合。",
            ]
        )
        for strategy, buckets in report.strategy_score_bucket_distributions.items():
            lines.extend(
                [
                    "",
                    f"### `{strategy}`",
                    "",
                    "| 模型排名分位 | 实际落点 | 实际占比 | 理论占比 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for bucket in buckets:
                lower, upper = bucket["rankPercentileRange"]
                lines.append(
                    f"| {lower:.0%}-{upper:.0%} | "
                    f"{bucket['hits']}/{report.period_count} | "
                    f"{bucket['rate']:.2%} | {bucket['expectedRate']:.2%} |"
                )
    if report.nested_tuning:
        selected = report.strategy_summaries[0].issues
        lines.extend(["", "## 嵌套调参证据", ""])
        for issue in selected:
            lines.append(
                f"- 目标期 `{issue.issue}`：选择 `{issue.selected_config}`，配置训练截止 `{issue.selected_config_train_end_issue}`。"
            )
    if report.window_comparison:
        lines.extend(
            [
                "",
                "## 独立窗口稳定性比较",
                "",
                "| 窗口 | Top候选命中 | 平均目标排名分位 | 稳定分 |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in report.window_comparison:
            lines.append(
                f"| {item['window']} | {item['topCandidateHits']}/{item['targetPeriods']} "
                f"({item['topCandidateHitRate']:.2%}) | {item['meanRankPercentile']:.2%} | "
                f"{item['stabilityScore']:.4f} |"
            )
    if report.model_performance:
        lines.extend(
            [
                "",
                "## 逐模型前推表现",
                "",
                "| 模型 | Top候选命中 | 样本 | 命中率 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in report.model_performance.items():
            lines.append(
                f"| {name} | {values['hits']} | {values['sampleCount']} | "
                f"{float(values['hitRate']):.2%} |"
            )
    lines.extend(["", "## 各位置命中覆盖（诊断项）", ""])
    for summary in report.strategy_summaries:
        coverage = "、".join(
            f"{position} {value['hits']}/{summary.target_periods} ({float(value['rate']):.2%})"
            for position, value in summary.position_hit_coverage.items()
        )
        lines.append(f"- `{summary.strategy}`：{coverage}")
    lines.extend(["", "## 相对随机基线", ""])
    for summary in report.strategy_summaries:
        if summary.strategy == "uniform_random":
            continue
        relative = summary.relative_to_baseline or {}
        lines.append(
            f"- `{summary.strategy}`：直选命中率差 `{relative.get('directHitRateDiff', 0.0):+.2%}`，"
            f"位置覆盖率差 `{relative.get('positionCoverageRateDiff', 0.0):+.2%}`"
        )
    lines.extend(
        [
            "",
            "> 说明：严格逐期前推避免了目标期之后数据参与训练，但历史结果仍不代表未来；本工具不能保证提高中奖概率。",
            "",
        ]
    )
    return "\n".join(lines)


def write_digit_walk_forward_reports(
    report: DigitWalkForwardReport,
    output_dir: str | Path,
    *,
    stem_prefix: str = "digit_walk_forward",
) -> tuple[Path, Path]:
    """写出 JSON 与 Markdown 报告。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = (
        stem_prefix.strip().replace("/", "_").replace("\\", "_") or "digit_walk_forward"
    )
    stem = f"{safe_prefix}_{report.rule_code}"
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"
    markdown_path.write_text(
        build_digit_walk_forward_markdown(report), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown_path, json_path
