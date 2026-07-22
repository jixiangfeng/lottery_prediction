# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import digit_online_gradient_variants as variants_module
from src.analysis.digit_learned_features import BEHAVIORAL_FEATURE_NAMES, FEATURE_NAMES
from src.analysis.digit_online_gradient import (
    OnlineGradientCandidate,
    OnlineGradientConfig,
    online_gradient_step,
    run_online_gradient_research,
)
from src.analysis.digit_online_gradient_variants import (
    run_online_gradient_research_variants,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    state = 90210
    rows = []
    for index in range(periods):
        values = []
        for _ in range(3):
            state = (1664525 * state + 1013904223) % (2**32)
            values.append(state % 10)
        rows.append(
            {
                "期数": str(2026001 + index),
                "百位": values[0],
                "十位": values[1],
                "个位": values[2],
            }
        )
    return pd.DataFrame(rows)


def _repeated_history(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(periods)],
            "百位": [1] * periods,
            "十位": [2] * periods,
            "个位": [3] * periods,
        }
    )


def _loss(
    matrix: np.ndarray,
    weights: np.ndarray,
    actual: int,
    shrinkage: float,
    temperature: float,
) -> float:
    scores = matrix @ weights / temperature
    model = np.exp(scores - scores.max())
    model /= model.sum()
    probability = shrinkage * model[actual] + (1 - shrinkage) / len(model)
    return float(-np.log(probability))


def test_online_gradient_matches_finite_difference():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(1000, len(FEATURE_NAMES)))
    weights = rng.normal(scale=0.2, size=len(FEATURE_NAMES))
    candidate = OnlineGradientCandidate(learning_rate=0.005, uniform_shrinkage=0.75)
    config = OnlineGradientConfig(
        development_end=600,
        temperature=0.8,
        l2_penalty=0,
        gradient_clip=100,
    )
    step = online_gradient_step(matrix, 321, weights, candidate, config)

    epsilon = 1e-6
    numerical = np.empty_like(weights)
    for index in range(len(weights)):
        right = weights.copy()
        left = weights.copy()
        right[index] += epsilon
        left[index] -= epsilon
        numerical[index] = (
            _loss(matrix, right, 321, 0.75, 0.8) - _loss(matrix, left, 321, 0.75, 0.8)
        ) / (2 * epsilon)
    assert np.allclose(step.gradient, numerical, atol=1e-6)


def test_zero_shrinkage_has_zero_feature_gradient():
    matrix = np.arange(1000 * len(FEATURE_NAMES), dtype=float).reshape(
        1000, len(FEATURE_NAMES)
    )
    config = OnlineGradientConfig(development_end=600, l2_penalty=0)
    step = online_gradient_step(
        matrix,
        123,
        np.zeros(len(FEATURE_NAMES)),
        OnlineGradientCandidate(learning_rate=0.01, uniform_shrinkage=0.0),
        config,
    )
    assert np.allclose(step.final_probabilities, 0.001)
    assert np.allclose(step.gradient, 0)
    assert np.allclose(step.weights_after, 0)


def test_sparse_regularization_and_zeroed_features_are_enforced():
    matrix = np.zeros((1000, len(FEATURE_NAMES)))
    weights = np.ones(len(FEATURE_NAMES))
    weights[FEATURE_NAMES.index("constraint_penalty")] = -1.0
    config = OnlineGradientConfig(
        development_end=600,
        learning_rates=(0.1,),
        shrinkages=(1.0,),
        l2_penalty=0.1,
        gradient_clip=100,
    )
    step = online_gradient_step(
        matrix,
        123,
        weights,
        OnlineGradientCandidate(learning_rate=0.1, uniform_shrinkage=1.0),
        config,
    )
    assert np.isclose(
        step.weights_after[FEATURE_NAMES.index("position_frequency")], 0.9
    )
    assert np.isclose(step.weights_after[FEATURE_NAMES.index("sum_distribution")], 0.95)
    assert np.isclose(
        step.weights_after[FEATURE_NAMES.index("position_omission")], 0.99
    )
    assert step.weights_after[FEATURE_NAMES.index("shape_transition")] == 0.0
    assert step.weights_after[FEATURE_NAMES.index("shape_recent_deviation")] == 0.0


def test_behavioral_features_use_configured_l2_without_changing_core_defaults():
    feature_names = (*FEATURE_NAMES, *BEHAVIORAL_FEATURE_NAMES)
    matrix = np.zeros((1000, len(feature_names)), dtype=float)
    weights = np.zeros(len(feature_names), dtype=float)
    for name in BEHAVIORAL_FEATURE_NAMES:
        weights[feature_names.index(name)] = 1.0
    config = OnlineGradientConfig(
        development_end=600,
        feature_names=feature_names,
        feature_l2_multipliers=tuple((name, 2.0) for name in BEHAVIORAL_FEATURE_NAMES),
        zeroed_features=(),
        l2_penalty=0.1,
        gradient_clip=100,
    )

    step = online_gradient_step(
        matrix,
        321,
        weights,
        OnlineGradientCandidate(learning_rate=0.1, uniform_shrinkage=0.5),
        config,
    )

    for name in BEHAVIORAL_FEATURE_NAMES:
        assert step.weights_after[feature_names.index(name)] == 0.98
    assert OnlineGradientConfig(development_end=600).feature_names == FEATURE_NAMES


def test_behavioral_gradient_clip_does_not_reduce_core_gradient_budget():
    core_matrix = np.zeros((1000, len(FEATURE_NAMES)), dtype=float)
    core_index = FEATURE_NAMES.index("position_frequency")
    core_matrix[321, core_index] = 10.0
    candidate = OnlineGradientCandidate(learning_rate=0.1, uniform_shrinkage=1.0)
    core_config = OnlineGradientConfig(
        development_end=600,
        zeroed_features=(),
        l2_penalty=0.0,
        gradient_clip=1.0,
    )
    core_step = online_gradient_step(
        core_matrix,
        321,
        np.zeros(len(FEATURE_NAMES)),
        candidate,
        core_config,
    )

    feature_names = (*FEATURE_NAMES, BEHAVIORAL_FEATURE_NAMES[0])
    mixed_matrix = np.zeros((1000, len(feature_names)), dtype=float)
    mixed_matrix[:, : len(FEATURE_NAMES)] = core_matrix
    mixed_matrix[321, -1] = 1000.0
    mixed_config = OnlineGradientConfig(
        development_end=600,
        feature_names=feature_names,
        feature_l2_multipliers=((BEHAVIORAL_FEATURE_NAMES[0], 2.0),),
        zeroed_features=(),
        l2_penalty=0.0,
        gradient_clip=1.0,
        behavioral_gradient_clip=0.25,
    )
    mixed_step = online_gradient_step(
        mixed_matrix,
        321,
        np.zeros(len(feature_names)),
        candidate,
        mixed_config,
    )

    assert np.isclose(
        mixed_step.weights_after[core_index], core_step.weights_after[core_index]
    )
    assert abs(mixed_step.clipped_gradient[-1]) <= 0.25


def test_monotonic_behavior_constraints_block_positive_risk_weights():
    feature_names = (*FEATURE_NAMES, *BEHAVIORAL_FEATURE_NAMES)
    matrix = np.zeros((1000, len(feature_names)), dtype=float)
    risk_index = feature_names.index(BEHAVIORAL_FEATURE_NAMES[0])
    matrix[321, risk_index] = 1.0
    weights = np.zeros(len(feature_names), dtype=float)
    candidate = OnlineGradientCandidate(learning_rate=0.1, uniform_shrinkage=1.0)
    common = dict(
        development_end=600,
        feature_names=feature_names,
        zeroed_features=(),
        l2_penalty=0.0,
        gradient_clip=100.0,
    )

    unconstrained = online_gradient_step(
        matrix,
        321,
        weights,
        candidate,
        OnlineGradientConfig(**common),
    )
    constrained = online_gradient_step(
        matrix,
        321,
        weights,
        candidate,
        OnlineGradientConfig(
            **common,
            nonpositive_features=BEHAVIORAL_FEATURE_NAMES,
        ),
    )

    assert unconstrained.weights_after[risk_index] > 0
    assert constrained.weights_after[risk_index] == 0


def test_online_gradient_research_is_prequential_and_recalibrates():
    report = run_online_gradient_research(
        _history(130),
        get_lottery_rule("fc3d"),
        OnlineGradientConfig(
            development_end=120,
            outer_periods=20,
            calibration_interval=10,
            search_lookback=30,
            validation_lookback=20,
            warmup_history=50,
            learning_rates=(0.0, 0.005),
            shrinkages=(0.0, 0.5),
        ),
    )
    assert report.frozen_test_read is False
    assert [item.target_index for item in report.periods] == list(range(100, 120))
    assert [item.block_start_index for item in report.selections] == [100, 110]
    assert all(item.target_issue != item.history_end_issue for item in report.periods)
    assert all(set(item.gradients) == set(FEATURE_NAMES) for item in report.periods)
    assert all(
        set(item.weights_before) == set(FEATURE_NAMES) for item in report.periods
    )
    assert report.to_dict()["metrics"]["periods"] == 20


def test_online_gradient_can_evaluate_the_same_policy_used_by_daily_top50():
    report = run_online_gradient_research(
        _repeated_history(130),
        get_lottery_rule("fc3d"),
        OnlineGradientConfig(
            development_end=120,
            outer_periods=20,
            calibration_interval=10,
            search_lookback=30,
            validation_lookback=20,
            warmup_history=50,
            learning_rates=(0.0, 0.005),
            shrinkages=(0.0, 0.5),
            daily_candidate_policy=True,
            maximum_top50_triples=0,
        ),
    )

    payload = report.to_dict()
    assert payload["config"]["dailyCandidatePolicy"] is True
    assert payload["config"]["maximumTop50Triples"] == 0
    assert all(item.candidate_policy_rank is None for item in report.periods)
    assert not any(item.research_direct_hit for item in report.periods)
    assert all(item.top50_shape_counts["豹子"] == 0 for item in report.periods)


def test_variant_runner_builds_each_target_feature_matrix_once_and_reports_progress(
    monkeypatch,
):
    base = OnlineGradientConfig(
        development_end=120,
        outer_periods=20,
        calibration_interval=10,
        search_lookback=30,
        validation_lookback=20,
        warmup_history=50,
        learning_rates=(0.0, 0.005),
        shrinkages=(0.0, 0.5),
    )
    behavior_names = (*FEATURE_NAMES, "exact_recency_risk")
    behavior = OnlineGradientConfig(
        **{
            **base.__dict__,
            "feature_names": behavior_names,
            "behavioral_gradient_clip": 0.25,
        }
    )
    monotonic = OnlineGradientConfig(
        **{
            **behavior.__dict__,
            "nonpositive_features": ("exact_recency_risk",),
        }
    )
    build_calls = 0
    original = variants_module.build_candidate_features

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(variants_module, "build_candidate_features", counted_build)
    progress: list[dict[str, object]] = []

    reports = run_online_gradient_research_variants(
        _history(130),
        get_lottery_rule("fc3d"),
        {"A": base, "B": behavior, "C": monotonic},
        progress_callback=progress.append,
        progress_interval=10,
    )

    assert build_calls == 70
    assert set(reports) == {"A", "B", "C"}
    assert all(len(report.periods) == 20 for report in reports.values())
    assert [item["processedOuterPeriods"] for item in progress] == [10, 20]
    assert all(item["totalOuterPeriods"] == 20 for item in progress)
    assert progress[-1]["completedFixedBlocks"] == 2
