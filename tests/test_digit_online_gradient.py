# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.digit_learned_features import (
    BEHAVIORAL_FEATURE_NAMES,
    FEATURE_NAMES,
)
from src.analysis.digit_online_gradient import (
    OnlineGradientCandidate,
    OnlineGradientConfig,
    online_gradient_step,
    run_online_gradient_research,
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


def test_behavioral_features_use_tenfold_l2_without_changing_core_defaults():
    feature_names = (*FEATURE_NAMES, *BEHAVIORAL_FEATURE_NAMES)
    matrix = np.zeros((1000, len(feature_names)), dtype=float)
    weights = np.zeros(len(feature_names), dtype=float)
    for name in BEHAVIORAL_FEATURE_NAMES:
        weights[feature_names.index(name)] = 1.0
    config = OnlineGradientConfig(
        development_end=600,
        feature_names=feature_names,
        feature_l2_multipliers=tuple((name, 10.0) for name in BEHAVIORAL_FEATURE_NAMES),
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
        assert step.weights_after[feature_names.index(name)] == 0.9
    assert OnlineGradientConfig(development_end=600).feature_names == FEATURE_NAMES


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
