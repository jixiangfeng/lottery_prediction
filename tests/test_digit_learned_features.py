# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 特征测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.digit_learned_features import (
    BEHAVIORAL_FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
    decay_weight,
    iter_rolling_history_states,
)
from src.lotteries import get_lottery_rule


def _history(periods: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(periods)
        ]
    )


def test_history_state_strictly_excludes_target_and_future_rows():
    rule = get_lottery_rule("fc3d")
    history = _history(12)

    state = build_history_state(
        history,
        rule,
        LearnedFeatureConfig(windows=(5, "all")),
        target_issue="2026008",
    )

    assert state.history_issues == tuple(str(2026001 + index) for index in range(7))
    assert state.history_end_issue == "2026007"
    assert state.target_issue == "2026008"


def test_candidate_features_cover_all_1000_candidates_and_are_finite():
    rule = get_lottery_rule("pl3")
    state = build_history_state(
        _history(), rule, LearnedFeatureConfig(windows=(10, 30, "all"), alpha=2.0)
    )

    features = build_candidate_features(state, rule)

    assert features.shape[0] == 1000
    assert features.iloc[0]["candidate"] == "000"
    assert features.iloc[-1]["candidate"] == "999"
    assert features["candidate"].is_unique
    numeric = features.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert {
        "position_frequency",
        "position_omission",
        "pair_frequency",
        "sum_distribution",
        "span_distribution",
        "recent_trend",
        "position_trend",
        "pair_trend",
        "shape_transition",
        "shape_recent_deviation",
        "constraint_penalty",
    }.issubset(features.columns)


def test_recency_features_are_present_and_long_history_is_not_equal_weighted():
    rule = get_lottery_rule("fc3d")
    config = LearnedFeatureConfig()
    state = build_history_state(_history(180), rule, config)
    features = build_candidate_features(state, rule)

    assert len(state.numbers) == 150
    assert np.isfinite(
        features[["position_trend", "shape_transition"]].to_numpy()
    ).all()


def test_rolling_history_states_match_independent_rebuilds():
    rule = get_lottery_rule("fc3d")
    history = _history(180)
    config = LearnedFeatureConfig()
    indices = (150, 160, 170)
    rolling = tuple(iter_rolling_history_states(history, rule, indices, config))

    for index, state in zip(indices, rolling):
        expected = build_history_state(
            history.iloc[:index],
            rule,
            config,
            target_issue=str(history.iloc[index]["期数"]),
        )
        assert state == expected
        actual_features = build_candidate_features(
            state, rule, candidates=("000", "123", "987")
        )
        expected_features = build_candidate_features(
            expected, rule, candidates=("000", "123", "987")
        )
        np.testing.assert_allclose(
            actual_features.drop(columns="candidate"),
            expected_features.drop(columns="candidate"),
            atol=1e-12,
        )


def test_behavioral_context_features_are_prior_only_and_candidate_specific():
    rule = get_lottery_rule("fc3d")
    history = pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(8)],
            "百位": [1, 4, 1, 7, 1, 3, 2, 1],
            "十位": [2, 5, 3, 8, 2, 2, 3, 2],
            "个位": [3, 6, 2, 9, 3, 1, 1, 3],
        }
    )
    config = LearnedFeatureConfig(windows=(5, "all"))
    state = build_history_state(history, rule, config)

    core = build_candidate_features(state, rule, candidates=("123", "132", "987"))
    behavior = build_candidate_features(
        state,
        rule,
        candidates=("123", "132", "987"),
        include_behavioral_context=True,
    ).set_index("candidate")

    assert not set(BEHAVIORAL_FEATURE_NAMES).intersection(core.columns)
    assert set(BEHAVIORAL_FEATURE_NAMES).issubset(behavior.columns)
    assert (
        behavior.loc["123", "exact_recency_risk"]
        > behavior.loc["132", "exact_recency_risk"]
    )
    assert (
        behavior.loc["132", "group_recency_risk"]
        > behavior.loc["123", "group_recency_risk"]
    )
    assert (
        behavior.loc["123", "last_position_overlap_risk"]
        > behavior.loc["132", "last_position_overlap_risk"]
    )
    assert (
        behavior.loc["132", "last_unordered_overlap_risk"]
        > behavior.loc["123", "last_unordered_overlap_risk"]
    )
    assert np.isfinite(behavior[list(BEHAVIORAL_FEATURE_NAMES)].to_numpy()).all()


def test_behavioral_context_features_have_comparable_per_query_scale():
    rule = get_lottery_rule("pl3")
    state = build_history_state(
        _history(180),
        rule,
        LearnedFeatureConfig(windows=(20, 50, 150)),
    )

    features = build_candidate_features(
        state,
        rule,
        include_behavioral_context=True,
    )

    for name in BEHAVIORAL_FEATURE_NAMES:
        values = features[name].to_numpy(dtype=float)
        assert np.isclose(values.mean(), 0.0, atol=1e-12)
        assert np.isclose(values.std(), 0.0, atol=1e-12) or np.isclose(
            values.std(), 1.0, atol=1e-12
        )


def test_behavioral_context_does_not_read_target_outcome():
    rule = get_lottery_rule("pl3")
    history = _history(30)
    mutated = history.copy()
    mutated.loc[20, ["百位", "十位", "个位"]] = [9, 9, 9]
    config = LearnedFeatureConfig(windows=(10, "all"))
    target_issue = str(history.iloc[20]["期数"])

    original = build_candidate_features(
        build_history_state(history, rule, config, target_issue=target_issue),
        rule,
        candidates=("000", "123", "999"),
        include_behavioral_context=True,
    )
    changed = build_candidate_features(
        build_history_state(mutated, rule, config, target_issue=target_issue),
        rule,
        candidates=("000", "123", "999"),
        include_behavioral_context=True,
    )

    pd.testing.assert_frame_equal(original, changed)


def test_decay_weight_decreases_monotonically_with_age():
    values = [decay_weight(age, 20.0) for age in range(1, 50)]
    assert all(left > right for left, right in zip(values, values[1:]))
    assert decay_weight(1, None) == decay_weight(100, None) == 1.0
