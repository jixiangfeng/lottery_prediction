# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 特征测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.digit_learned_features import (
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
    decay_weight,
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
        "shape_distribution",
        "sum_distribution",
        "span_distribution",
        "parity_bigsmall",
        "recent_trend",
        "latest_distance",
        "repeat_latest",
        "omission_rebound",
        "constraint_penalty",
    }.issubset(features.columns)
    assert {"omission_10", "omission_30", "omission_all"}.issubset(features.columns)


def test_recency_regime_features_are_present_and_long_history_is_not_equal_weighted():
    rule = get_lottery_rule("fc3d")
    config = LearnedFeatureConfig(windows=(10, 30, 50, 100, 150, 300, "all"))
    state = build_history_state(_history(180), rule, config)
    features = build_candidate_features(state, rule)

    assert {"regime_gap_50_all", "regime_gap_100_all", "regime_gap_150_all"}.issubset(
        features.columns
    )
    assert all(
        config.window_weight_map()[str(window)] > config.window_weight_map()["all"]
        for window in (10, 30, 50, 100, 150, 300)
    )
    assert np.isfinite(
        features[
            ["regime_gap_50_all", "regime_gap_100_all", "regime_gap_150_all"]
        ].to_numpy()
    ).all()
    assert features["regime_gap_50_all"].nunique() > 1


def test_decay_weight_decreases_monotonically_with_age():
    values = [decay_weight(age, 20.0) for age in range(1, 50)]
    assert all(left > right for left, right in zip(values, values[1:]))
    assert decay_weight(1, None) == decay_weight(100, None) == 1.0
