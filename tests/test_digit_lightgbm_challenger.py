# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.digit_lightgbm_challenger import (
    LightGBMChallengeConfig,
    LightGBMParams,
    build_lightgbm_feature_table,
    joint_digit_probabilities,
    run_lightgbm_block_backtest,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    return pd.DataFrame(
        {
            "期数": [str(2026001 + i) for i in range(periods)],
            "百位": rng.integers(0, 10, periods),
            "十位": rng.integers(0, 10, periods),
            "个位": rng.integers(0, 10, periods),
        }
    )


def test_feature_at_target_does_not_read_target_outcome():
    rule = get_lottery_rule("fc3d")
    original = _history(80)
    changed = original.copy()
    changed.loc[40, ["百位", "十位", "个位"]] = [9, 9, 9]
    first = build_lightgbm_feature_table(original, rule, windows=(10, 20), lag_count=2)
    second = build_lightgbm_feature_table(changed, rule, windows=(10, 20), lag_count=2)
    row = list(first.target_indices).index(40)
    assert np.array_equal(first.features[row], second.features[row])
    assert not np.array_equal(first.features[row + 1], second.features[row + 1])


def test_joint_position_probabilities_form_canonical_1000_distribution():
    position = np.zeros((1, 3, 10), dtype=float)
    position[0, 0, 1] = 1.0
    position[0, 1, 2] = 1.0
    position[0, 2, 3] = 1.0
    joint = joint_digit_probabilities(position)
    assert joint.shape == (1, 1000)
    assert np.isclose(joint.sum(), 1.0)
    assert int(np.argmax(joint[0])) == 123


def test_backtest_reports_every_complete_block_without_formal_activation():
    config = LightGBMChallengeConfig(
        windows=(10, 20),
        lag_count=2,
        minimum_train_periods=40,
        inner_validation_periods=20,
        block_size=20,
        parameter_grid=(
            LightGBMParams(
                name="tiny",
                num_leaves=7,
                max_depth=3,
                min_child_samples=10,
                n_estimators=10,
            ),
        ),
        shrinkages=(0.25,),
    )
    result = run_lightgbm_block_backtest(
        _history(120), get_lottery_rule("pl3"), config
    ).to_dict()
    assert result["evaluationKind"] == "lightgbm_position_multiclass_blocks"
    assert result["selectionPolicy"] == "inner_validation_only"
    assert result["blocksEvaluated"] == 2
    assert all(block["periods"] == 20 for block in result["blocks"])
    assert result["formalPredictionActivated"] is False
    assert result["topK"] == 50
