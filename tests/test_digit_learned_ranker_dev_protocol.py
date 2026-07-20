# -*- coding: utf-8 -*-

from src.analysis.digit_learned_ranker_dev_protocol import (
    assess_joint_rolling_gate,
    build_rolling_development_folds,
)


def test_rolling_development_folds_never_touch_frozen_segment():
    folds = build_rolling_development_folds(
        development_end=500,
        min_train_size=100,
        initial_selection_end=200,
        validation_size=50,
        evaluation_stride=5,
    )

    assert len(folds) == 6
    assert folds[0].selection_indices == tuple(range(100, 200, 5))
    assert folds[0].validation_indices == tuple(range(200, 250, 5))
    assert folds[-1].validation_indices == tuple(range(450, 500, 5))
    assert max(index for fold in folds for index in fold.all_target_indices) < 500
    assert all(fold.frozen_test_used is False for fold in folds)


def test_joint_rolling_gate_rejects_when_one_lottery_has_a_weak_fold():
    report = assess_joint_rolling_gate(
        {
            "fc3d": [
                {"fold": 1, "lift": 1.2},
                {"fold": 2, "lift": 1.1},
            ],
            "pl3": [
                {"fold": 1, "lift": 1.3},
                {"fold": 2, "lift": 0.9},
            ],
        },
        minimum_lift=1.0,
    )

    assert report["qualified"] is False
    assert report["worstLottery"] == "pl3"
    assert report["worstFold"] == 2
    assert report["worstFoldLift"] == 0.9
    assert report["frozenTestAllowed"] is False
    assert report["reasons"] == ["pl3 第2折 lift=0.900000 未超过闸门 1.000000"]
