# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_behavioral_context import (
    BehavioralContextConfig,
    run_behavioral_context_challenge,
)
from src.analysis.digit_online_gradient import OnlineGradientConfig
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(periods)],
            "百位": [(index * 7) % 10 for index in range(periods)],
            "十位": [(index * 3 + 1) % 10 for index in range(periods)],
            "个位": [(index * 9 + 2) % 10 for index in range(periods)],
        }
    )


def test_behavioral_context_challenge_runs_paired_prior_only_groups():
    online = OnlineGradientConfig(
        development_end=130,
        outer_periods=30,
        warmup_history=50,
        search_lookback=30,
        validation_lookback=20,
        calibration_interval=10,
        learning_rates=(0.0, 0.02),
        shrinkages=(0.0, 0.25),
    )

    report = run_behavioral_context_challenge(
        _history(130),
        get_lottery_rule("fc3d"),
        BehavioralContextConfig(online=online, paired_permutations=99),
    )

    assert report["modelVersion"] == "behavioral_context_v1"
    assert report["frozenTestRead"] is False
    assert report["groups"]["A"]["periods"] == 30
    assert report["groups"]["B"]["periods"] == 30
    assert set(report["groups"]) == {"A", "B"}
    assert report["comparison"]["pairedPeriods"] == 30
    assert report["behavioralFeatureL2Multiplier"] == 10.0
    assert report["gate"]["passed"] is False
    assert report["gate"]["minimumPeriods"] == 500
