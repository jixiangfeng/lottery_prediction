# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.digit_full_history_shadow import (
    FullHistoryShadowConfig,
    decay_shadow_weights,
)
from src.analysis.digit_historical_blocks import run_historical_block_backtest
from src.analysis.digit_learned_features import FEATURE_NAMES
from src.analysis.digit_online_gradient import OnlineGradientCandidate
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "期数": [str(2026001 + i) for i in range(periods)],
            "百位": [(i * 7) % 10 for i in range(periods)],
            "十位": [(i * 3 + 1) % 10 for i in range(periods)],
            "个位": [(i * 9 + 2) % 10 for i in range(periods)],
        }
    )


def test_shadow_decay_is_shared_and_locks_zeroed_features():
    weights = np.ones(len(FEATURE_NAMES))
    decayed = decay_shadow_weights(
        weights,
        OnlineGradientCandidate(0.02, 0.25),
        300.0,
        ("shape_transition", "shape_recent_deviation"),
    )
    assert decayed[FEATURE_NAMES.index("position_frequency")] < 1.0
    assert decayed[FEATURE_NAMES.index("shape_transition")] == 0.0
    assert decayed[FEATURE_NAMES.index("shape_recent_deviation")] == 0.0
    assert decayed[FEATURE_NAMES.index("constraint_penalty")] == 0.0


def test_all_complete_historical_blocks_are_reported_without_selection():
    result = run_historical_block_backtest(
        _history(200),
        get_lottery_rule("pl3"),
        FullHistoryShadowConfig(
            warmup_history=20,
            search_lookback=20,
            validation_lookback=10,
            calibration_interval=5,
            learning_rates=(0.0, 0.02),
            shrinkages=(0.0, 0.25),
        ),
        block_size=50,
    )
    payload = result.to_dict()
    assert payload["selectionPolicy"] == "all_complete_nonoverlapping_blocks"
    assert payload["blockSize"] == 50
    assert payload["blockSelectionAllowed"] is False
    assert payload["blocksEvaluated"] == 3
    assert len(payload["blocks"]) == 3
    assert all(block["periods"] == 50 for block in payload["blocks"])
    assert payload["evidenceStatus"] == "retrospective_robustness_only"
