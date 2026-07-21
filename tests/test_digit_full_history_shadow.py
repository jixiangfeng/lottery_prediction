# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.digit_full_history_shadow import (
    FullHistoryShadowConfig,
    shadow_state_sha256,
    train_full_history_shadow,
    validate_locked_shadow_state,
    write_locked_shadow_state,
)
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


def test_full_history_shadow_trains_every_available_target_and_locks_future_queue(
    tmp_path,
):
    history = _history(130)
    result = train_full_history_shadow(
        history,
        get_lottery_rule("pl3"),
        FullHistoryShadowConfig(
            warmup_history=50,
            search_lookback=30,
            validation_lookback=20,
            calibration_interval=10,
            prospective_periods=500,
            learning_rates=(0.0, 0.02),
            shrinkages=(0.0, 0.25),
            weight_half_life=30,
        ),
    )
    payload = result.to_dict()
    assert payload["trainingStartIndex"] == 50
    assert payload["trainingEndIndex"] == 130
    assert payload["updatesPerCandidate"] == 80
    assert payload["latestHistoryIssue"] == str(2026130)
    assert payload["config"]["featureWindows"] == [20, 50, 150, 300, 500]
    assert payload["historicalReplacementFrozenAllowed"] is False
    assert payload["formalPredictionActivated"] is False
    assert payload["prospectiveValidation"] == {
        "status": "collecting",
        "requiredPeriods": 500,
        "observedPeriods": 0,
        "startAfterIssue": str(2026130),
        "parameterChangesAllowed": False,
    }
    assert len(payload["candidateStates"]) == 4
    candidates = payload["nextPrediction"]["researchTop50"]
    assert len(candidates) == 50
    assert len(set(candidates)) == 50
    latest_row = history.iloc[-1]
    latest_exact = "".join(
        str(int(latest_row[column])) for column in ("百位", "十位", "个位")
    )
    assert latest_exact not in candidates
    assert sum(len(set(candidate)) == 1 for candidate in candidates) <= 1

    destination = tmp_path / "shadow.json"
    write_locked_shadow_state(result, destination)
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["stateSha256"]
    assert shadow_state_sha256(stored) == stored["stateSha256"]
    assert validate_locked_shadow_state(stored, expected_lottery="pl3") == stored

    changed = dict(stored)
    changed["latestHistoryIssue"] = "9999999"
    with pytest.raises(ValueError, match="内容指纹不匹配"):
        validate_locked_shadow_state(changed, expected_lottery="pl3")

    stale = dict(stored)
    stale["sourceFingerprint"] = "stale"
    with pytest.raises(ValueError, match="源码指纹不匹配"):
        validate_locked_shadow_state(stale, expected_lottery="pl3")

    with pytest.raises(ValueError, match="彩种与请求不一致"):
        validate_locked_shadow_state(stored, expected_lottery="fc3d")
    with pytest.raises(RuntimeError, match="已存在"):
        write_locked_shadow_state(result, destination)
