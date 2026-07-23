# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.digit_online_gradient import (
    OnlineGradientConfig,
    run_online_gradient_research,
)
from src.analysis.digit_sparse_frozen import (
    claim_frozen_run,
    create_sparse_v4_lock,
    load_and_verify_sparse_v4_lock,
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


def test_sparse_frozen_lock_is_deterministic_and_marker_is_one_shot(tmp_path):
    lock_path = tmp_path / "protocol.json"
    first_path, first_hash = create_sparse_v4_lock(lock_path)
    second_path, second_hash = create_sparse_v4_lock(lock_path)
    payload, loaded_hash = load_and_verify_sparse_v4_lock(lock_path)
    assert first_path == second_path == lock_path
    assert first_hash == second_hash == loaded_hash
    assert payload["gate"]["jointActivationRequiresBothLotteries"] is True

    marker = tmp_path / "fc3d.marker.json"
    claim_frozen_run(marker, "fc3d", first_hash)
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "started"
    with pytest.raises(RuntimeError, match="禁止重复运行"):
        claim_frozen_run(marker, "fc3d", first_hash)


def test_frozen_report_marks_independent_evidence_and_gate():
    report = run_online_gradient_research(
        _history(130),
        get_lottery_rule("pl3"),
        OnlineGradientConfig(
            development_end=120,
            outer_periods=20,
            calibration_interval=10,
            search_lookback=30,
            validation_lookback=20,
            warmup_history=50,
            learning_rates=(0.0, 0.02),
            shrinkages=(0.0, 0.25),
        ),
        frozen_test_read=True,
    ).to_dict()
    assert report["frozenTestRead"] is True
    assert report["evidenceStatus"] == "independent_frozen_test"
    assert report["evaluationKind"] == "frozen_sparse_online_gradient"
    assert report["frozenGate"] is not None
    metrics = report["metrics"]
    assert metrics["researchTop50PValue"] is None
    assert "Frozen没有可评估的非均匀Top50排名" in report["frozenGate"]["reasons"]
