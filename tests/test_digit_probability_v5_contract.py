# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import digit_probability_v5 as probability_v5
from src.analysis import digit_probability_v5_null as null_module
from src.analysis.digit_probability_v5 import (
    build_probability_v5_protocol,
    probability_v5_smoke_config,
    run_probability_v5_development,
    run_registered_probability_v5_development,
    write_probability_v5_protocol,
    write_probability_v5_report,
)
from src.analysis.digit_probability_v5_null import (
    run_formal_probability_v5_null_simulation,
)
from src.lotteries import get_lottery_rule


def _history(periods: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(periods)],
            "百位": [(index * 7) % 10 for index in range(periods)],
            "十位": [(index * 3 + 1) % 10 for index in range(periods)],
            "个位": [(index * 9 + 2) % 10 for index in range(periods)],
        }
    )


def _registered_files(tmp_path: Path):
    history = _history()
    rule = get_lottery_rule("fc3d")
    config = probability_v5_smoke_config()
    protocol_path = tmp_path / "protocol.json"
    report_path = tmp_path / "report.json"
    protocol = build_probability_v5_protocol(
        history, rule, config, frozen_periods_excluded=500
    )
    write_probability_v5_protocol(protocol, protocol_path)
    report = run_registered_probability_v5_development(
        protocol_path,
        history,
        rule,
        config,
        frozen_periods_excluded=500,
    )
    write_probability_v5_report(report, report_path)
    return history, rule, config, protocol_path, report_path, report.to_dict()


def test_generic_development_never_claims_registered_protocol():
    report = run_probability_v5_development(
        _history(),
        get_lottery_rule("fc3d"),
        probability_v5_smoke_config(),
        frozen_periods_excluded=500,
    ).to_dict()

    assert report["protocolIdentity"] is None
    assert report["protocol"]["developmentProtocolRegistered"] is False
    assert not hasattr(probability_v5, "VerifiedProbabilityV5Protocol")


def test_registered_development_loads_exact_read_only_protocol(tmp_path: Path):
    history, rule, config, protocol_path, _, payload = _registered_files(tmp_path)

    assert payload["protocol"]["developmentProtocolRegistered"] is True
    assert protocol_path.stat().st_mode & 0o222 == 0
    protocol_path.chmod(0o644)
    with pytest.raises(PermissionError, match="只读"):
        run_registered_probability_v5_development(
            protocol_path,
            history,
            rule,
            config,
            frozen_periods_excluded=500,
        )


def test_raw_fingerprint_hashes_consumed_float64_array():
    config = probability_v5_smoke_config()
    records = probability_v5._run_prequential(
        _history(), get_lottery_rule("fc3d"), config
    )
    record = records[0]

    expected = hashlib.sha256(record.probabilities.astype("<f8").tobytes()).hexdigest()
    assert record.probabilities.dtype == np.dtype("float64")
    assert record.raw_distribution_fingerprint == expected


def test_null_trial_gate_requires_search_and_evaluation(monkeypatch):
    class FakeReport:
        def to_dict(self):
            return {
                "selectedTemperature": 1.0,
                "search": {"strictStatisticalGatePassed": False},
                "evaluation": {
                    "strictStatisticalGatePassed": True,
                    "meanDeltaLogLoss": 0.1,
                    "meanDeltaBrier": 0.1,
                    "policyTopKHits": 1,
                    "postFinalUpdateExpertWeights": {"uniform": 1.0},
                },
            }

    monkeypatch.setattr(
        null_module, "run_probability_v5_development", lambda *a, **k: FakeReport()
    )
    task = null_module._NullTask(
        lottery="fc3d",
        config=probability_v5_smoke_config(),
        history_periods=50,
        frozen_periods_excluded=500,
        index=0,
    )

    assert null_module._run_null_trial(task).strict_statistical_gate_passed is False


@pytest.mark.parametrize("iterations", [4999, 5001])
def test_formal_entry_requires_exactly_5000_iterations(tmp_path: Path, iterations: int):
    with pytest.raises(ValueError, match="5000"):
        run_formal_probability_v5_null_simulation(
            tmp_path / "protocol.json",
            tmp_path / "report.json",
            _history(),
            lottery="fc3d",
            config=probability_v5_smoke_config(),
            frozen_periods_excluded=500,
            iterations=iterations,
            checkpoint_dir=tmp_path / "checkpoint",
        )


def test_checkpoint_uses_claimed_report_hash_and_protocol_boundaries(tmp_path: Path):
    history, _, config, protocol_path, report_path, report = _registered_files(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = null_module._formal_checkpoint_manifest(
        protocol_path,
        report_path,
        history,
        lottery="fc3d",
        config=config,
        frozen_periods_excluded=500,
        iterations=5000,
    )

    assert manifest["referenceReportSha256"] == report["reportSha256"]
    assert manifest["historyPeriods"] == protocol["developmentData"]["periods"]
    assert (
        manifest["frozenPeriodsExcluded"]
        == protocol["frozenBoundary"]["periodsExcluded"]
    )

    with pytest.raises(ValueError, match="Frozen"):
        null_module._formal_checkpoint_manifest(
            protocol_path,
            report_path,
            history,
            lottery="fc3d",
            config=config,
            frozen_periods_excluded=501,
            iterations=5000,
        )
    with pytest.raises(ValueError, match="历史期数"):
        extra = pd.DataFrame(
            {"期数": ["2026051"], "百位": [1], "十位": [2], "个位": [3]}
        )
        null_module._formal_checkpoint_manifest(
            protocol_path,
            report_path,
            pd.concat([history, extra], ignore_index=True),
            lottery="fc3d",
            config=config,
            frozen_periods_excluded=500,
            iterations=5000,
        )


def test_completed_trial_is_checkpointed_before_slow_future_finishes(
    monkeypatch, tmp_path: Path
):
    fast = Future()
    slow = Future()
    fast.set_result(
        null_module.ProbabilityV5NullTrial(
            0, 1, 1.0, 0.0, 0.0, 0, False, (("uniform", 1.0),)
        )
    )
    writes: list[int] = []

    class Executor:
        def __init__(self, max_workers: int):
            self.futures = iter((fast, slow))

        def submit(self, function, task):
            return next(self.futures)

        def shutdown(self, wait: bool, cancel_futures: bool):
            assert writes == [0, 1]

    def controlled_as_completed(futures):
        yield fast
        assert writes == [0]
        slow.set_result(
            null_module.ProbabilityV5NullTrial(
                1, 2, 1.0, 0.0, 0.0, 0, False, (("uniform", 1.0),)
            )
        )
        yield slow

    monkeypatch.setattr(null_module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(null_module, "as_completed", controlled_as_completed)
    monkeypatch.setattr(
        null_module,
        "_write_checkpoint_trial",
        lambda _p, _s, trial: writes.append(trial.index),
    )
    monkeypatch.setattr(null_module, "_load_checkpoint_trials", lambda *a, **k: {})
    monkeypatch.setattr(null_module, "_write_immutable_json", lambda *a, **k: None)
    reference = run_probability_v5_development(
        _history(),
        get_lottery_rule("fc3d"),
        probability_v5_smoke_config(),
        frozen_periods_excluded=500,
        include_period_details=False,
    ).to_dict()

    null_module.run_probability_v5_null_simulation(
        reference,
        lottery="fc3d",
        config=probability_v5_smoke_config(),
        history_periods=50,
        frozen_periods_excluded=500,
        iterations=2,
        workers=2,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert writes == [0, 1]


def test_post_final_weights_equal_last_period_after_update():
    report = run_probability_v5_development(
        _history(),
        get_lottery_rule("fc3d"),
        probability_v5_smoke_config(),
        frozen_periods_excluded=500,
    ).to_dict()

    assert (
        report["evaluation"]["postFinalUpdateExpertWeights"]
        == report["periods"][-1]["expertWeightsAfter"]
    )
