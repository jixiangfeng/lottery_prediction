# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from scripts.digit_probability_v5_null import main as probability_v5_null_main
from src.analysis import digit_probability_v5_null as null_module
from src.analysis.digit_probability_v5 import (
    probability_v5_smoke_config,
    run_probability_v5_development,
)
from src.analysis.digit_probability_v5_null import (
    run_probability_v5_null_simulation,
    write_probability_v5_null_report,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    values = [((index * 137) + 29) % 1000 for index in range(periods)]
    return pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(periods)],
            "百位": [value // 100 for value in values],
            "十位": [(value // 10) % 10 for value in values],
            "个位": [value % 10 for value in values],
        }
    )


def _reference(protocol_sha256: str | None = None) -> tuple[dict[str, object], object]:
    config = probability_v5_smoke_config()
    report = run_probability_v5_development(
        _history(50),
        get_lottery_rule("fc3d"),
        config,
        frozen_periods_excluded=500,
        development_protocol_sha256=protocol_sha256,
        include_period_details=False,
    )
    return report.to_dict(), config


def test_serial_and_parallel_full_pipeline_null_trials_are_identical(tmp_path: Path):
    reference, config = _reference("protocol-test")
    common = {
        "lottery": "fc3d",
        "config": config,
        "history_periods": 50,
        "frozen_periods_excluded": 500,
        "iterations": 2,
        "formal": False,
        "protocol_sha256": None,
    }

    serial = run_probability_v5_null_simulation(reference, workers=1, **common)
    parallel = run_probability_v5_null_simulation(reference, workers=2, **common)

    assert serial.trials == parallel.trials
    assert serial.summary == parallel.summary
    assert serial.summary["nullSimulationPassed"] is False
    assert serial.to_dict()["execution"]["fullPipelineReplayed"] is True
    assert serial.to_dict()["promotionPassed"] is False

    path = tmp_path / "null.json"
    assert write_probability_v5_null_report(serial, path) == path
    assert write_probability_v5_null_report(serial, path) == path
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        write_probability_v5_null_report(replace(serial, workers=99), path)


def test_formal_null_simulation_rejects_incomplete_or_unregistered_execution():
    reference, config = _reference("protocol-test")

    with pytest.raises(ValueError, match="必须执行5000次"):
        run_probability_v5_null_simulation(
            reference,
            lottery="fc3d",
            config=config,
            history_periods=50,
            frozen_periods_excluded=500,
            iterations=1,
            formal=True,
            protocol_sha256="protocol-test",
        )

    unregistered, _ = _reference()
    with pytest.raises(ValueError, match="未登记开发协议"):
        run_probability_v5_null_simulation(
            unregistered,
            lottery="fc3d",
            config=config,
            history_periods=50,
            frozen_periods_excluded=500,
            iterations=5000,
            formal=True,
            protocol_sha256="protocol-test",
            checkpoint_dir="unused-checkpoint",
        )


def test_checkpoint_resumes_without_recomputing_completed_trials(
    tmp_path: Path, monkeypatch
):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    common = {
        "lottery": "fc3d",
        "config": config,
        "history_periods": 50,
        "frozen_periods_excluded": 500,
        "iterations": 2,
        "workers": 1,
        "formal": False,
        "checkpoint_dir": checkpoint,
    }
    first = run_probability_v5_null_simulation(reference, **common)
    assert (checkpoint / "manifest.json").exists()
    assert len(list((checkpoint / "trials").glob("trial_*.json"))) == 2

    def fail_if_recomputed(task):
        raise AssertionError(f"不应重新计算试验{task.index}")

    monkeypatch.setattr(null_module, "_run_null_trial", fail_if_recomputed)
    resumed = run_probability_v5_null_simulation(reference, **common)
    assert resumed.trials == first.trials
    assert resumed.summary == first.summary
    assert resumed.to_dict()["execution"]["checkpointUsed"] is True

    changed_reference = dict(reference)
    changed_reference["checkpointIdentityProbe"] = True
    with pytest.raises(FileExistsError, match="检查点.*禁止覆盖"):
        run_probability_v5_null_simulation(changed_reference, **common)


def test_null_cli_smoke_excludes_frozen_and_cannot_promote(tmp_path: Path, capsys):
    csv_path = tmp_path / "history.csv"
    output = tmp_path / "null_smoke.json"
    _history(550).to_csv(csv_path, index=False, encoding="utf-8")

    exit_code = probability_v5_null_main(
        [
            "--lottery",
            "fc3d",
            "--csv",
            str(csv_path),
            "--output",
            str(output),
            "--frozen-test-periods",
            "500",
            "--iterations",
            "1",
            "--smoke",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "frozenExcluded=500" in stdout
    assert payload["evidenceStatus"] == "smoke_only"
    assert payload["execution"]["iterations"] == 1
    assert payload["summary"]["nullSimulationPassed"] is False
    assert payload["promotionPassed"] is False
    assert payload["recommendationEnabled"] is False
