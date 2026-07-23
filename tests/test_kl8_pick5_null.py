# -*- coding: utf-8 -*-
"""快乐8选5随机零假设契约测试。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import kl8_pick5_null
from src.analysis import kl8_pick5_null as null_module
from src.analysis.kl8_pick5_null import (
    Kl8NullReport,
    Kl8NullTrial,
    build_formal_manifest,
    run_formal_kl8_null,
    run_kl8_null_smoke,
    write_kl8_null_report,
)
from src.analysis.kl8_pick5_probability_v1 import (
    EXPERT_NAMES,
    Kl8Pick5Config,
    payload_sha256,
    run_kl8_development,
)
from tests.test_kl8_pick5_probability_v1 import _frozen_boundary, _history


def _reference():
    config = Kl8Pick5Config.smoke()
    report = run_kl8_development(_history(config.required_periods), config)
    return report.to_dict(), config


def _rehash_trial(payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "trialSha256"}
    serialized = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )
    payload["trialSha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _rewrite_trial(path: Path, payload: dict[str, object]) -> None:
    payload["trialSha256"] = payload_sha256(
        {key: value for key, value in payload.items() if key != "trialSha256"}
    )
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o444)


def _gate_inputs(*, passed: bool) -> dict[str, object]:
    return {
        "deltaLogLossVsUniform": 0.01,
        "deltaBrierVsUniform": 0.01,
        "expectedPositiveDeviation": 0.0,
        "exactPortfolioTotalHitsPValue": 0.01 if passed else 1.0,
        "meanHitsPerTicket": 1.3,
        "meanPortfolioTotalHits": 6.5,
        "blockBootstrap": {
            "deltaLogLoss": {"pValueMeanNonPositive": 0.01},
            "deltaBrier": {"pValueMeanNonPositive": 0.01},
        },
        "blockStability": [
            {
                "deltaLogLoss": 0.01,
                "deltaBrier": 0.01,
                "meanHitsPerTicket": 1.3,
            }
            for _ in range(5)
        ],
    }


def test_null_replays_joint_gate_and_is_smoke_only(tmp_path: Path):
    reference, config = _reference()
    report = run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=2,
        workers=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    payload = report.to_dict()
    assert payload["evidenceStatus"] == "smoke_only"
    assert payload["promotionPassed"] is False
    assert all(
        "searchPassed" in trial and "evaluationPassed" in trial
        for trial in payload["trials"]
    )
    assert payload["orderedTrialSetSha256"]
    assert all(trial["trialSha256"] for trial in payload["trials"])
    assert set(payload["summary"]["observedStatisticPValues"]) == {
        "evaluationDeltaLogLoss",
        "evaluationDeltaBrier",
        "evaluationMeanHitsPerTicket",
        "evaluationMeanPortfolioTotalHits",
        "evaluationPortfolioBestHitAtLeast3Rate",
        "evaluationPortfolioBestHitAtLeast4Rate",
        "evaluationPortfolioBestHitExactly5Rate",
    }
    assert all(
        "evaluationPortfolioBestHitAtLeast4Rate" in trial
        and "evaluationPortfolioBestHitExactly5Rate" in trial
        for trial in payload["trials"]
    )
    assert len(payload["summary"]["finalExpertWeightDistributions"]) == 6
    assert all(
        set(distribution) == {"mean", "std", "min", "q25", "median", "q75", "max"}
        for distribution in payload["summary"][
            "finalExpertWeightDistributions"
        ].values()
    )
    output = tmp_path / "null.json"
    write_kl8_null_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8"))["reportSha256"]


def test_formal_iterations_boundary_checked_before_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    reference, _ = _reference()
    config = Kl8Pick5Config()
    with pytest.raises(ValueError, match="至少5000"):
        build_formal_manifest(
            reference,
            config=config,
            history_periods=config.required_periods,
            iterations=4999,
        )
    manifest = build_formal_manifest(
        reference,
        config=config,
        history_periods=config.required_periods,
        iterations=5000,
    )
    assert manifest["iterations"] == 5000
    monkeypatch.setattr(
        null_module,
        "_run_null_trial",
        lambda task: (_ for _ in ()).throw(AssertionError("不得执行")),
    )


def test_formal_cli_rejects_4999_before_loading_history(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        kl8_pick5_null,
        "load_kl8_development_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不得加载")),
    )
    with pytest.raises(SystemExit) as error:
        kl8_pick5_null.main(
            [
                "--csv",
                "missing.csv",
                "--output",
                "missing.json",
                "--protocol",
                "protocol.json",
                "--reference-report",
                "report.json",
                "--checkpoint-dir",
                "checkpoint",
                "--iterations",
                "4999",
            ]
        )
    assert error.value.code == 2


def test_formal_public_boundary_validates_before_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = Kl8Pick5Config()
    history = _history(0)
    boundary = _frozen_boundary()
    protocol_path = tmp_path / "protocol.json"
    report_path = tmp_path / "report.json"
    sentinel = object()
    captured = {}

    def fake_run_simulation(reference, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(null_module, "_run_simulation", fake_run_simulation)
    monkeypatch.setattr(
        null_module, "load_and_verify_kl8_report", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        null_module,
        "_load_readonly_json",
        lambda *args, **kwargs: {
            "frozenBoundary": {
                "periodsExcluded": 500,
                "firstIssue": boundary["firstIssue"],
                "lastIssue": boundary["lastIssue"],
                "numbersRead": False,
            },
            "developmentData": {"periods": 0},
        },
    )
    with pytest.raises(ValueError, match="至少5000"):
        run_formal_kl8_null(
            protocol_path,
            report_path,
            history,
            config=config,
            frozen_periods_excluded=500,
            frozen_boundary=boundary,
            iterations=4999,
            workers=1,
            checkpoint_dir=tmp_path / "checkpoint",
        )
    result = run_formal_kl8_null(
        protocol_path,
        report_path,
        history,
        config=config,
        frozen_periods_excluded=500,
        frozen_boundary=boundary,
        iterations=5000,
        workers=1,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    assert result is sentinel
    assert captured["formal"] is True
    assert captured["iterations"] == 5000


def test_formal_null_pass_requires_all_preregistered_pvalues():
    config = Kl8Pick5Config()
    failing_gate_inputs = _gate_inputs(passed=False)
    trials = tuple(
        Kl8NullTrial(
            index=index,
            seed=index,
            search_passed=False,
            evaluation_passed=False,
            joint_passed=False,
            search_delta_logloss=0.0,
            search_delta_brier=0.0,
            search_mean_hits_per_ticket=1.0,
            search_mean_portfolio_total_hits=5.0,
            evaluation_delta_logloss=1.0 if index < 300 else 0.0,
            evaluation_delta_brier=0.0,
            evaluation_mean_hits_per_ticket=1.0,
            evaluation_mean_portfolio_total_hits=5.0,
            evaluation_portfolio_best_hit_at_least3_rate=0.0,
            evaluation_portfolio_best_hit_at_least4_rate=0.0,
            evaluation_portfolio_best_hit_exactly5_rate=0.0,
            final_expert_weights={name: 1 / 6 for name in EXPERT_NAMES},
            search_gate_inputs=failing_gate_inputs,
            evaluation_gate_inputs=failing_gate_inputs,
        )
        for index in range(5000)
    )
    report = Kl8NullReport(
        reference_report_sha256="reference",
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=5000,
        workers=1,
        formal=True,
        trials=trials,
        new_completion_order=tuple(range(5000)),
        checkpoint_used=True,
        observed_statistics={
            "evaluationDeltaLogLoss": 1.0,
            "evaluationDeltaBrier": 1.0,
            "evaluationMeanHitsPerTicket": 2.0,
            "evaluationMeanPortfolioTotalHits": 10.0,
            "evaluationPortfolioBestHitAtLeast3Rate": 1.0,
            "evaluationPortfolioBestHitAtLeast4Rate": 1.0,
            "evaluationPortfolioBestHitExactly5Rate": 1.0,
        },
    ).to_dict()
    assert report["summary"]["empiricalFalsePositiveRate"] == 0.0
    assert (
        report["summary"]["observedStatisticPValues"]["evaluationDeltaLogLoss"]
        > config.alpha
    )
    assert report["summary"]["nullSimulationPassed"] is False


def test_checkpoint_completion_order_resume_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    completion_order: list[int] = []
    original = null_module._run_null_trial

    def delayed(task):
        if task.index == 0:
            time.sleep(0.1)
        return original(task)

    monkeypatch.setattr(null_module, "_run_null_trial", delayed)
    monkeypatch.setattr(null_module, "ProcessPoolExecutor", ThreadPoolExecutor)
    report = run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=2,
        workers=2,
        checkpoint_dir=checkpoint,
        progress_callback=lambda index, total: completion_order.append(index),
    )
    assert report.to_dict()["execution"]["checkpointUsed"] is True
    assert completion_order == [1, 2]
    assert report.to_dict()["execution"]["newCompletionOrder"] == [1, 0]
    assert "completionOrder" not in report.to_dict()["execution"]

    trial_path = checkpoint / "trials" / "trial_000001.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    payload["seed"] += 1
    trial_path.chmod(0o644)
    trial_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="检查点"):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=2,
            workers=1,
            checkpoint_dir=checkpoint,
        )

    valid = original(
        null_module._NullTask(
            1, null_module.NULL_SEED_BASE + 1, config, config.required_periods, 500
        )
    ).to_dict()
    valid["jointPassed"] = not (valid["searchPassed"] and valid["evaluationPassed"])
    unsigned = {key: value for key, value in valid.items() if key != "trialSha256"}
    valid["trialSha256"] = payload_sha256(unsigned)
    trial_path.chmod(0o644)
    trial_path.write_text(json.dumps(valid), encoding="utf-8")
    trial_path.chmod(0o444)
    with pytest.raises(ValueError, match="派生"):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=2,
            workers=1,
            checkpoint_dir=checkpoint,
        )


def test_checkpoint_rejects_rehashed_gate_boolean_forgery(tmp_path: Path):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=1,
        workers=1,
        checkpoint_dir=checkpoint,
    )
    trial_path = checkpoint / "trials" / "trial_000000.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    assert payload["searchPassed"] is False
    assert payload["evaluationPassed"] is False
    assert payload["jointPassed"] is False
    payload["searchPassed"] = True
    payload["evaluationPassed"] = True
    payload["jointPassed"] = True
    _rewrite_trial(trial_path, payload)

    with pytest.raises(ValueError, match="门控派生"):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=1,
            workers=1,
            checkpoint_dir=checkpoint,
        )


def test_checkpoint_rejects_rehashed_failing_gate_inputs_claiming_true(tmp_path: Path):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=1,
        workers=1,
        checkpoint_dir=checkpoint,
    )
    trial_path = checkpoint / "trials" / "trial_000000.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    payload["searchGateInputs"]["exactPortfolioTotalHitsPValue"] = 1.0
    payload["searchPassed"] = True
    payload["evaluationPassed"] = True
    payload["jointPassed"] = True
    _rewrite_trial(trial_path, payload)

    with pytest.raises(ValueError, match="门控派生"):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=1,
            workers=1,
            checkpoint_dir=checkpoint,
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({name: 1 / 6 for name in EXPERT_NAMES[:-1]}, "专家权重名称"),
        ({name: 0.2 for name in EXPERT_NAMES}, "专家权重"),
        (
            {
                name: (-0.1 if index == 0 else 1.1 if index == 1 else 0.0)
                for index, name in enumerate(EXPERT_NAMES)
            },
            "专家权重",
        ),
    ],
)
def test_checkpoint_rejects_invalid_expert_weights(tmp_path: Path, weights, message):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=1,
        workers=1,
        checkpoint_dir=checkpoint,
    )
    trial_path = checkpoint / "trials" / "trial_000000.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    payload["finalExpertWeights"] = weights
    unsigned = {key: value for key, value in payload.items() if key != "trialSha256"}
    payload["trialSha256"] = payload_sha256(unsigned)
    trial_path.chmod(0o644)
    trial_path.write_text(json.dumps(payload), encoding="utf-8")
    trial_path.chmod(0o444)
    with pytest.raises(ValueError, match=message):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=1,
            workers=1,
            checkpoint_dir=checkpoint,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("index", True, "index必须为JSON整数"),
        ("seed", str(null_module.NULL_SEED_BASE), "seed必须为JSON整数"),
        ("evaluationDeltaLogLoss", "0.0", "统计量必须为JSON数值"),
        (
            "finalExpertWeights",
            {name: str(1 / 6) for name in EXPERT_NAMES},
            "专家权重必须为JSON数值",
        ),
    ],
)
def test_checkpoint_rejects_noncanonical_json_numeric_types(
    tmp_path: Path, field: str, replacement: object, message: str
):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=1,
        workers=1,
        checkpoint_dir=checkpoint,
    )
    trial_path = checkpoint / "trials" / "trial_000000.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    payload[field] = replacement
    _rewrite_trial(trial_path, payload)

    with pytest.raises(ValueError, match=message):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=1,
            workers=1,
            checkpoint_dir=checkpoint,
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda payload: payload["searchGateInputs"].update({"extra": 0}),
            "searchGateInputs字段集合",
        ),
        (
            lambda payload: payload["searchGateInputs"]["blockBootstrap"][
                "deltaLogLoss"
            ].update({"extra": 0}),
            "blockBootstrap.deltaLogLoss字段集合",
        ),
        (
            lambda payload: payload["searchGateInputs"]["blockStability"][0].update(
                {"extra": 0}
            ),
            r"blockStability\[0\]字段集合",
        ),
        (
            lambda payload: payload["searchGateInputs"].update(
                {"expectedPositiveDeviation": "0.0"}
            ),
            "expectedPositiveDeviation必须为JSON数值",
        ),
    ],
)
def test_checkpoint_rejects_noncanonical_nested_gate_inputs(
    tmp_path: Path, tamper, message: str
):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=1,
        workers=1,
        checkpoint_dir=checkpoint,
    )
    trial_path = checkpoint / "trials" / "trial_000000.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    tamper(payload)
    _rewrite_trial(trial_path, payload)

    with pytest.raises(ValueError, match=message):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=1,
            workers=1,
            checkpoint_dir=checkpoint,
        )


def test_process_failure_has_no_thread_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    reference, config = _reference()

    class FailedExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def submit(self, function, task):
            future: Future[object] = Future()
            future.set_exception(PermissionError("worker unavailable"))
            return future

        def shutdown(self, wait: bool, cancel_futures: bool):
            assert cancel_futures is True

    monkeypatch.setattr(null_module, "ProcessPoolExecutor", FailedExecutor)
    with pytest.raises(RuntimeError, match="失败关闭"):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=2,
            workers=2,
            checkpoint_dir=tmp_path / "failed",
        )


def test_null_trial_requires_search_and_evaluation_jointly(
    monkeypatch: pytest.MonkeyPatch,
):
    reference, config = _reference()

    class FakeReport:
        failing_gate_inputs = _gate_inputs(passed=False)
        passing_gate_inputs = _gate_inputs(passed=True)
        search = {
            "metrics": {
                **failing_gate_inputs,
            },
            "gate": {"passed": False},
        }
        evaluation = {
            "metrics": {
                **passing_gate_inputs,
                "portfolioBestHitAtLeast3Rate": 0.5,
                "portfolioBestHitAtLeast4Rate": 0.25,
                "portfolioBestHitExactly5Rate": 0.1,
            },
            "gate": {"passed": True},
        }
        final_expert_weights = {name: 1 / 6 for name in null_module.EXPERT_NAMES}

    monkeypatch.setattr(
        null_module, "run_kl8_development", lambda *args, **kwargs: FakeReport()
    )
    task = null_module._NullTask(0, 123, config, config.required_periods, 500)
    trial = null_module._run_null_trial(task)
    assert trial.search_passed is False
    assert trial.evaluation_passed is True
    assert trial.joint_passed is False
    assert trial.evaluation_portfolio_best_hit_at_least4_rate == pytest.approx(0.25)
    assert trial.evaluation_portfolio_best_hit_exactly5_rate == pytest.approx(0.1)
    assert reference["developmentSignalsPassed"] in {True, False}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluationPortfolioBestHitAtLeast4Rate", 1.1, "命中率"),
        ("evaluationMeanHitsPerTicket", 5.1, "每票均值"),
        ("evaluationMeanPortfolioTotalHits", 25.1, "组合总命中均值"),
        ("evaluationDeltaLogLoss", math.nan, "身份或哈希"),
        ("evaluationDeltaBrier", math.inf, "身份或哈希"),
    ],
)
def test_checkpoint_rejects_invalid_numeric_statistics(
    tmp_path: Path, field: str, value: float, message: str
):
    reference, config = _reference()
    checkpoint = tmp_path / "checkpoint"
    run_kl8_null_smoke(
        reference,
        config=config,
        history_periods=config.required_periods,
        frozen_periods_excluded=500,
        iterations=1,
        workers=1,
        checkpoint_dir=checkpoint,
    )
    trial_path = checkpoint / "trials" / "trial_000000.json"
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    payload[field] = value
    if math.isfinite(value):
        unsigned = {key: item for key, item in payload.items() if key != "trialSha256"}
        payload["trialSha256"] = payload_sha256(unsigned)
    else:
        _rehash_trial(payload)
    trial_path.chmod(0o644)
    trial_path.write_text(json.dumps(payload), encoding="utf-8")
    trial_path.chmod(0o444)
    with pytest.raises(ValueError, match=message):
        run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=config.required_periods,
            frozen_periods_excluded=500,
            iterations=1,
            workers=1,
            checkpoint_dir=checkpoint,
        )
