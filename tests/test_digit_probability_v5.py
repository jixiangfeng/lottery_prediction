# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.digit_probability_v5 import main as probability_v5_main
from src.analysis import digit_probability_v5 as probability_v5
from src.analysis.digit_probability_v5 import (
    ProbabilityV5DevelopmentConfig,
    build_probability_v5_protocol,
    load_and_verify_probability_v5_protocol,
    probability_v5_smoke_config,
    run_probability_v5_development,
    write_probability_v5_protocol,
    write_probability_v5_report,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    state = 20260722
    rows = []
    for index in range(periods):
        digits = []
        for _ in range(3):
            state = (1664525 * state + 1013904223) % (2**32)
            digits.append(state % 10)
        rows.append(
            {
                "期数": str(2026001 + index),
                "百位": digits[0],
                "十位": digits[1],
                "个位": digits[2],
            }
        )
    return pd.DataFrame(rows)


def _config() -> ProbabilityV5DevelopmentConfig:
    return probability_v5_smoke_config()


def test_expert_and_mixture_probabilities_are_complete_and_normalized():
    ewma = probability_v5._EWMAProbabilityState.create(80, 300)
    for row in _history(20)[["百位", "十位", "个位"]].to_numpy():
        ewma.update(row)

    position = ewma.position_probabilities()
    pairwise = ewma.pairwise_probabilities()
    records = probability_v5._run_prequential(
        _history(50), get_lottery_rule("fc3d"), _config()
    )

    for probabilities in (position, pairwise, records[0].probabilities):
        assert probabilities.shape == (1000,)
        assert np.all(probabilities >= 0)
        assert np.isclose(probabilities.sum(), 1.0)
    assert np.allclose(records[0].expert_weights, (0.5, 0.2, 0.15, 0.15))
    assert all(value > 0 for value in records[0].expert_actual_probabilities)
    serialized = _config().to_dict()
    assert serialized["randomSeed"] == 20260722
    assert serialized["legacyFeatureConfig"]["windowWeights"] == {
        "20": 2.5,
        "50": 1.5,
        "150": 0.75,
    }
    assert serialized["legacyGradientConfig"]["l2Penalty"] == 0.001


def test_prequential_prediction_is_prior_only_and_calibration_ignores_evaluation():
    original = _history(50)
    changed = original.copy()
    changed.loc[40, ["百位", "十位", "个位"]] = [
        (int(original.loc[40, column]) + 1) % 10 for column in ("百位", "十位", "个位")
    ]
    rule = get_lottery_rule("fc3d")

    first = run_probability_v5_development(
        original, rule, _config(), frozen_periods_excluded=500
    )
    second = run_probability_v5_development(
        changed, rule, _config(), frozen_periods_excluded=500
    )

    first_payload = first.to_dict()
    second_payload = second.to_dict()
    assert (
        first_payload["periods"][0]["distributionFingerprint"]
        == second_payload["periods"][0]["distributionFingerprint"]
    )
    assert (
        first_payload["periods"][1]["distributionFingerprint"]
        != second_payload["periods"][1]["distributionFingerprint"]
    )
    assert first.selected_temperature == second.selected_temperature
    assert first.calibration == second.calibration


def test_raw_and_daily_policy_top50_are_both_auditable(monkeypatch):
    history = _history(50)
    history.loc[19, ["百位", "十位", "个位"]] = [0, 0, 1]
    monkeypatch.setattr(
        probability_v5,
        "rank_candidate_indices",
        lambda probabilities, candidates: np.arange(1000),
    )

    record = probability_v5._run_prequential(
        history, get_lottery_rule("fc3d"), _config()
    )[0]

    assert record.raw_top_k == tuple(f"{value:03d}" for value in range(50))
    assert len(record.policy_top_k) == 50
    assert "001" not in record.policy_top_k
    assert sum(len(set(value)) == 1 for value in record.policy_top_k) == 1
    assert record.raw_top_k != record.policy_top_k


def test_development_report_can_never_enable_ranking_or_validation():
    report = run_probability_v5_development(
        _history(50),
        get_lottery_rule("pl3"),
        _config(),
        frozen_periods_excluded=500,
    ).to_dict()

    assert report["evidenceStatus"] == "exploratory_reused_development"
    assert report["protocol"]["frozenRead"] is False
    assert report["protocol"]["frozenPeriodsExcluded"] == 500
    assert report["protocol"]["validationOpened"] is False
    assert report["protocol"]["legacyStateCompatibility"] is False
    assert report["search"]["policyTopKViability"]["periods"] == 10
    assert len(report["search"]["periodDetails"][0]["rawTopK"]) == 50
    assert len(report["search"]["periodDetails"][0]["policyTopK"]) == 50
    assert report["promotionPassed"] is False
    assert report["researchRankingEnabled"] is False
    assert report["recommendationEnabled"] is False
    assert report["researchTop50"] == []
    assert report["formalRecommendation"] is None


def test_development_report_rejects_an_unprotected_frozen_boundary():
    with pytest.raises(ValueError, match="至少500期Frozen"):
        run_probability_v5_development(
            _history(50),
            get_lottery_rule("fc3d"),
            _config(),
            frozen_periods_excluded=499,
        )


def test_protocol_is_deterministic_immutable_and_bound_to_current_inputs(
    tmp_path: Path,
):
    history = _history(60)
    rule = get_lottery_rule("fc3d")
    config = _config()
    protocol = build_probability_v5_protocol(
        history, rule, config, frozen_periods_excluded=500
    )
    path = tmp_path / "protocol.json"

    assert write_probability_v5_protocol(protocol, path) == path
    assert write_probability_v5_protocol(protocol, path) == path
    assert (
        load_and_verify_probability_v5_protocol(
            path,
            history,
            rule,
            config,
            frozen_periods_excluded=500,
        )
        == protocol
    )
    assert protocol["developmentData"]["periods"] == 50
    assert protocol["developmentData"]["firstIssue"] == str(2026011)

    older_change = history.copy()
    older_change.loc[0, "百位"] = (int(older_change.loc[0, "百位"]) + 1) % 10
    assert (
        load_and_verify_probability_v5_protocol(
            path,
            older_change,
            rule,
            config,
            frozen_periods_excluded=500,
        )
        == protocol
    )

    changed = history.copy()
    changed.loc[10, "百位"] = (int(changed.loc[10, "百位"]) + 1) % 10
    with pytest.raises(RuntimeError, match="不一致"):
        load_and_verify_probability_v5_protocol(
            path,
            changed,
            rule,
            config,
            frozen_periods_excluded=500,
        )


def test_development_report_is_immutable_and_records_protocol(tmp_path: Path):
    history = _history(50)
    report = run_probability_v5_development(
        history,
        get_lottery_rule("fc3d"),
        _config(),
        frozen_periods_excluded=500,
        development_protocol_sha256="protocol-test",
    )
    path = tmp_path / "development.json"

    assert write_probability_v5_report(report, path) == path
    assert write_probability_v5_report(report, path) == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol"]["developmentProtocolRegistered"] is True
    assert payload["protocol"]["developmentProtocolSha256"] == "protocol-test"

    changed_history = history.copy()
    changed_history.loc[40, "个位"] = (int(changed_history.loc[40, "个位"]) + 1) % 10
    changed_report = run_probability_v5_development(
        changed_history,
        get_lottery_rule("fc3d"),
        _config(),
        frozen_periods_excluded=500,
        development_protocol_sha256="protocol-test",
    )
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        write_probability_v5_report(changed_report, path)


def test_cli_smoke_excludes_frozen_and_only_writes_development_report(
    tmp_path: Path, capsys
):
    csv_path = tmp_path / "history.csv"
    output = tmp_path / "probability_v5.json"
    _history(550).to_csv(csv_path, index=False, encoding="utf-8")

    exit_code = probability_v5_main(
        [
            "--lottery",
            "fc3d",
            "--csv",
            str(csv_path),
            "--output",
            str(output),
            "--frozen-test-periods",
            "500",
            "--smoke",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "frozenExcluded=500" in stdout
    assert "frozenRead=false" in stdout
    assert payload["evaluationKind"] == "development_prequential_challenger"
    assert payload["protocol"]["frozenPeriodsExcluded"] == 500
    assert payload["fullPipelineNullSimulation"]["status"] == "not_run"
    assert payload["promotionPassed"] is False
    assert {path.name for path in tmp_path.iterdir()} == {"history.csv", output.name}


def test_cli_registers_protocol_and_full_run_requires_it(tmp_path: Path, capsys):
    csv_path = tmp_path / "history.csv"
    protocol_path = tmp_path / "protocol.json"
    _history(2000).to_csv(csv_path, index=False, encoding="utf-8")
    register_args = [
        "--lottery",
        "fc3d",
        "--csv",
        str(csv_path),
        "--protocol",
        str(protocol_path),
        "--frozen-test-periods",
        "500",
        "--register-protocol",
    ]

    assert probability_v5_main(register_args) == 0
    assert probability_v5_main(register_args) == 0
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["immutable"] is True
    assert protocol["developmentData"]["periods"] == 1400
    assert protocol["developmentData"]["firstIssue"] == str(2026101)
    assert protocol["frozenBoundary"]["frozenRead"] is False
    assert "protocolSha256=" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        probability_v5_main(
            [
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--output",
                str(tmp_path / "development.json"),
            ]
        )
