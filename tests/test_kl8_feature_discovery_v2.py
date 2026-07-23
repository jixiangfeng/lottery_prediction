# -*- coding: utf-8 -*-
"""快乐8 v2 探索性特征发现测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import kl8_feature_discovery_v2 as cli_module
from src.analysis.kl8_feature_discovery_v2 import (
    ALL_FEATURE_COLUMNS,
    CANDIDATE_FEATURE_SETS,
    Kl8FeatureDiscoveryConfig,
    build_prior_only_number_panel,
    run_kl8_feature_discovery_v2,
    write_kl8_feature_discovery_report,
)


def _history(periods: int) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2020-01-01")
    for period in range(periods):
        numbers = sorted({((period * 7 + offset * 3) % 80) + 1 for offset in range(20)})
        rows.append(
            {
                "issue": str(2020001 + period),
                "date": (start + pd.Timedelta(days=period)).date().isoformat(),
                "numbers": numbers,
            }
        )
    return pd.DataFrame(rows)


def _smoke_config(*, periods: int = 40) -> Kl8FeatureDiscoveryConfig:
    assert periods == 40
    return Kl8FeatureDiscoveryConfig(
        initial_train=20,
        search_periods=10,
        evaluation_periods=10,
        refit_interval=50,
        n_estimators=8,
        num_leaves=5,
        max_depth=3,
        min_child_samples=10,
        n_jobs=1,
    )


def _without_elapsed(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _without_elapsed(value)
            for key, value in payload.items()
            if key != "elapsedSeconds"
        }
    if isinstance(payload, list):
        return [_without_elapsed(value) for value in payload]
    return payload


def test_prior_only_features_ignore_current_and_future_mutations() -> None:
    history = _history(30)
    original = build_prior_only_number_panel(history)
    mutated = history.copy(deep=True)
    mutated.loc[12:, "numbers"] = mutated.loc[12:, "numbers"].map(
        lambda numbers: sorted({((number + 37) % 80) + 1 for number in numbers})
    )
    changed = build_prior_only_number_panel(mutated)

    original_target = original.loc[original["periodIndex"] == 12, ALL_FEATURE_COLUMNS]
    changed_target = changed.loc[changed["periodIndex"] == 12, ALL_FEATURE_COLUMNS]
    np.testing.assert_array_equal(original_target.to_numpy(), changed_target.to_numpy())


def test_feature_panel_has_expected_shape_and_finite_float64_values() -> None:
    panel = build_prior_only_number_panel(_history(24))

    assert panel.shape[0] == 24 * 80
    assert panel.groupby("periodIndex", sort=True).size().eq(80).all()
    assert panel["number"].min() == 1
    assert panel["number"].max() == 80
    assert set(panel["target"].unique()) <= {0.0, 1.0}
    assert all(
        panel[column].dtype == np.dtype("float64") for column in ALL_FEATURE_COLUMNS
    )
    assert np.isfinite(panel[list(ALL_FEATURE_COLUMNS)].to_numpy()).all()


def test_tiny_smoke_is_deterministic() -> None:
    history = _history(40)
    config = _smoke_config()
    candidates = {
        "frequency": CANDIDATE_FEATURE_SETS["frequency"],
        "frequency+omission": CANDIDATE_FEATURE_SETS["frequency+omission"],
    }

    first = run_kl8_feature_discovery_v2(
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
        candidate_feature_sets=candidates,
    )
    second = run_kl8_feature_discovery_v2(
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
        candidate_feature_sets=candidates,
    )

    assert _without_elapsed(first) == _without_elapsed(second)


def test_feature_selection_uses_search_only() -> None:
    config = _smoke_config()
    history = _history(40)
    candidates = {
        "frequency": CANDIDATE_FEATURE_SETS["frequency"],
        "frequency+omission": CANDIDATE_FEATURE_SETS["frequency+omission"],
    }
    first = run_kl8_feature_discovery_v2(
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
        candidate_feature_sets=candidates,
    )
    mutated = history.copy(deep=True)
    mutated.loc[30:, "numbers"] = mutated.loc[30:, "numbers"].map(
        lambda numbers: sorted({((number + 41) % 80) + 1 for number in numbers})
    )
    second = run_kl8_feature_discovery_v2(
        mutated,
        config,
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
        candidate_feature_sets=candidates,
    )

    assert first["selectedFeatureSet"] == second["selectedFeatureSet"]
    assert _without_elapsed(first["searchCandidates"]) == _without_elapsed(
        second["searchCandidates"]
    )


def test_cli_loader_never_parses_poisoned_frozen_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "kl8.csv"
    history = _history(41)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue", "date", "numbers"])
        writer.writeheader()
        for row in history.iloc[:40].itertuples(index=False):
            writer.writerow(
                {
                    "issue": row.issue,
                    "date": row.date,
                    "numbers": " ".join(str(number) for number in row.numbers),
                }
            )
        frozen = history.iloc[40]
        writer.writerow(
            {"issue": frozen.issue, "date": frozen.date, "numbers": "POISON_FROZEN"}
        )

    captured: dict[str, object] = {}

    def fake_run(development: pd.DataFrame, config: object, **kwargs: object) -> dict:
        captured["periods"] = len(development)
        captured["numbers"] = development.iloc[-1]["numbers"]
        return {"schemaVersion": "test", "selectedFeatureSet": "uniform"}

    monkeypatch.setattr(cli_module, "run_kl8_feature_discovery_v2", fake_run)
    monkeypatch.setattr(
        cli_module,
        "write_kl8_feature_discovery_report",
        lambda report, path: Path(path),
    )
    monkeypatch.setattr(
        cli_module,
        "fixed_config_for_development_length",
        lambda length, n_jobs: object(),
    )

    assert (
        cli_module.main(
            [
                "--csv",
                str(csv_path),
                "--frozen-periods",
                "1",
                "--output",
                str(tmp_path / "report.json"),
                "--n-jobs",
                "1",
            ]
        )
        == 0
    )
    assert captured == {"periods": 40, "numbers": history.iloc[39]["numbers"]}


def test_empty_candidate_set_falls_back_to_uniform_no_signal() -> None:
    report = run_kl8_feature_discovery_v2(
        _history(40),
        _smoke_config(),
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
        candidate_feature_sets={},
    )

    assert report["selectedFeatureSet"] == "uniform"
    assert report["selectionReason"] == "no_eligible_feature_set"
    assert report["searchCandidates"] == []
    assert report["evaluation"]["trainCount"] == 0


def test_report_schema_safety_flags_and_immutable_write(tmp_path: Path) -> None:
    report = run_kl8_feature_discovery_v2(
        _history(40),
        _smoke_config(),
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
        candidate_feature_sets={},
    )

    assert report["schemaVersion"] == "kl8_feature_discovery_v2"
    assert report["evidenceStatus"] == "exploratory_feature_discovery_only"
    assert report["frozenRead"] is False
    assert report["promotionPassed"] is False
    assert report["recommendationEnabled"] is False
    assert report["userVisibleCandidates"] == []
    assert report["boundaries"]["development"]["periods"] == 40
    assert report["boundaries"]["frozen"]["periodsExcluded"] == 500
    assert len(report["dataSha256"]) == 64
    assert len(report["sourceFingerprint"]) == 64

    destination = tmp_path / "report.json"
    first = write_kl8_feature_discovery_report(report, destination)
    second = write_kl8_feature_discovery_report(report, destination)
    assert first == second == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["frozenRead"] is False
    changed = dict(report, selectionReason="changed")
    with pytest.raises(FileExistsError, match="不同内容"):
        write_kl8_feature_discovery_report(changed, destination)
