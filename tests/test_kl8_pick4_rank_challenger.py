# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.kl8_pick4_rank_challenger import main as challenger_main
from src.analysis.kl8_pick4_rank_challenger import (
    PICK4_RANK_FEATURES,
    Kl8Pick4RankConfig,
    audit_rank_probabilities,
    build_pick4_rank_panel,
    build_pick4_ranker,
    ranked_pick4_portfolio,
    run_pick4_rank_challenger,
    write_pick4_rank_report,
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


def _smoke_config() -> Kl8Pick4RankConfig:
    return Kl8Pick4RankConfig(
        initial_train=20,
        evaluation_periods=20,
        refit_interval=10,
        stability_blocks=5,
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


def test_ranked_pick4_portfolio_is_five_disjoint_balanced_tickets() -> None:
    scores = np.arange(80, dtype=np.float64)
    tickets = ranked_pick4_portfolio(scores)

    assert tickets == [
        [61, 66, 71, 76],
        [62, 67, 72, 77],
        [63, 68, 73, 78],
        [64, 69, 74, 79],
        [65, 70, 75, 80],
    ]
    assert len({number for ticket in tickets for number in ticket}) == 20
    assert all(len(ticket) == len(set(ticket)) == 4 for ticket in tickets)


def test_pick4_rank_features_are_prior_only() -> None:
    history = _history(40)
    original = build_pick4_rank_panel(history)
    mutated = history.copy(deep=True)
    mutated.loc[25:, "numbers"] = mutated.loc[25:, "numbers"].map(
        lambda values: sorted({((value + 37) % 80) + 1 for value in values})
    )
    changed = build_pick4_rank_panel(mutated)

    left = original.loc[original["periodIndex"] == 25, PICK4_RANK_FEATURES]
    right = changed.loc[changed["periodIndex"] == 25, PICK4_RANK_FEATURES]
    np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())


def test_ranker_locks_top4_truncation_and_binary_label_gain() -> None:
    config = _smoke_config()
    params = build_pick4_ranker(config).get_params()

    assert params["lambdarank_truncation_level"] == 4
    assert params["label_gain"] == [0, 1]


def test_rank_probability_audit_is_bounded_sum20_and_zero_lambda_is_uniform() -> None:
    scores = np.linspace(-3.0, 4.0, 80)
    config = _smoke_config()
    probabilities = audit_rank_probabilities(scores, config)
    uniform = audit_rank_probabilities(
        scores, replace(config, probability_shrinkage=0.0)
    )

    assert np.isfinite(probabilities).all()
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert probabilities.sum() == pytest.approx(20.0)
    np.testing.assert_array_equal(uniform, np.full(80, 0.25))


def test_tiny_pick4_rank_challenger_is_deterministic_and_safe() -> None:
    history = _history(40)
    config = _smoke_config()
    first = run_pick4_rank_challenger(
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
    )
    second = run_pick4_rank_challenger(
        history,
        config,
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
    )

    assert _without_elapsed(first) == _without_elapsed(second)
    assert first["schemaVersion"] == "kl8_pick4_rank_challenger_v2"
    assert first["evidenceStatus"] == "exploratory_post_failure_reused_development"
    assert first["frozenRead"] is False
    assert first["promotionPassed"] is False
    assert first["recommendationEnabled"] is False
    assert first["formalRecommendation"] is None
    assert first["userVisibleCandidates"] == []
    assert first["baseline"]["meanHitsPerTicket"] == pytest.approx(1.0)
    assert first["baseline"]["portfolioMeanTotalHits"] == pytest.approx(5.0)
    assert len(first["evaluation"]["blocks"]) == 5
    assert first["evaluation"]["periods"] == 20
    assert first["evaluation"]["trainCount"] == 2
    assert 0.0 <= first["evaluation"]["deltaLogLossBootstrapPValue"] <= 1.0
    assert 0.0 <= first["evaluation"]["deltaBrierBootstrapPValue"] <= 1.0
    assert set(first["evaluation"]["holmAdjustedPValues"]) == {
        "primaryTop4",
        "portfolioTop20",
        "deltaLogLoss",
        "deltaBrier",
    }


def test_pick4_rank_report_is_immutable(tmp_path: Path) -> None:
    report = run_pick4_rank_challenger(
        _history(40),
        _smoke_config(),
        frozen_periods_excluded=500,
        frozen_boundary={"firstIssue": "F001", "lastIssue": "F500"},
    )
    path = tmp_path / "report.json"

    assert write_pick4_rank_report(report, path) == path
    assert write_pick4_rank_report(report, path) == path
    assert json.loads(path.read_text(encoding="utf-8"))["frozenRead"] is False
    with pytest.raises(FileExistsError, match="不同内容"):
        write_pick4_rank_report(dict(report, gatePassed=True), path)


def test_pick4_rank_cli_does_not_parse_poisoned_frozen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "kl8.csv"
    history = _history(42)
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
        for row in history.iloc[40:].itertuples(index=False):
            writer.writerow(
                {"issue": row.issue, "date": row.date, "numbers": "POISON_FROZEN"}
            )

    output = tmp_path / "challenger.json"
    assert (
        challenger_main(
            [
                "--csv",
                str(csv_path),
                "--frozen-periods",
                "2",
                "--output",
                str(output),
                "--smoke",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "frozenRead=false" in printed
    assert "exploratory_post_failure_reused_development" in printed
    assert payload["boundaries"]["development"]["periods"] == 40
    assert payload["boundaries"]["frozen"]["numbersRead"] is False
    assert payload["userVisibleCandidates"] == []
