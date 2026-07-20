# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.digit_data import load_digit_development_csv
from src.analysis.digit_predictability_audit import (
    PredictabilityAuditConfig,
    run_predictability_audit,
)
from src.lotteries import get_lottery_rule


def _history(periods: int, *, deterministic: bool = False) -> pd.DataFrame:
    rows = []
    state = 20260719
    for index in range(periods):
        if deterministic:
            digits = (index % 10, (index + 3) % 10, (index + 7) % 10)
        else:
            values = []
            for _ in range(3):
                state = (1103515245 * state + 12345) % (2**31)
                values.append(state % 10)
            digits = tuple(values)
        rows.append(
            {
                "期数": str(2026001 + index),
                "百位": digits[0],
                "十位": digits[1],
                "个位": digits[2],
            }
        )
    return pd.DataFrame(rows)


def test_predictability_audit_is_deterministic_and_complete():
    config = PredictabilityAuditConfig(
        min_train_size=50,
        permutation_trials=39,
        block_size=10,
        seed=7,
    )
    history = _history(180)
    first = run_predictability_audit(history, get_lottery_rule("fc3d"), config)
    second = run_predictability_audit(history, get_lottery_rule("fc3d"), config)

    assert first.to_dict() == second.to_dict()
    assert first.evaluated_targets == 130
    assert len(first.sequence_tests) == 18
    assert len(first.baselines) == 7
    assert first.frozen_test_read is False
    assert all(0 <= item.p_value <= 1 for item in first.sequence_tests)
    assert all(0 <= item.q_value <= 1 for item in first.sequence_tests)


def test_deterministic_cycle_has_corrected_sequence_signal():
    report = run_predictability_audit(
        _history(240, deterministic=True),
        get_lottery_rule("pl3"),
        PredictabilityAuditConfig(
            min_train_size=50,
            permutation_trials=99,
            block_size=10,
            seed=11,
        ),
    )
    assert any(item.passed_fdr for item in report.sequence_tests)


def test_development_csv_loader_does_not_parse_frozen_numbers(tmp_path):
    history = _history(220).sort_values("期数", ascending=False).astype(object)
    frozen_issues = set(
        history.sort_values("期数", ascending=True).tail(20)["期数"].astype(str)
    )
    mask = history["期数"].astype(str).isin(frozen_issues)
    history.loc[mask, ["百位", "十位", "个位"]] = "UNREAD"
    path = tmp_path / "history.csv"
    history.to_csv(path, index=False)

    development, total = load_digit_development_csv(
        path, get_lottery_rule("fc3d"), frozen_test_periods=20
    )
    assert total == 220
    assert len(development) == 200
    assert not set(development["期数"]).intersection(frozen_issues)


def test_predictability_audit_rejects_pl5():
    with pytest.raises(ValueError, match="只支持fc3d/pl3"):
        run_predictability_audit(
            _history(180),
            get_lottery_rule("pl5"),
            PredictabilityAuditConfig(min_train_size=50, permutation_trials=19),
        )
