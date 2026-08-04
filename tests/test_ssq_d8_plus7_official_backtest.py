# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.analysis.ssq_d8_plus7_official_backtest import (
    evaluate_d8_plus7_official_backtest,
)
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_prizegrades import SSQPrizeGrade, SSQPrizeGradeRecord
from src.lotteries.ssq import SSQ_RULE


def _draws(count: int) -> list[SSQDraw]:
    draws: list[SSQDraw] = []
    for index in range(count):
        red, blue = SSQ_RULE.validate_draw(
            tuple(sorted(((index + offset * 5) % 33) + 1 for offset in range(6))),
            index % 16 + 1,
        )
        draws.append(
            SSQDraw(
                str(2024001 + index),
                "2024-01-01",
                red,
                blue,
                "fixture",
                f"{index:064x}",
                {},
            )
        )
    return draws


def _record(issue: str) -> SSQPrizeGradeRecord:
    return SSQPrizeGradeRecord(
        issue=issue,
        prizegrades=tuple(SSQPrizeGrade(grade, 1, grade * 10) for grade in range(1, 7)),
        source_url="fixture",
        raw_hash="f" * 64,
        raw={},
        fetched_at="2026-01-01T00:00:00+00:00",
    )


def test_d8_plus7_backtest_constructs_35_tickets_from_prior_draws_and_settles_each() -> (
    None
):
    draws = _draws(121)

    report = evaluate_d8_plus7_official_backtest(draws, [_record(draws[-1].issue)])

    assert report["researchOnly"] is True
    assert report["predictionClaim"] is False
    assert report["formalRecommendationStatus"] == "uniform_abstain"
    assert report["history"] == {
        "warmupPeriods": 120,
        "evaluatedPeriods": 1,
        "firstEvaluatedIssue": draws[-1].issue,
        "lastEvaluatedIssue": draws[-1].issue,
    }
    assert report["summary"]["ticketsPerIssue"] == 35
    assert report["summary"]["totalCostYuan"] == "70.00"
    assert set(report["summary"]["prizeTierDistribution"]) == {
        "一等奖",
        "二等奖",
        "三等奖",
        "四等奖",
        "五等奖",
        "六等奖",
    }
    assert report["perIssue"][0]["tickets"] == 35
    assert report["perIssue"][0]["officialPrizeRawHash"] == "f" * 64


def test_d8_plus7_backtest_fails_closed_when_evaluated_issue_has_no_official_prizes() -> (
    None
):
    with pytest.raises(ValueError, match="缺少官方奖级"):
        evaluate_d8_plus7_official_backtest(_draws(121), [])


@pytest.mark.parametrize(
    "grades",
    [
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 5),
    ],
)
def test_d8_plus7_backtest_fails_closed_for_missing_or_duplicate_official_grade(
    grades: tuple[int, ...],
) -> None:
    draws = _draws(121)
    invalid = SSQPrizeGradeRecord(
        issue=draws[-1].issue,
        prizegrades=tuple(SSQPrizeGrade(grade, 1, grade * 10) for grade in grades),
        source_url="fixture",
        raw_hash="f" * 64,
        raw={},
        fetched_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="官方奖级不完整"):
        evaluate_d8_plus7_official_backtest(draws, [invalid])


def test_d8_plus7_official_backtest_script_writes_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ssq_d8_plus7_official_backtest_cli",
        root / "scripts/ssq_d8_plus7_official_backtest.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "report.json"
    expected = {"summary": {"ticketsPerIssue": 35}}
    monkeypatch.setattr(module, "load_official_history_csv", lambda _path: _draws(121))
    monkeypatch.setattr(module, "_records", lambda _path: [_record("2024121")])
    monkeypatch.setattr(
        module, "evaluate_d8_plus7_official_backtest", lambda *_: expected
    )

    assert (
        module.main(
            [
                "--csv",
                "history.csv",
                "--prizegrades",
                "prizes.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == expected
