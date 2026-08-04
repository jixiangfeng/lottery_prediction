from __future__ import annotations

import pytest

from src.analysis.ssq_d8_official_backtest import evaluate_d8_official_backtest
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


def test_d8_backtest_requires_every_evaluated_issue_to_have_official_prizes() -> None:
    with pytest.raises(ValueError, match="缺少官方奖级"):
        evaluate_d8_official_backtest(_draws(121), [])


def test_d8_backtest_rejects_duplicate_or_missing_official_prize_grades() -> None:
    draws = _draws(121)
    invalid = SSQPrizeGradeRecord(
        issue=draws[-1].issue,
        prizegrades=tuple(
            SSQPrizeGrade(grade, 1, grade * 10) for grade in (1, 2, 3, 4, 5, 5)
        ),
        source_url="fixture",
        raw_hash="f" * 64,
        raw={},
        fetched_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="官方奖级不完整"):
        evaluate_d8_official_backtest(draws, [invalid])


def test_d8_backtest_replays_ticket_level_settlement_and_reports_roi() -> None:
    draws = _draws(121)
    report = evaluate_d8_official_backtest(draws, [_record(draws[-1].issue)])

    assert report["researchOnly"] is True
    assert report["predictionClaim"] is False
    assert report["formalRecommendationStatus"] == "uniform_abstain"
    assert report["history"]["evaluatedPeriods"] == 1
    assert report["summary"]["totalCostYuan"] == "56.00"
    assert report["summary"]["ticketsPerIssue"] == 28
    assert report["perIssue"][0]["issue"] == draws[-1].issue
