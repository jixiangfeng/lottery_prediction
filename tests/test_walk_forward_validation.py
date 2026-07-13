# -*- coding: utf-8 -*-

from collections import Counter
from dataclasses import dataclass

from src.analysis.backtest import BacktestSummary
from src.analysis.walk_forward_validation import build_walk_forward_markdown, validate_parameter_walk_forward


@dataclass(frozen=True)
class DummyConfig:
    name: str


@dataclass(frozen=True)
class DummyWindow:
    label: str
    summary: BacktestSummary


@dataclass(frozen=True)
class DummySliding:
    windows: list[DummyWindow]


@dataclass(frozen=True)
class DummyResult:
    config: DummyConfig
    sliding_summary: DummySliding
    score: float = 0.0


def _summary(roi: float, hit: float = 4.5) -> BacktestSummary:
    return BacktestSummary(
        draw_count=100,
        group_count=10,
        total_bets=1000,
        ticket_price=2,
        total_cost=2000,
        total_prize=int(2000 * (1 + roi)),
        roi=roi,
        average_hit=hit,
        hit_distribution=Counter({4: 10}),
        max_miss_streak=8,
    )


def test_validate_parameter_walk_forward_marks_stable_parameter():
    result = DummyResult(
        DummyConfig("stable"),
        DummySliding([DummyWindow("w1", _summary(-0.1)), DummyWindow("w2", _summary(-0.05)), DummyWindow("w3", _summary(-0.08))]),
    )

    validation = validate_parameter_walk_forward([result], train_window_count=2)

    assert validation.bestParameter == "stable"
    assert validation.rows[0].testWindowCount == 1
    assert validation.rows[0].riskLevel == "低"
    assert validation.rows[0].trainMeanRoi == -0.075
    assert validation.rows[0].testMeanRoi == -0.08


def test_validate_parameter_walk_forward_penalizes_train_test_gap():
    good_train_bad_test = DummyResult(
        DummyConfig("overfit"),
        DummySliding([DummyWindow("w1", _summary(0.5)), DummyWindow("w2", _summary(0.4)), DummyWindow("w3", _summary(-0.5))]),
    )

    validation = validate_parameter_walk_forward([good_train_bad_test], train_window_count=2)

    assert validation.rows[0].riskLevel == "高"
    assert validation.rows[0].generalizationGap > 0.5


def test_build_walk_forward_markdown_contains_title():
    result = DummyResult(
        DummyConfig("stable"),
        DummySliding([DummyWindow("w1", _summary(-0.1)), DummyWindow("w2", _summary(-0.05)), DummyWindow("w3", _summary(-0.08))]),
    )
    validation = validate_parameter_walk_forward([result], train_window_count=2)

    markdown = build_walk_forward_markdown(validation)

    assert "反过拟合前推验证" in markdown
    assert "stable" in markdown
