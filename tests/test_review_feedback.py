# -*- coding: utf-8 -*-

from dataclasses import dataclass

import pandas as pd

from src.analysis.pick_tracking import GroupEvaluation, PickEvaluation
from src.analysis.review_feedback import apply_review_feedback, build_review_feedback, recent_adjacent_repeat_count


@dataclass(frozen=True)
class DummyConfig:
    name: str


@dataclass(frozen=True)
class DummyResult:
    config: DummyConfig
    score: float


def _history() -> pd.DataFrame:
    latest = [1, 2, 3, 6, 12, 17, 23, 25, 28, 39, 48, 49, 52, 53, 55, 63, 68, 69, 73, 78]
    previous = [6, 10, 12, 18, 22, 28, 30, 33, 35, 38, 42, 48, 58, 59, 63, 64, 65, 71, 76, 79]
    rows = []
    for issue, nums in [("2026182", latest), ("2026181", previous)]:
        rows.append({"期数": issue, **{f"红球_{index + 1}": number for index, number in enumerate(nums)}})
    return pd.DataFrame(rows)


def _evaluation() -> PickEvaluation:
    return PickEvaluation(
        target_issue="2026182",
        source_issue="2026181",
        parameter_name="omission_mix",
        draw_numbers=[],
        group_results=[
            GroupEvaluation(1, [], [], 3, 0),
            GroupEvaluation(2, [], [], 4, 0),
            GroupEvaluation(3, [], [], 4, 0),
            GroupEvaluation(4, [], [], 2, 0),
        ],
        total_cost=8,
        total_prize=0,
        roi=-1.0,
    )


def test_recent_adjacent_repeat_count_counts_latest_two_draw_overlap():
    assert recent_adjacent_repeat_count(_history()) == 5


def test_build_review_feedback_summarizes_latest_evaluation():
    feedback = build_review_feedback(_history(), [_evaluation()], promoted_parameter="repeat_hot_mix")

    assert feedback.enabled is True
    assert feedback.latest_issue == "2026182"
    assert feedback.recent_repeat_count == 5
    assert feedback.hit4_count == 2
    assert feedback.promoted_parameter == "repeat_hot_mix"


def test_apply_review_feedback_promotes_repeat_strategy_when_near_miss_and_close_score():
    historical = DummyResult(DummyConfig("hot_omission"), -0.62)
    repeat = DummyResult(DummyConfig("repeat_hot_mix"), -0.73)

    selected, mode, feedback = apply_review_feedback(
        historical,
        [historical, repeat],
        mode="auto",
        history=_history(),
        evaluations=[_evaluation()],
        score_tolerance=0.18,
    )

    assert selected.config.name == "repeat_hot_mix"
    assert mode == "auto_review_feedback"
    assert feedback.enabled is True


def test_apply_review_feedback_does_not_override_manual_mode():
    manual = DummyResult(DummyConfig("hot_omission"), -0.62)
    repeat = DummyResult(DummyConfig("repeat_hot_mix"), -0.7)

    selected, mode, feedback = apply_review_feedback(
        manual,
        [manual, repeat],
        mode="manual",
        history=_history(),
        evaluations=[_evaluation()],
    )

    assert selected is manual
    assert mode == "manual"
    assert feedback.enabled is False
