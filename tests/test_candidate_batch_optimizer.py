# -*- coding: utf-8 -*-

from dataclasses import dataclass

from src.analysis.candidate_batch_optimizer import select_best_candidate_batch
from src.analysis.candidate_coverage import analyze_candidate_coverage
from src.analysis.candidate_portfolio_score import score_candidate_portfolio


@dataclass(frozen=True)
class DummyGroup:
    numbers: list[int]
    score: float = 80.0


def _score(groups):
    coverage = analyze_candidate_coverage(groups)
    return score_candidate_portfolio(groups, coverage, number_range_size=20, group_size=5)


def test_select_best_candidate_batch_picks_highest_portfolio_score():
    repeated = [DummyGroup([1, 2, 3, 4, 5]), DummyGroup([1, 2, 3, 4, 5]), DummyGroup([1, 2, 3, 4, 5])]
    diverse = [DummyGroup([1, 2, 3, 4, 5]), DummyGroup([6, 7, 8, 9, 10]), DummyGroup([11, 12, 13, 14, 15])]

    result = select_best_candidate_batch([repeated, diverse], _score)

    assert result.best_index == 2
    assert result.trial_count == 2
    assert result.best_score.final_score > 70
    assert result.groups == diverse


def test_select_best_candidate_batch_handles_single_batch():
    batch = [DummyGroup([1, 2, 3, 4, 5])]

    result = select_best_candidate_batch([batch], _score)

    assert result.best_index == 1
    assert result.trial_count == 1
    assert result.groups == batch
