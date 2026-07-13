# -*- coding: utf-8 -*-

from dataclasses import dataclass

from src.analysis.candidate_coverage import analyze_candidate_coverage
from src.analysis.candidate_portfolio_score import score_candidate_portfolio, build_portfolio_score_markdown


@dataclass(frozen=True)
class DummyGroup:
    numbers: list[int]
    score: float


def test_score_candidate_portfolio_balances_coverage_overlap_and_structure():
    groups = [
        DummyGroup([1, 2, 3, 4, 5], 80),
        DummyGroup([6, 7, 8, 9, 10], 70),
        DummyGroup([11, 12, 13, 14, 15], 60),
    ]
    coverage = analyze_candidate_coverage(groups)

    result = score_candidate_portfolio(groups, coverage, number_range_size=20, group_size=5)

    assert result.group_count == 3
    assert result.coverage_score == 0.75
    assert result.overlap_score == 1.0
    assert 0 < result.final_score <= 100
    assert result.grade in {"优秀", "良好", "一般", "偏弱"}


def test_score_candidate_portfolio_penalizes_repeated_groups():
    repeated = [
        DummyGroup([1, 2, 3, 4, 5], 80),
        DummyGroup([1, 2, 3, 4, 5], 80),
        DummyGroup([1, 2, 3, 4, 5], 80),
    ]
    diverse = [
        DummyGroup([1, 2, 3, 4, 5], 80),
        DummyGroup([6, 7, 8, 9, 10], 80),
        DummyGroup([11, 12, 13, 14, 15], 80),
    ]

    repeated_score = score_candidate_portfolio(repeated, analyze_candidate_coverage(repeated), number_range_size=20, group_size=5)
    diverse_score = score_candidate_portfolio(diverse, analyze_candidate_coverage(diverse), number_range_size=20, group_size=5)

    assert diverse_score.final_score > repeated_score.final_score
    assert repeated_score.overlap_score == 0.0


def test_score_candidate_portfolio_penalizes_extreme_single_group_shapes():
    balanced = [
        DummyGroup([1, 2, 3, 4, 5, 46, 47, 48, 49, 50], 80),
        DummyGroup([11, 12, 13, 14, 15, 56, 57, 58, 59, 60], 80),
    ]
    extreme = [
        DummyGroup([1, 3, 5, 7, 9, 11, 13, 15, 17, 61], 80),
        DummyGroup([2, 4, 6, 8, 10, 12, 14, 16, 18, 62], 80),
    ]

    balanced_score = score_candidate_portfolio(balanced, analyze_candidate_coverage(balanced))
    extreme_score = score_candidate_portfolio(extreme, analyze_candidate_coverage(extreme))

    assert balanced_score.shape_balance_score > extreme_score.shape_balance_score
    assert balanced_score.final_score > extreme_score.final_score


def test_build_portfolio_score_markdown_contains_final_score():
    groups = [DummyGroup([1, 2, 3, 4, 5], 80), DummyGroup([6, 7, 8, 9, 10], 70)]
    result = score_candidate_portfolio(groups, analyze_candidate_coverage(groups), number_range_size=20, group_size=5)

    markdown = build_portfolio_score_markdown(result)

    assert "候选组合总评分" in markdown
    assert "最终评分" in markdown
