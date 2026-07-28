# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy

import pytest

from src.analysis.ssq_diversified_portfolio_v2 import (
    build_diversified_portfolio_v2,
    ranked_balls,
    validate_diversified_portfolio_v2,
)


def _probabilities(size: int) -> list[float]:
    return [float(size - index) for index in range(size)]


def test_fixed_builder_is_deterministic_and_satisfies_all_hard_constraints() -> None:
    red = _probabilities(33)
    blue = _probabilities(16)
    first = build_diversified_portfolio_v2(red, blue)
    second = build_diversified_portfolio_v2(red, blue)

    assert first == second
    groups = first["groups"]
    red_sets = [tuple(group["red"]) for group in groups]
    blues = [group["blue"] for group in groups]
    expanded = [
        (tuple(ticket["red"]), ticket["blue"])
        for group in groups
        for ticket in group["expandedTickets"]
    ]
    assert len(red_sets) == len(set(red_sets)) == 5
    assert set().union(*(set(red7) for red7 in red_sets)) == set(range(1, 34))
    assert max(sum(ball in red7 for red7 in red_sets) for ball in range(1, 34)) == 2
    assert (
        max(
            len(set(red_sets[left]).intersection(red_sets[right]))
            for left in range(5)
            for right in range(left + 1, 5)
        )
        <= 3
    )
    assert len(blues) == len(set(blues)) == 5
    assert len(expanded) == len(set(expanded)) == 35


def test_top5_blue_ranking_is_distinct_probability_descending_ball_ascending() -> None:
    blue = [0.0] * 16
    for ball, probability in {8: 0.9, 3: 0.9, 5: 0.8, 2: 0.7, 16: 0.7}.items():
        blue[ball - 1] = probability
    portfolio = build_diversified_portfolio_v2(_probabilities(33), blue)

    assert ranked_balls(blue, 16)[:5] == (3, 8, 5, 2, 16)
    assert portfolio["blueTop5Ranking"] == [3, 8, 5, 2, 16]
    assert [group["blue"] for group in portfolio["groups"]] == [3, 8, 5, 2, 16]


def test_audit_mismatch_fails_closed() -> None:
    red = _probabilities(33)
    blue = _probabilities(16)
    portfolio = build_diversified_portfolio_v2(red, blue)
    tampered = deepcopy(portfolio)
    tampered["groups"][0]["blue"] = tampered["groups"][1]["blue"]

    with pytest.raises(ValueError):
        validate_diversified_portfolio_v2(
            tampered, red_probabilities=red, blue_probabilities=blue
        )
