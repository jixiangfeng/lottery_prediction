# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import math

import pytest

from src.analysis.dlt_fixed_cardinality_v1 import (
    FixedCardinalityDistribution,
    fixed_cardinality_marginals,
    fixed_cardinality_observed_log_probability,
)


def _brute_force(
    scores: list[float], k: int, tau: float, epsilon: float
) -> tuple[dict[tuple[int, ...], float], list[float]]:
    subsets = list(itertools.combinations(range(len(scores)), k))
    logits = [sum(scores[index] / tau for index in subset) for subset in subsets]
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    total = sum(weights)
    uniform = 1.0 / len(subsets)
    probabilities = {
        subset: (1.0 - epsilon) * weight / total + epsilon * uniform
        for subset, weight in zip(subsets, weights, strict=True)
    }
    marginals = [
        sum(probability for subset, probability in probabilities.items() if i in subset)
        for i in range(len(scores))
    ]
    return probabilities, marginals


@pytest.mark.parametrize("n,k", [(n, k) for n in range(2, 9) for k in range(1, n + 1)])
def test_small_distribution_agrees_with_brute_force(n: int, k: int) -> None:
    scores = [((index * 7) % 11 - 5) / 3.0 for index in range(n)]
    epsilon = 0.1
    distribution = FixedCardinalityDistribution(scores, k, tau=0.75, epsilon=epsilon)
    probabilities, expected_marginals = _brute_force(scores, k, 0.75, epsilon)

    assert distribution.log_partition == pytest.approx(
        math.log(
            sum(
                math.exp(sum(scores[index] / 0.75 for index in subset))
                for subset in probabilities
            )
        ),
        abs=1e-12,
    )
    assert distribution.marginals == pytest.approx(expected_marginals, abs=1e-12)
    for subset, probability in probabilities.items():
        assert distribution.log_probability(subset) == pytest.approx(
            math.log(probability), abs=1e-12
        )


def test_real_dlt_zone_sizes_have_exact_cardinality_marginals() -> None:
    front = fixed_cardinality_marginals(
        [math.sin(index) for index in range(35)], 5, tau=0.5, epsilon=0.05
    )
    back = fixed_cardinality_marginals(
        [math.cos(index) for index in range(12)], 2, tau=2.0, epsilon=0.2
    )

    assert len(front) == 35
    assert len(back) == 12
    assert sum(front) == pytest.approx(5.0, abs=1e-12)
    assert sum(back) == pytest.approx(2.0, abs=1e-12)
    assert all(0.0 <= probability <= 1.0 for probability in front + back)


def test_uniform_scores_are_uniform_at_all_valid_temperatures() -> None:
    for tau in (1e-12, 1.0, 1e12):
        distribution = FixedCardinalityDistribution([3.5] * 8, 3, tau=tau)
        assert distribution.marginals == pytest.approx([3.0 / 8.0] * 8, abs=1e-12)
        assert distribution.log_probability((0, 3, 7)) == pytest.approx(
            -math.log(math.comb(8, 3)), abs=1e-12
        )


def test_small_temperature_remains_stable_and_concentrates_on_top_set() -> None:
    distribution = FixedCardinalityDistribution(
        [0.0, 1.0, 2.0, 3.0], 2, tau=1e-9, epsilon=0.0
    )

    assert math.isfinite(distribution.log_partition)
    assert distribution.log_probability((2, 3)) == pytest.approx(0.0, abs=1e-10)
    assert distribution.marginals == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-10)


def test_uniform_mixture_is_applied_to_set_probability_and_marginals() -> None:
    scores = [-2.0, -1.0, 1.0, 4.0]
    expected_probabilities, expected_marginals = _brute_force(scores, 2, 0.5, 1.0)

    assert fixed_cardinality_marginals(
        scores, 2, tau=0.5, epsilon=1.0
    ) == pytest.approx(expected_marginals, abs=1e-12)
    assert fixed_cardinality_observed_log_probability(
        scores, (0, 3), 2, tau=0.5, epsilon=1.0
    ) == pytest.approx(math.log(expected_probabilities[(0, 3)]), abs=1e-12)


@pytest.mark.parametrize(
    "scores,k,tau,epsilon",
    [
        ([0.0, math.nan], 1, 1.0, 0.0),
        ([0.0, math.inf], 1, 1.0, 0.0),
        ([0.0, 1.0], 0, 1.0, 0.0),
        ([0.0, 1.0], 3, 1.0, 0.0),
        ([0.0, 1.0], 1, 0.0, 0.0),
        ([0.0, 1.0], 1, math.inf, 0.0),
        ([0.0, 1.0], 1, 1.0, -0.1),
        ([0.0, 1.0], 1, 1.0, 1.1),
    ],
)
def test_invalid_distribution_inputs_fail_closed(
    scores: list[float], k: int, tau: float, epsilon: float
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FixedCardinalityDistribution(scores, k, tau=tau, epsilon=epsilon)


def test_observed_set_must_be_complete_unique_and_in_range() -> None:
    distribution = FixedCardinalityDistribution([0.0, 1.0, 2.0], 2)

    for observed in ((0,), (0, 0), (-1, 2), (1, 3), (True, 2)):
        with pytest.raises((TypeError, ValueError)):
            distribution.log_probability(observed)
