# -*- coding: utf-8 -*-

import math

import pytest

from src.analysis.prediction_viability import (
    build_prediction_viability_report,
    calculate_group_random_probability,
    evaluate_viability_metric,
    poisson_binomial_right_tail,
)


def _even_hits(periods: int, hits_per_block: int) -> list[bool]:
    values = [False] * periods
    for block_index in range(3):
        start = block_index * periods // 3
        end = (block_index + 1) * periods // 3
        for offset in range(hits_per_block):
            values[start + offset * max(1, (end - start) // hits_per_block)] = True
    return values


def _hits_by_block(periods: int, counts: tuple[int, int, int]) -> list[bool]:
    values = [False] * periods
    for block_index, count in enumerate(counts):
        start = block_index * periods // 3
        end = (block_index + 1) * periods // 3
        for offset in range(count):
            values[start + offset * max(1, (end - start) // count)] = True
    return values


def test_poisson_binomial_matches_binomial_for_constant_probability():
    probabilities = [0.1] * 10
    expected = sum(
        math.comb(10, hits) * 0.1**hits * 0.9 ** (10 - hits) for hits in range(3, 11)
    )

    assert poisson_binomial_right_tail(probabilities, 3) == pytest.approx(expected)


def test_poisson_binomial_supports_varying_issue_probabilities():
    assert poisson_binomial_right_tail([0.1, 0.2, 0.3], 2) == pytest.approx(0.098)


def test_three_hundred_periods_cannot_pass_minimum_sample_gate():
    probabilities = [0.01] * 300
    eight_hits = evaluate_viability_metric(
        "direct", _hits_by_block(300, (3, 3, 2)), probabilities
    )
    nine_hits = evaluate_viability_metric("direct", _even_hits(300, 3), probabilities)

    assert eight_hits.p_value >= 0.01
    assert nine_hits.p_value < 0.01
    assert nine_hits.conditions["enoughPeriods"] is False
    assert nine_hits.viable is False


def test_five_hundred_periods_need_twelve_evenly_distributed_direct_hits():
    probabilities = [0.01] * 500
    eleven_hits = evaluate_viability_metric(
        "direct", _hits_by_block(500, (4, 4, 3)), probabilities
    )
    twelve_hits = evaluate_viability_metric("direct", _even_hits(500, 4), probabilities)

    assert eleven_hits.p_value >= 0.01
    assert twelve_hits.p_value < 0.01
    assert twelve_hits.wilson_lower_bound_99 > 0.01
    assert twelve_hits.viable is True


def test_one_weak_time_block_rejects_apparent_overall_advantage():
    probabilities = [0.01] * 500
    hits = [False] * 500
    for index in range(20):
        hits[index] = True

    metric = evaluate_viability_metric("direct", hits, probabilities)

    assert metric.p_value < 0.01
    assert metric.conditions["stableAcrossBlocks"] is False
    assert metric.viable is False


def test_group_random_probability_counts_ordered_number_coverage():
    group6 = ["012", "013", "014", "015", "016", "017", "018", "019"]
    group3 = ["001", "002"]

    assert calculate_group_random_probability(group6 + group3) == pytest.approx(
        54 / 1000
    )


def test_three_digit_report_requires_direct_and_group_gates_to_pass():
    direct_hits = _even_hits(500, 4)
    group_hits = _even_hits(500, 10)
    report = build_prediction_viability_report(
        direct_hits,
        [0.01] * 500,
        group_hits=group_hits,
        group_random_probabilities=[0.054] * 500,
    )

    assert report.direct_gate.viable is True
    assert report.group_gate is not None
    assert report.group_gate.viable is False
    assert report.viable is False
    assert report.to_dict()["groupGate"]["conditions"]
