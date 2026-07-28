# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

import src.analysis.dlt_ridge_candidates_v1 as models
from src.analysis.dlt_ridge_candidates_v1 import (
    CANDIDATE_IDS,
    BlockPrediction,
    FeatureSnapshotCache,
    build_candidate_features,
    fit_candidate_scores,
    iter_block_walk_forward_predictions,
    predict_block_walk_forward,
)


@dataclass(frozen=True)
class Draw:
    front: tuple[int, ...]
    back: tuple[int, ...]


def draw(front: tuple[int, ...], back: tuple[int, ...]) -> Draw:
    return Draw(front=front, back=back)


def cyclic_draws(count: int) -> list[Draw]:
    return [
        draw(
            tuple(sorted({(index * 5 + offset) % 35 + 1 for offset in range(5)})),
            tuple(sorted({(index * 2) % 12 + 1, (index * 2 + 5) % 12 + 1})),
        )
        for index in range(count)
    ]


def test_candidate_ids_are_exactly_frozen_four() -> None:
    assert CANDIDATE_IDS == (
        "C1_LONG_RIDGE",
        "C2_MULTISCALE_RIDGE",
        "C3_PAIR_GRAPH_RIDGE",
        "C4_EQUAL_LOGPOOL",
    )


def test_c1_feature_equations_are_exact_on_small_fixture() -> None:
    history = [
        draw((1, 2, 3, 4, 5), (1, 2)),
        draw((1, 6, 7, 8, 9), (1, 3)),
        draw((2, 6, 10, 11, 12), (2, 4)),
    ]

    features = build_candidate_features(history, "C1_LONG_RIDGE")
    assert features.names == ("frequency500", "ewma365", "gap365", "sqrt_gap365")
    expected_ewma = (2.0 ** (-2.0 / 365.0) + 2.0 ** (-1.0 / 365.0)) / sum(
        2.0 ** (-age / 365.0) for age in (2, 1, 0)
    )
    assert features.front[0] == pytest.approx((2.0 / 3.0, expected_ewma, 1.0, 1.0))
    assert features.front[1][0] == pytest.approx(2.0 / 3.0)
    assert features.front[1][2:] == pytest.approx((0.0, 0.0))
    assert features.front[34][2:] == pytest.approx((3.0, math.sqrt(3.0)))


def test_c2_feature_equations_use_fixed_windows_and_contrasts() -> None:
    history = cyclic_draws(25)
    features = build_candidate_features(history, "C2_MULTISCALE_RIDGE")
    number = 1
    indicators = [number in item.front for item in history]

    def frequency(window: int) -> float:
        values = indicators[-window:]
        return sum(values) / len(values)

    def ewma(half_life: int) -> float:
        weights = [2.0 ** (-age / half_life) for age in reversed(range(len(history)))]
        return sum(weight * value for weight, value in zip(weights, indicators)) / sum(
            weights
        )

    row = features.front[number - 1]
    assert features.names == (
        "frequency20",
        "frequency60",
        "frequency200",
        "ewma30",
        "ewma120",
        "gap200",
        "frequency20_minus_frequency200",
        "ewma30_minus_ewma120",
    )
    assert row == pytest.approx(
        (
            frequency(20),
            frequency(60),
            frequency(200),
            ewma(30),
            ewma(120),
            3.0,
            frequency(20) - frequency(200),
            ewma(30) - ewma(120),
        )
    )


def test_c3_graph_feature_equations_include_exact_50_step_pagerank() -> None:
    history = [
        draw((1, 2, 3, 4, 5), (1, 2)),
        draw((1, 2, 6, 7, 8), (1, 3)),
    ]
    features = build_candidate_features(history, "C3_PAIR_GRAPH_RIDGE")
    row = features.front[0]
    # Number 1 has four co-occurrences in each draw: weighted degree 8.
    assert features.names[-4:] == (
        "graph_weighted_degree",
        "graph_mean_cooccurrence",
        "graph_pagerank_0_85_50",
        "graph_degree_minus_zone_average",
    )
    assert row[-4] == pytest.approx(8.0)
    assert row[-3] == pytest.approx(8.0 / 34.0)
    assert row[-1] == pytest.approx(8.0 - 40.0 / 35.0)
    # Independent literal implementation of damping=.85 and exactly 50 iterations.
    size = 35
    adjacency = [[0.0] * size for _ in range(size)]
    for item in history:
        for left in item.front:
            for right in item.front:
                if left != right:
                    adjacency[left - 1][right - 1] += 1.0
    ranks = [1.0 / size] * size
    for _ in range(50):
        updated = [(1.0 - 0.85) / size] * size
        for source in range(size):
            degree = sum(adjacency[source])
            if degree:
                for target in range(size):
                    updated[target] += (
                        0.85 * ranks[source] * adjacency[source][target] / degree
                    )
            else:
                for target in range(size):
                    updated[target] += 0.85 * ranks[source] / size
        ranks = updated
    assert row[-2] == pytest.approx(ranks[0], abs=1e-15)


def test_fits_are_deterministic_and_front_back_are_separate() -> None:
    history = cyclic_draws(18)
    first = fit_candidate_scores(history, "C2_MULTISCALE_RIDGE")
    second = fit_candidate_scores(history, "C2_MULTISCALE_RIDGE")
    assert first == second
    assert len(first.front) == 35
    assert len(first.back) == 12
    assert all(math.isfinite(value) for value in first.front + first.back)

    changed_back = [Draw(item.front, (11, 12)) for item in history]
    changed = fit_candidate_scores(changed_back, "C2_MULTISCALE_RIDGE")
    assert changed.front == first.front
    assert changed.back != first.back


def test_snapshot_cache_is_prefix_invariant_and_contains_target_1900() -> None:
    original = cyclic_draws(31)
    changed = original + [draw((31, 32, 33, 34, 35), (11, 12))] * 4
    first = FeatureSnapshotCache(original, stop_index=31)
    second = FeatureSnapshotCache(changed, stop_index=31)
    for candidate_id in CANDIDATE_IDS[:3]:
        assert first.features(candidate_id, 30) == second.features(candidate_id, 30)
    large = FeatureSnapshotCache(cyclic_draws(1901), stop_index=1901)
    assert large.features("C1_LONG_RIDGE", 1900).front


def test_vectorized_ridge_matches_literal_reference_on_tiny_fixture() -> None:
    rows = [
        (-1.0, 0.25),
        (0.0, -0.5),
        (0.75, 1.0),
        (1.5, -0.25),
        (-0.4, 0.8),
    ]
    labels = [0, 0, 1, 1, 0]
    optimized = models._fit_ridge_logit(rows, labels)
    reference = models._fit_ridge_logit_reference(rows, labels)
    assert np.asarray(optimized) == pytest.approx(reference, abs=1e-10)


def test_c4_is_exact_arithmetic_mean_of_three_raw_logits() -> None:
    history = cyclic_draws(12)
    components = [
        fit_candidate_scores(history, candidate) for candidate in CANDIDATE_IDS[:3]
    ]
    pooled = fit_candidate_scores(history, "C4_EQUAL_LOGPOOL")
    assert pooled.front == tuple(
        sum(component.front[index] for component in components) / 3.0
        for index in range(35)
    )
    assert pooled.back == tuple(
        sum(component.back[index] for component in components) / 3.0
        for index in range(12)
    )


def test_walk_forward_refits_only_at_global_multiples_of_25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_fit(prefix: object, candidate_id: str) -> models.CandidateScores:
        calls.append(len(prefix))  # type: ignore[arg-type]
        value = float(len(prefix))  # type: ignore[arg-type]
        return models.CandidateScores((value,) * 35, (-value,) * 12)

    monkeypatch.setattr(models, "fit_candidate_scores", fake_fit)
    predictions = list(
        iter_block_walk_forward_predictions(
            cyclic_draws(57), "C1_LONG_RIDGE", start_index=23
        )
    )
    assert calls == [0, 25, 50]
    assert [item.fit_cutoff for item in predictions[:3]] == [0, 0, 25]
    assert all(item.scores == predictions[2].scores for item in predictions[2:27])
    assert predictions[27].fit_cutoff == 50
    assert [item.target_index for item in predictions] == list(range(23, 57))


def test_walk_forward_is_invariant_to_suffix_and_later_label_changes() -> None:
    history = cyclic_draws(31)
    original = predict_block_walk_forward(
        history, "C1_LONG_RIDGE", start_index=25, stop_index=31
    )
    suffix_changed = history[:31] + [draw((31, 32, 33, 34, 35), (11, 12))] * 4
    with_suffix = predict_block_walk_forward(
        suffix_changed, "C1_LONG_RIDGE", start_index=25, stop_index=31
    )
    later_changed = list(history)
    later_changed[29] = draw((31, 32, 33, 34, 35), (11, 12))
    changed = predict_block_walk_forward(
        later_changed, "C1_LONG_RIDGE", start_index=25, stop_index=31
    )
    assert original == with_suffix == changed
    assert all(isinstance(item, BlockPrediction) for item in original)


def test_invalid_candidate_and_ranges_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_candidate_scores(cyclic_draws(3), "C5")
    with pytest.raises(ValueError):
        build_candidate_features(cyclic_draws(3), "C4_EQUAL_LOGPOOL")
    with pytest.raises(ValueError):
        list(
            iter_block_walk_forward_predictions(
                cyclic_draws(3), "C1_LONG_RIDGE", start_index=-1
            )
        )
