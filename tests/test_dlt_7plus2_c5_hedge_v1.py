# -*- coding: utf-8 -*-
"""大乐透C5在线Hedge融合的严格前序回归测试。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from src.analysis.dlt_7plus2_c5_hedge_v1 import (
    C5_BACK_EPSILON,
    C5_BACK_TAU,
    C5_FRONT_EPSILON,
    C5_FRONT_TAU,
    EXPERT_IDS,
    HEDGE_LOOKBACK,
    WEIGHT_FLOOR,
    build_c5_output,
    current_c5_prediction,
    hedge_weights,
    iter_hedge_predictions,
    normalized_expert_loss,
    walk_forward_c5_predictions,
    weighted_logpool,
)
from src.analysis.dlt_ridge_candidates_v1 import CandidateScores


@dataclass(frozen=True)
class Draw:
    front: tuple[int, ...]
    back: tuple[int, ...]


def _scores(front_peak: int, back_peak: int) -> CandidateScores:
    front = tuple(2.0 if index == front_peak - 1 else 0.0 for index in range(35))
    back = tuple(2.0 if index == back_peak - 1 else 0.0 for index in range(12))
    return CandidateScores(front, back)


def _experts() -> dict[str, CandidateScores]:
    return {
        EXPERT_IDS[0]: _scores(1, 1),
        EXPERT_IDS[1]: _scores(2, 2),
        EXPERT_IDS[2]: _scores(3, 3),
    }


def test_protocol_constants_are_frozen() -> None:
    assert HEDGE_LOOKBACK == 200
    assert WEIGHT_FLOOR == 0.10
    assert C5_FRONT_TAU == 1.0
    assert C5_FRONT_EPSILON == 0.10
    assert C5_BACK_TAU == 2.0
    assert C5_BACK_EPSILON == 0.20


def test_no_history_starts_equal_and_weights_sum_to_one() -> None:
    weights = hedge_weights([])
    assert weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert math.fsum(weights) == pytest.approx(1.0)


def test_lower_recent_loss_gets_more_weight_without_breaking_floor() -> None:
    rows = [(0.8, 1.0, 1.2)] * 25
    weights = hedge_weights(rows)
    assert weights[0] > weights[1] > weights[2]
    assert min(weights) >= WEIGHT_FLOOR
    assert math.fsum(weights) == pytest.approx(1.0)


def test_only_latest_200_losses_are_used() -> None:
    old = [(0.0, 10.0, 10.0)] * 10
    recent = [(1.0, 0.8, 1.2)] * HEDGE_LOOKBACK
    assert hedge_weights(old + recent) == pytest.approx(hedge_weights(recent))


def test_uniform_scores_have_normalized_loss_one() -> None:
    uniform = CandidateScores((0.0,) * 35, (0.0,) * 12)
    loss = normalized_expert_loss(uniform, (1, 2, 3, 4, 5), (1, 2))
    assert loss == pytest.approx(1.0)


def test_weighted_logpool_is_componentwise_and_validates_weights() -> None:
    experts = _experts()
    scores = weighted_logpool(experts, (0.5, 0.3, 0.2))
    assert scores.front[0] == pytest.approx(1.0)
    assert scores.front[1] == pytest.approx(0.6)
    assert scores.front[2] == pytest.approx(0.4)
    assert scores.back[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="权重"):
        weighted_logpool(experts, (0.5, 0.5, 0.5))


def test_current_prediction_never_uses_current_or_future_outcome() -> None:
    score_rows = [_experts(), _experts(), _experts()]
    draws_a = [
        Draw((1, 4, 5, 6, 7), (1, 4)),
        Draw((2, 8, 9, 10, 11), (2, 5)),
        Draw((3, 12, 13, 14, 15), (3, 6)),
    ]
    draws_b = [
        draws_a[0],
        Draw((30, 31, 32, 33, 34), (10, 11)),
        Draw((20, 21, 22, 23, 24), (7, 8)),
    ]
    predictions_a = tuple(iter_hedge_predictions(score_rows, draws_a))
    predictions_b = tuple(iter_hedge_predictions(score_rows, draws_b))
    assert predictions_a[0] == predictions_b[0]
    assert predictions_a[1] == predictions_b[1]
    assert predictions_a[2].weights != pytest.approx(predictions_b[2].weights)


def test_output_is_fixed_7plus2_with_21_unique_tickets() -> None:
    output = build_c5_output(weighted_logpool(_experts(), (1 / 3,) * 3))
    assert len(output.front) == 7
    assert len(output.back) == 2
    assert len(output.tickets) == 21
    assert len(set(output.tickets)) == 21
    assert all(
        len(front) == 5 and back == output.back for front, back in output.tickets
    )


def test_walk_forward_refits_only_on_global_25_boundaries() -> None:
    draws = [Draw((1, 2, 3, 4, 5), (1, 2)) for _ in range(626)]
    calls: list[tuple[int, str]] = []

    def fitter(_cache: object, cutoff: int, expert_id: str) -> CandidateScores:
        calls.append((cutoff, expert_id))
        return _experts()[expert_id]

    predictions = walk_forward_c5_predictions(
        draws, start_index=600, stop_index=626, score_fitter=fitter
    )
    assert len(predictions) == 26
    assert len(predictions[0].expert_scores) == 3
    assert {item.fit_cutoff for item in predictions} == {600, 625}
    assert calls == [
        (cutoff, expert_id) for cutoff in (600, 625) for expert_id in EXPERT_IDS
    ]


def test_current_prediction_uses_only_settled_rows_and_global_cutoff() -> None:
    draws = [Draw((1, 2, 3, 4, 5), (1, 2)) for _ in range(626)]

    def fitter(_cache: object, cutoff: int, expert_id: str) -> CandidateScores:
        return _experts()[expert_id]

    prediction = current_c5_prediction(draws, score_fitter=fitter)
    assert prediction.target_index == 626
    assert prediction.fit_cutoff == 625
    assert math.fsum(prediction.weights) == pytest.approx(1.0)
