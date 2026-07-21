# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.digit_learned_features import LearnedFeatureConfig
from src.analysis.digit_rank_ftrl import (
    FTRLConfig,
    FTRLState,
    candidate_shape_health,
    cap_expert_weights,
    ftrl_update,
    ftrl_weights,
    group_key,
    group_multiplicity,
    poisson_binomial_upper_tail,
    rank_boundary_gradient,
    run_rank_ftrl_blocks,
    weighted_boundary_contributions,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    return pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(periods)],
            "百位": rng.integers(0, 10, periods),
            "十位": rng.integers(0, 10, periods),
            "个位": rng.integers(0, 10, periods),
        }
    )


def test_ftrl_keeps_small_gradient_sparse():
    state = FTRLState.zeros(3)
    updated = ftrl_update(
        state,
        np.array([0.1, -0.1, 0.05]),
        alpha=0.05,
        beta=1.0,
        l1=1.0,
        l2=np.ones(3),
    )
    assert np.array_equal(
        ftrl_weights(updated, 0.05, 1.0, 1.0, np.ones(3)), np.zeros(3)
    )


def test_rank_gradient_moves_actual_above_boundary():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    initial = FTRLState.zeros(2)
    gradient = rank_boundary_gradient(
        matrix, np.zeros(2), actual_index=0, boundary_indices=np.array([1])
    )
    updated = ftrl_update(
        initial, gradient, alpha=0.5, beta=1.0, l1=0.0, l2=np.ones(2) * 0.01
    )
    weights = ftrl_weights(updated, 0.5, 1.0, 0.0, np.ones(2) * 0.01)
    assert matrix[0] @ weights > matrix[1] @ weights


def test_expert_weight_cap_preserves_three_experts():
    capped = cap_expert_weights(np.array([0.9, 0.05, 0.05]), maximum=0.35)
    assert np.isclose(capped.sum(), 1.0)
    assert capped.max() <= 0.35 + 1e-12
    assert np.count_nonzero(capped) == 3


def test_weighted_boundary_contribution_is_exact_for_linear_experts():
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    expert_vectors = (np.array([1.0, 0.0]), np.array([0.0, 2.0]))
    contributions = weighted_boundary_contributions(
        matrix,
        expert_vectors,
        np.array([0.25, 0.75]),
        actual_index=0,
        boundary_index=1,
    )
    assert np.allclose(contributions, np.array([0.25, -3.0]))


def test_group_multiplicity_uses_permutation_weighted_baseline():
    assert group_key("321") == "123"
    assert group_multiplicity("777") == 1
    assert group_multiplicity("112") == 3
    assert group_multiplicity("123") == 6


def test_poisson_binomial_tail_matches_simple_two_trial_case():
    assert np.isclose(poisson_binomial_upper_tail([0.5, 0.5], 2), 0.25)
    assert np.isclose(poisson_binomial_upper_tail([0.5, 0.5], 1), 0.75)


def test_shape_health_rejects_triple_concentration():
    candidates = (
        tuple(f"{digit}{digit}{digit}" for digit in range(10))
        + tuple(
            value
            for value in (f"{number:03d}" for number in range(1000))
            if len(set(value)) > 1
        )[:40]
    )
    health = candidate_shape_health(candidates[:50], maximum_triples=1)
    assert health["passed"] is False
    assert health["counts"]["豹子"] == 10
    assert "triple_concentration" in health["reasons"]


def test_rank_ftrl_reports_all_complete_blocks():
    config = FTRLConfig(
        warmup_history=20,
        calibration_lookback=20,
        block_size=50,
        expert_alphas=(0.02, 0.05, 0.1),
        feature_config=LearnedFeatureConfig(
            windows=(5, 10, 20),
            window_weights=(("5", 2.0), ("10", 1.0), ("20", 0.5)),
        ),
    )
    payload = run_rank_ftrl_blocks(
        _history(190), get_lottery_rule("fc3d"), config
    ).to_dict()
    assert payload["evaluationKind"] == "rank_aware_ftrl_blocks"
    assert payload["blocksEvaluated"] == 3
    assert all(block["periods"] == 50 for block in payload["blocks"])
    assert payload["formalPredictionActivated"] is False
    assert payload["evidenceStatus"] == "exploratory_post_failure_redesign"
    assert set(payload["featureAttribution"]) == set(payload["featureNames"])
    assert sum(payload["rankBuckets"].values()) == payload["periodsEvaluated"]
    assert len(payload["nextPrediction"]["researchTop50"]) == 50
    assert payload["nextPrediction"]["formalRecommendation"] is None
    assert payload["nextPrediction"]["abstained"] is True
    assert payload["nextPrediction"]["userVisibleCandidates"] == []
    assert payload["groupEvaluation"]["independentTopK"] == 10
    assert (
        payload["groupEvaluation"]["projectedFromDirect"]["periods"]
        == payload["periodsEvaluated"]
    )
    assert (
        payload["groupEvaluation"]["independent"]["periods"]
        == payload["periodsEvaluated"]
    )
