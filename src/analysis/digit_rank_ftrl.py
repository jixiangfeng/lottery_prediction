# -*- coding: utf-8 -*-
"""Rank-aware FTRL稀疏v4.1探索挑战模型。"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import learned_ranker_source_fingerprint
from src.lotteries.base import LotteryRule

_CANDIDATES = np.array([f"{value:03d}" for value in range(1000)])
_GROUP_KEYS = tuple(sorted({"".join(sorted(value)) for value in _CANDIDATES}))
_GROUP_MEMBERS = tuple(
    np.asarray(
        [
            index
            for index, value in enumerate(_CANDIDATES)
            if "".join(sorted(value)) == key
        ],
        dtype=int,
    )
    for key in _GROUP_KEYS
)
_GROUP_INDEX = {key: index for index, key in enumerate(_GROUP_KEYS)}
_GROUP_PRIOR = np.asarray([len(indices) / 1000.0 for indices in _GROUP_MEMBERS])
_UNIFORM_LOG_LOSS = math.log(1000.0)
_UNIFORM_BRIER = 0.999


def group_key(candidate: str) -> str:
    if len(candidate) != 3 or not candidate.isdigit():
        raise ValueError("组选键需要三位数字")
    return "".join(sorted(candidate))


def group_multiplicity(key: str) -> int:
    normalized = group_key(key)
    return len(set(itertools.permutations(normalized)))


def poisson_binomial_upper_tail(
    probabilities: list[float], observed_hits: int
) -> float:
    if observed_hits <= 0:
        return 1.0
    if observed_hits > len(probabilities):
        return 0.0
    distribution = np.zeros(observed_hits, dtype=float)
    distribution[0] = 1.0
    for index, probability in enumerate(probabilities):
        upper = min(index + 1, observed_hits - 1)
        for hits in range(upper, 0, -1):
            distribution[hits] = (
                distribution[hits] * (1.0 - probability)
                + distribution[hits - 1] * probability
            )
        distribution[0] *= 1.0 - probability
    return float(max(0.0, 1.0 - distribution.sum()))


@dataclass
class FTRLState:
    z: np.ndarray
    n: np.ndarray

    @classmethod
    def zeros(cls, feature_count: int) -> "FTRLState":
        return cls(
            np.zeros(feature_count, dtype=float), np.zeros(feature_count, dtype=float)
        )


@dataclass(frozen=True)
class FTRLConfig:
    warmup_history: int = 150
    calibration_lookback: int = 400
    block_size: int = 500
    top_k: int = 50
    group_top_k: int = 10
    expert_alphas: tuple[float, ...] = (0.01, 0.03, 0.05)
    beta: float = 1.0
    l1: float = 0.01
    base_l2: float = 0.01
    logloss_weight: float = 0.30
    rank_weight: float = 0.70
    boundary_start: int = 45
    boundary_end: int = 60
    hedge_eta: float = 0.5
    maximum_expert_weight: float = 0.35
    maximum_uniform_shrinkage: float = 0.35
    minimum_confident_shrinkage: float = 0.05
    maximum_top50_triples: int = 1
    gradient_clip: float = 5.0
    excluded_features: tuple[str, ...] = ()
    feature_config: LearnedFeatureConfig = LearnedFeatureConfig(
        windows=(20, 50, 150, 300, 500),
        window_weights=(
            ("20", 2.5),
            ("50", 1.5),
            ("150", 0.75),
            ("300", 0.35),
            ("500", 0.15),
        ),
    )

    def __post_init__(self) -> None:
        if len(self.expert_alphas) < 3:
            raise ValueError("FTRL至少需要三个专家")
        if self.maximum_expert_weight < 1.0 / len(self.expert_alphas):
            raise ValueError("专家权重上限过低，无法归一化")
        if not math.isclose(self.logloss_weight + self.rank_weight, 1.0):
            raise ValueError("双目标权重之和必须为1")
        if self.boundary_start < 1 or self.boundary_end <= self.boundary_start:
            raise ValueError("Top50边界区间无效")
        unknown = set(self.excluded_features) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"未知排除特征: {sorted(unknown)}")


@dataclass(frozen=True)
class RankFTRLBlockResult:
    lottery: str
    config: FTRLConfig
    blocks: tuple[dict[str, Any], ...]
    feature_attribution: dict[str, dict[str, Any]]
    rank_buckets: dict[str, int]
    next_prediction: dict[str, Any]
    group_evaluation: dict[str, Any]
    data_sha256: str
    source_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        periods = sum(int(block["periods"]) for block in self.blocks)
        hits = sum(int(block["researchTop50Hits"]) for block in self.blocks)
        mean_log_loss = float(
            np.average(
                [float(block["deployedMeanLogLoss"]) for block in self.blocks],
                weights=[int(block["periods"]) for block in self.blocks],
            )
        )
        mean_brier = float(
            np.average(
                [float(block["deployedMeanBrier"]) for block in self.blocks],
                weights=[int(block["periods"]) for block in self.blocks],
            )
        )
        blocks_above = sum(
            float(block["researchTop50HitRate"]) >= self.config.top_k / 1000.0
            for block in self.blocks
        )
        pooled_p = float(binom.sf(hits - 1, periods, self.config.top_k / 1000.0))
        joint_blocks = sum(bool(block["jointGatePassed"]) for block in self.blocks)
        stable_required = math.ceil(len(self.blocks) * 0.70)
        passed = (
            hits / periods > self.config.top_k / 1000.0
            and pooled_p < 0.05
            and mean_log_loss < _UNIFORM_LOG_LOSS
            and mean_brier < _UNIFORM_BRIER
            and blocks_above >= stable_required
            and joint_blocks >= 2
        )
        admission_reasons: list[str] = []
        if not passed:
            admission_reasons.append("historical_gate_failed")
        if (
            float(self.next_prediction["dynamicShrinkage"])
            < self.config.minimum_confident_shrinkage
        ):
            admission_reasons.append("shrinkage_below_minimum")
        shape_health = self.next_prediction["shapeHealth"]
        if not bool(shape_health["passed"]):
            admission_reasons.extend(shape_health["reasons"])
        next_prediction = dict(self.next_prediction)
        next_prediction.update(
            {
                "abstained": bool(admission_reasons),
                "admissionReasons": admission_reasons,
                "userVisibleCandidates": (
                    []
                    if admission_reasons
                    else list(self.next_prediction["researchTop50"])
                ),
            }
        )
        return {
            "modelVersion": "rank_aware_ftrl_v4_1",
            "evaluationKind": "rank_aware_ftrl_blocks",
            "evidenceStatus": "exploratory_post_failure_redesign",
            "lottery": self.lottery,
            "selectionPolicy": "online_hedge_prior_only",
            "blockSelectionAllowed": False,
            "topK": self.config.top_k,
            "blocksEvaluated": len(self.blocks),
            "periodsEvaluated": periods,
            "config": {
                "warmupHistory": self.config.warmup_history,
                "calibrationLookback": self.config.calibration_lookback,
                "expertAlphas": list(self.config.expert_alphas),
                "lossWeights": {
                    "logLoss": self.config.logloss_weight,
                    "top50BoundaryRank": self.config.rank_weight,
                },
                "boundaryRanks": [self.config.boundary_start, self.config.boundary_end],
                "maximumExpertWeight": self.config.maximum_expert_weight,
                "maximumUniformShrinkage": self.config.maximum_uniform_shrinkage,
                "minimumConfidentShrinkage": self.config.minimum_confident_shrinkage,
                "maximumTop50Triples": self.config.maximum_top50_triples,
                "groupTopK": self.config.group_top_k,
                "featureWindows": list(self.config.feature_config.windows),
                "excludedFeatures": list(self.config.excluded_features),
            },
            "featureNames": list(FEATURE_NAMES),
            "featureAttribution": self.feature_attribution,
            "rankBuckets": self.rank_buckets,
            "nextPrediction": next_prediction,
            "groupEvaluation": self.group_evaluation,
            "dataSha256": self.data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "summary": {
                "researchTop50Hits": hits,
                "researchTop50HitRate": hits / periods,
                "researchTop50PValue": pooled_p,
                "deployedMeanLogLoss": mean_log_loss,
                "uniformLogLoss": _UNIFORM_LOG_LOSS,
                "deployedMeanBrier": mean_brier,
                "uniformBrier": _UNIFORM_BRIER,
                "blocksAtOrAboveBaseline": blocks_above,
                "stableBlocksRequired": stable_required,
                "jointGateBlocks": joint_blocks,
            },
            "gatePassed": passed,
            "formalPredictionActivated": False,
            "blocks": list(self.blocks),
        }


def ftrl_weights(
    state: FTRLState,
    alpha: float,
    beta: float,
    l1: float,
    l2: np.ndarray,
) -> np.ndarray:
    absolute = np.abs(state.z)
    denominator = (beta + np.sqrt(state.n)) / alpha + l2
    weights = np.zeros_like(state.z)
    active = absolute > l1
    weights[active] = (
        -(state.z[active] - np.sign(state.z[active]) * l1) / denominator[active]
    )
    return weights


def ftrl_update(
    state: FTRLState,
    gradient: np.ndarray,
    *,
    alpha: float,
    beta: float,
    l1: float,
    l2: np.ndarray,
) -> FTRLState:
    weights = ftrl_weights(state, alpha, beta, l1, l2)
    new_n = state.n + gradient * gradient
    sigma = (np.sqrt(new_n) - np.sqrt(state.n)) / alpha
    return FTRLState(state.z + gradient - sigma * weights, new_n)


def rank_boundary_gradient(
    matrix: np.ndarray,
    weights: np.ndarray,
    *,
    actual_index: int,
    boundary_indices: np.ndarray,
) -> np.ndarray:
    if len(boundary_indices) == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    actual_score = float(matrix[actual_index] @ weights)
    boundary_scores = matrix[boundary_indices] @ weights
    logits = np.clip(boundary_scores - actual_score, -30.0, 30.0)
    factors = 1.0 / (1.0 + np.exp(-logits))
    differences = matrix[boundary_indices] - matrix[actual_index]
    return np.mean(factors[:, None] * differences, axis=0)


def weighted_boundary_contributions(
    matrix: np.ndarray,
    expert_weight_vectors: tuple[np.ndarray, ...] | list[np.ndarray],
    hedge_weights: np.ndarray,
    *,
    actual_index: int,
    boundary_index: int,
) -> np.ndarray:
    feature_difference = matrix[actual_index] - matrix[boundary_index]
    blended_weights = np.average(
        np.vstack(expert_weight_vectors), axis=0, weights=hedge_weights
    )
    return blended_weights * feature_difference


def candidate_shape_health(
    candidates: tuple[str, ...] | list[str], *, maximum_triples: int
) -> dict[str, Any]:
    counts = {"豹子": 0, "组三": 0, "组六": 0}
    for candidate in candidates:
        unique = len(set(candidate))
        name = "豹子" if unique == 1 else ("组三" if unique == 2 else "组六")
        counts[name] += 1
    reasons: list[str] = []
    if counts["豹子"] > maximum_triples:
        reasons.append("triple_concentration")
    return {"passed": not reasons, "counts": counts, "reasons": reasons}


def cap_expert_weights(weights: np.ndarray, *, maximum: float) -> np.ndarray:
    values = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if values.sum() <= 0:
        values = np.ones_like(values)
    values /= values.sum()
    fixed = np.zeros_like(values, dtype=bool)
    for _ in range(len(values)):
        over = (~fixed) & (values > maximum)
        if not np.any(over):
            break
        values[over] = maximum
        fixed |= over
        remaining = 1.0 - float(values[fixed].sum())
        free = ~fixed
        if not np.any(free):
            break
        base = values[free]
        values[free] = remaining * base / float(base.sum())
    return values / values.sum()


def _l2_vector(config: FTRLConfig) -> np.ndarray:
    multipliers = np.ones(len(FEATURE_NAMES), dtype=float)
    multipliers[FEATURE_NAMES.index("position_frequency")] = 10.0
    for name in (
        "sum_distribution",
        "pair_frequency",
        "pair_trend",
        "position_trend",
        "recent_trend",
    ):
        multipliers[FEATURE_NAMES.index(name)] = 5.0
    return config.base_l2 * multipliers


def _apply_weight_constraints(weights: np.ndarray, config: FTRLConfig) -> np.ndarray:
    constrained = weights.copy()
    for name in (
        "shape_transition",
        "shape_recent_deviation",
        *config.excluded_features,
    ):
        constrained[FEATURE_NAMES.index(name)] = 0.0
    index = FEATURE_NAMES.index("constraint_penalty")
    constrained[index] = min(0.0, constrained[index])
    return constrained


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(scores.max())
    probabilities = np.exp(shifted)
    return probabilities / float(probabilities.sum())


def _hedge_weights(
    loss_histories: tuple[deque[float], ...], config: FTRLConfig
) -> np.ndarray:
    totals = np.array([sum(history) for history in loss_histories], dtype=float)
    totals -= float(totals.min())
    raw = np.exp(-config.hedge_eta * np.clip(totals, 0.0, 50.0))
    return cap_expert_weights(raw, maximum=config.maximum_expert_weight)


def _dynamic_shrinkage(blended_losses: deque[float], config: FTRLConfig) -> float:
    if len(blended_losses) < config.calibration_lookback:
        return 0.0
    improvement = _UNIFORM_LOG_LOSS - float(np.mean(blended_losses))
    return float(
        np.clip(
            improvement / 0.01 * config.maximum_uniform_shrinkage,
            0.0,
            config.maximum_uniform_shrinkage,
        )
    )


def run_rank_ftrl_blocks(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: FTRLConfig = FTRLConfig(),
) -> RankFTRLBlockResult:
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    first_block_start = config.warmup_history + config.calibration_lookback
    block_count = (len(chronological) - first_block_start) // config.block_size
    if block_count <= 0:
        raise ValueError("历史不足以形成FTRL区块")
    states = tuple(FTRLState.zeros(len(FEATURE_NAMES)) for _ in config.expert_alphas)
    loss_histories: tuple[deque[float], ...] = tuple(
        deque(maxlen=config.calibration_lookback) for _ in config.expert_alphas
    )
    blended_losses: deque[float] = deque(maxlen=config.calibration_lookback)
    l2 = _l2_vector(config)
    blocks: list[dict[str, Any]] = [
        {
            "blockIndex": index,
            "startIndex": first_block_start + index * config.block_size,
            "endIndex": first_block_start + (index + 1) * config.block_size,
            "periods": 0,
            "researchHits": 0,
            "researchLogLoss": [],
            "deployedLogLoss": [],
            "deployedBrier": [],
            "shrinkages": [],
            "expertWeights": [],
            "projectedGroupHits": 0,
            "projectedGroupBaselines": [],
            "projectedGroupCounts": [],
            "independentGroupHits": 0,
            "independentGroupBaselines": [],
            "independentGroupLogLoss": [],
            "independentGroupBrier": [],
            "groupPriorLogLoss": [],
            "groupPriorBrier": [],
        }
        for index in range(block_count)
    ]
    attribution_raw: dict[str, dict[str, float]] = {
        name: {
            "boundarySum": 0.0,
            "boundaryCount": 0.0,
            "addedHits": 0.0,
            "lostHits": 0.0,
            "ablatedHits": 0.0,
            "ablatedLogLossSum": 0.0,
            "ablatedBrierSum": 0.0,
        }
        for name in FEATURE_NAMES
    }
    rank_buckets = {
        "1-50": 0,
        "51-100": 0,
        "101-300": 0,
        "301-500": 0,
        "501-1000": 0,
    }
    rolling_states = iter_rolling_history_states(
        chronological,
        rule,
        range(config.warmup_history, len(chronological)),
        config.feature_config,
    )
    mutable_states = list(states)
    candidate_indices = np.arange(1000)
    for target_index, history_state in zip(
        range(config.warmup_history, len(chronological)), rolling_states
    ):
        matrix = build_candidate_features(history_state, rule)[
            list(FEATURE_NAMES)
        ].to_numpy(dtype=float)
        expert_weights_vectors: list[np.ndarray] = []
        expert_scores: list[np.ndarray] = []
        expert_probabilities: list[np.ndarray] = []
        for state, alpha in zip(mutable_states, config.expert_alphas):
            weights = _apply_weight_constraints(
                ftrl_weights(state, alpha, config.beta, config.l1, l2), config
            )
            expert_weights_vectors.append(weights)
            scores = matrix @ weights
            expert_scores.append(scores)
            expert_probabilities.append(_softmax(scores))
        hedge = _hedge_weights(loss_histories, config)
        blended = np.average(np.vstack(expert_probabilities), axis=0, weights=hedge)
        shrinkage = _dynamic_shrinkage(blended_losses, config)
        deployed = shrinkage * blended + (1.0 - shrinkage) / 1000.0
        row = chronological.iloc[target_index]
        actual_index = int(
            "".join(str(int(row[column])) for column in rule.number_columns)
        )
        actual_research_probability = float(blended[actual_index])
        actual_deployed_probability = float(deployed[actual_index])
        research_log_loss = -math.log(max(actual_research_probability, 1e-15))
        deployed_log_loss = -math.log(max(actual_deployed_probability, 1e-15))
        deployed_brier = (
            float(np.sum(deployed * deployed)) + 1.0 - 2.0 * actual_deployed_probability
        )
        actual_value = blended[actual_index]
        rank = (
            1
            + int(np.sum(blended > actual_value))
            + int(
                np.sum((blended == actual_value) & (candidate_indices < actual_index))
            )
        )
        if target_index >= first_block_start:
            block_index = (target_index - first_block_start) // config.block_size
            if block_index < block_count:
                block = blocks[block_index]
                if rank <= 50:
                    rank_buckets["1-50"] += 1
                elif rank <= 100:
                    rank_buckets["51-100"] += 1
                elif rank <= 300:
                    rank_buckets["101-300"] += 1
                elif rank <= 500:
                    rank_buckets["301-500"] += 1
                else:
                    rank_buckets["501-1000"] += 1
                base_hit = rank <= config.top_k
                base_order = np.argsort(-blended, kind="mergesort")
                actual_group_key = group_key(_CANDIDATES[actual_index])
                actual_group_index = _GROUP_INDEX[actual_group_key]
                projected_groups = {
                    group_key(_CANDIDATES[index])
                    for index in base_order[: config.top_k]
                }
                projected_baseline = (
                    sum(group_multiplicity(key) for key in projected_groups) / 1000.0
                )
                block["projectedGroupHits"] += actual_group_key in projected_groups
                block["projectedGroupBaselines"].append(projected_baseline)
                block["projectedGroupCounts"].append(len(projected_groups))
                group_research = np.asarray(
                    [float(blended[indices].sum()) for indices in _GROUP_MEMBERS]
                )
                group_deployed = np.asarray(
                    [float(deployed[indices].sum()) for indices in _GROUP_MEMBERS]
                )
                group_order = np.argsort(-group_research, kind="mergesort")
                selected_group_indices = group_order[: config.group_top_k]
                independent_baseline = float(_GROUP_PRIOR[selected_group_indices].sum())
                group_hit = actual_group_index in selected_group_indices
                group_probability = float(group_deployed[actual_group_index])
                prior_probability = float(_GROUP_PRIOR[actual_group_index])
                block["independentGroupHits"] += group_hit
                block["independentGroupBaselines"].append(independent_baseline)
                block["independentGroupLogLoss"].append(
                    -math.log(max(group_probability, 1e-15))
                )
                block["independentGroupBrier"].append(
                    float(np.sum(group_deployed * group_deployed))
                    + 1.0
                    - 2.0 * group_probability
                )
                block["groupPriorLogLoss"].append(
                    -math.log(max(prior_probability, 1e-15))
                )
                block["groupPriorBrier"].append(
                    float(np.sum(_GROUP_PRIOR * _GROUP_PRIOR))
                    + 1.0
                    - 2.0 * prior_probability
                )
                if not base_hit:
                    contributions = weighted_boundary_contributions(
                        matrix,
                        expert_weights_vectors,
                        hedge,
                        actual_index=actual_index,
                        boundary_index=int(base_order[config.top_k - 1]),
                    )
                    for feature_index, name in enumerate(FEATURE_NAMES):
                        attribution_raw[name]["boundarySum"] += float(
                            contributions[feature_index]
                        )
                        attribution_raw[name]["boundaryCount"] += 1.0
                for feature_index, name in enumerate(FEATURE_NAMES):
                    ablated_experts = [
                        _softmax(
                            scores - matrix[:, feature_index] * weights[feature_index]
                        )
                        for scores, weights in zip(
                            expert_scores, expert_weights_vectors
                        )
                    ]
                    ablated = np.average(
                        np.vstack(ablated_experts), axis=0, weights=hedge
                    )
                    ablated_actual = float(ablated[actual_index])
                    ablated_rank = (
                        1
                        + int(np.sum(ablated > ablated_actual))
                        + int(
                            np.sum(
                                (ablated == ablated_actual)
                                & (candidate_indices < actual_index)
                            )
                        )
                    )
                    ablated_hit = ablated_rank <= config.top_k
                    attribution_raw[name]["addedHits"] += float(
                        ablated_hit and not base_hit
                    )
                    attribution_raw[name]["lostHits"] += float(
                        base_hit and not ablated_hit
                    )
                    attribution_raw[name]["ablatedHits"] += float(ablated_hit)
                    ablated_deployed = shrinkage * ablated + (1.0 - shrinkage) / 1000.0
                    probability = float(ablated_deployed[actual_index])
                    attribution_raw[name]["ablatedLogLossSum"] += -math.log(
                        max(probability, 1e-15)
                    )
                    attribution_raw[name]["ablatedBrierSum"] += (
                        float(np.sum(ablated_deployed * ablated_deployed))
                        + 1.0
                        - 2.0 * probability
                    )
                block["periods"] += 1
                block["researchHits"] += rank <= config.top_k
                block["researchLogLoss"].append(research_log_loss)
                block["deployedLogLoss"].append(deployed_log_loss)
                block["deployedBrier"].append(deployed_brier)
                block["shrinkages"].append(shrinkage)
                block["expertWeights"].append(hedge.tolist())
        for expert_index, (weights, probabilities, alpha) in enumerate(
            zip(expert_weights_vectors, expert_probabilities, config.expert_alphas)
        ):
            log_gradient = matrix.T @ probabilities - matrix[actual_index]
            order = np.argsort(-(matrix @ weights), kind="mergesort")
            boundary = order[config.boundary_start - 1 : config.boundary_end]
            boundary = boundary[boundary != actual_index]
            rank_gradient = rank_boundary_gradient(
                matrix,
                weights,
                actual_index=actual_index,
                boundary_indices=boundary,
            )
            gradient = (
                config.logloss_weight * log_gradient
                + config.rank_weight * rank_gradient
            )
            for name in config.excluded_features:
                gradient[FEATURE_NAMES.index(name)] = 0.0
            norm = float(np.linalg.norm(gradient))
            if norm > config.gradient_clip:
                gradient *= config.gradient_clip / norm
            mutable_states[expert_index] = ftrl_update(
                mutable_states[expert_index],
                gradient,
                alpha=alpha,
                beta=config.beta,
                l1=config.l1,
                l2=l2,
            )
            loss_histories[expert_index].append(
                -math.log(max(float(probabilities[actual_index]), 1e-15))
            )
        blended_losses.append(research_log_loss)
    summaries: list[dict[str, Any]] = []
    for raw in blocks:
        periods = int(raw["periods"])
        if periods != config.block_size:
            raise RuntimeError("存在未完成FTRL区块")
        hits = int(raw["researchHits"])
        mean_log_loss = float(np.mean(raw["deployedLogLoss"]))
        mean_brier = float(np.mean(raw["deployedBrier"]))
        p_value = float(binom.sf(hits - 1, periods, config.top_k / 1000.0))
        projected_group_hits = int(raw["projectedGroupHits"])
        projected_group_p = poisson_binomial_upper_tail(
            list(raw["projectedGroupBaselines"]), projected_group_hits
        )
        independent_group_hits = int(raw["independentGroupHits"])
        independent_group_p = poisson_binomial_upper_tail(
            list(raw["independentGroupBaselines"]), independent_group_hits
        )
        summaries.append(
            {
                "blockIndex": raw["blockIndex"],
                "startIndex": raw["startIndex"],
                "endIndex": raw["endIndex"],
                "periods": periods,
                "researchTop50Hits": hits,
                "researchTop50HitRate": hits / periods,
                "researchTop50PValue": p_value,
                "researchMeanLogLoss": float(np.mean(raw["researchLogLoss"])),
                "deployedMeanLogLoss": mean_log_loss,
                "deployedMeanBrier": mean_brier,
                "uniformLogLoss": _UNIFORM_LOG_LOSS,
                "uniformBrier": _UNIFORM_BRIER,
                "meanShrinkage": float(np.mean(raw["shrinkages"])),
                "meanExpertWeights": np.mean(
                    np.asarray(raw["expertWeights"]), axis=0
                ).tolist(),
                "projectedGroupHits": projected_group_hits,
                "projectedGroupMeanBaseline": float(
                    np.mean(raw["projectedGroupBaselines"])
                ),
                "projectedGroupMeanCount": float(np.mean(raw["projectedGroupCounts"])),
                "projectedGroupPValue": projected_group_p,
                "independentGroupHits": independent_group_hits,
                "independentGroupMeanBaseline": float(
                    np.mean(raw["independentGroupBaselines"])
                ),
                "independentGroupPValue": independent_group_p,
                "independentGroupMeanLogLoss": float(
                    np.mean(raw["independentGroupLogLoss"])
                ),
                "independentGroupMeanBrier": float(
                    np.mean(raw["independentGroupBrier"])
                ),
                "groupPriorMeanLogLoss": float(np.mean(raw["groupPriorLogLoss"])),
                "groupPriorMeanBrier": float(np.mean(raw["groupPriorBrier"])),
                "jointGatePassed": (
                    mean_log_loss < _UNIFORM_LOG_LOSS
                    and mean_brier < _UNIFORM_BRIER
                    and p_value < 0.05
                ),
            }
        )
    total_periods = sum(int(item["periods"]) for item in summaries)
    base_hits = sum(int(item["researchTop50Hits"]) for item in summaries)
    base_log_loss = float(
        np.average(
            [float(item["deployedMeanLogLoss"]) for item in summaries],
            weights=[int(item["periods"]) for item in summaries],
        )
    )
    base_brier = float(
        np.average(
            [float(item["deployedMeanBrier"]) for item in summaries],
            weights=[int(item["periods"]) for item in summaries],
        )
    )
    feature_attribution: dict[str, dict[str, Any]] = {}
    for name, raw in attribution_raw.items():
        boundary_count = int(raw["boundaryCount"])
        ablated_log_loss = raw["ablatedLogLossSum"] / total_periods
        ablated_brier = raw["ablatedBrierSum"] / total_periods
        added = int(raw["addedHits"])
        lost = int(raw["lostHits"])
        feature_attribution[name] = {
            "meanMissBoundaryContribution": (
                raw["boundarySum"] / boundary_count if boundary_count else 0.0
            ),
            "missPeriods": boundary_count,
            "addedHitsWhenZeroed": added,
            "lostHitsWhenZeroed": lost,
            "netHitDeltaWhenZeroed": added - lost,
            "baseTop50Hits": base_hits,
            "ablatedTop50Hits": int(raw["ablatedHits"]),
            "ablatedMeanLogLoss": ablated_log_loss,
            "logLossDeltaWhenZeroed": ablated_log_loss - base_log_loss,
            "ablatedMeanBrier": ablated_brier,
            "brierDeltaWhenZeroed": ablated_brier - base_brier,
        }
    projected_baselines = [
        float(value) for raw in blocks for value in raw["projectedGroupBaselines"]
    ]
    independent_baselines = [
        float(value) for raw in blocks for value in raw["independentGroupBaselines"]
    ]
    projected_hits = sum(int(raw["projectedGroupHits"]) for raw in blocks)
    independent_hits = sum(int(raw["independentGroupHits"]) for raw in blocks)
    group_log_loss = float(
        np.mean([value for raw in blocks for value in raw["independentGroupLogLoss"]])
    )
    group_brier = float(
        np.mean([value for raw in blocks for value in raw["independentGroupBrier"]])
    )
    prior_log_loss = float(
        np.mean([value for raw in blocks for value in raw["groupPriorLogLoss"]])
    )
    prior_brier = float(
        np.mean([value for raw in blocks for value in raw["groupPriorBrier"]])
    )
    group_joint_blocks = sum(
        float(item["independentGroupPValue"]) < 0.05
        and float(item["independentGroupMeanLogLoss"])
        < float(item["groupPriorMeanLogLoss"])
        and float(item["independentGroupMeanBrier"])
        < float(item["groupPriorMeanBrier"])
        for item in summaries
    )
    group_evaluation: dict[str, Any] = {
        "baselineKind": "permutation_weighted_poisson_binomial",
        "independentTopK": config.group_top_k,
        "projectedFromDirect": {
            "periods": total_periods,
            "hits": projected_hits,
            "hitRate": projected_hits / total_periods,
            "meanSelectedGroupCount": float(
                np.mean(
                    [value for raw in blocks for value in raw["projectedGroupCounts"]]
                )
            ),
            "meanMatchedBaseline": float(np.mean(projected_baselines)),
            "pValue": poisson_binomial_upper_tail(projected_baselines, projected_hits),
        },
        "independent": {
            "periods": total_periods,
            "hits": independent_hits,
            "hitRate": independent_hits / total_periods,
            "meanMatchedBaseline": float(np.mean(independent_baselines)),
            "pValue": poisson_binomial_upper_tail(
                independent_baselines, independent_hits
            ),
            "meanLogLoss": group_log_loss,
            "priorLogLoss": prior_log_loss,
            "meanBrier": group_brier,
            "priorBrier": prior_brier,
            "jointGateBlocks": group_joint_blocks,
            "gatePassed": (
                poisson_binomial_upper_tail(independent_baselines, independent_hits)
                < 0.05
                and group_log_loss < prior_log_loss
                and group_brier < prior_brier
                and group_joint_blocks >= 2
            ),
        },
        "blocks": [
            {
                key: item[key]
                for key in (
                    "blockIndex",
                    "projectedGroupHits",
                    "projectedGroupMeanBaseline",
                    "projectedGroupMeanCount",
                    "projectedGroupPValue",
                    "independentGroupHits",
                    "independentGroupMeanBaseline",
                    "independentGroupPValue",
                    "independentGroupMeanLogLoss",
                    "independentGroupMeanBrier",
                    "groupPriorMeanLogLoss",
                    "groupPriorMeanBrier",
                )
            }
            for item in summaries
        ],
    }
    next_history_state = build_history_state(chronological, rule, config.feature_config)
    next_matrix = build_candidate_features(next_history_state, rule)[
        list(FEATURE_NAMES)
    ].to_numpy(dtype=float)
    next_expert_probabilities: list[np.ndarray] = []
    for state, alpha in zip(mutable_states, config.expert_alphas):
        weights = _apply_weight_constraints(
            ftrl_weights(state, alpha, config.beta, config.l1, l2), config
        )
        next_expert_probabilities.append(_softmax(next_matrix @ weights))
    next_hedge = _hedge_weights(loss_histories, config)
    next_blended = np.average(
        np.vstack(next_expert_probabilities), axis=0, weights=next_hedge
    )
    next_order = np.argsort(-next_blended, kind="mergesort")
    next_shrinkage = _dynamic_shrinkage(blended_losses, config)
    top50_candidates = [_CANDIDATES[index] for index in next_order[: config.top_k]]
    next_shape_health = candidate_shape_health(
        top50_candidates, maximum_triples=config.maximum_top50_triples
    )
    next_prediction: dict[str, Any] = {
        "historyEndIssue": str(chronological.iloc[-1]["期数"]),
        "targetIssue": None,
        "researchTop50": top50_candidates,
        "researchTop50ProbabilityMass": float(
            next_blended[next_order[: config.top_k]].sum()
        ),
        "dynamicShrinkage": next_shrinkage,
        "expertWeights": next_hedge.tolist(),
        "excludedFeatures": list(config.excluded_features),
        "shapeHealth": next_shape_health,
        "formalRecommendation": None,
        "shadowOnly": True,
    }
    data_hash = hashlib.sha256(
        chronological.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    return RankFTRLBlockResult(
        lottery=rule.code,
        config=config,
        blocks=tuple(summaries),
        feature_attribution=feature_attribution,
        rank_buckets=rank_buckets,
        next_prediction=next_prediction,
        group_evaluation=group_evaluation,
        data_sha256=data_hash,
        source_fingerprint=learned_ranker_source_fingerprint(),
    )


def write_rank_ftrl_report(result: RankFTRLBlockResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["reportSha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return destination


__all__ = [
    "FTRLConfig",
    "FTRLState",
    "RankFTRLBlockResult",
    "candidate_shape_health",
    "cap_expert_weights",
    "ftrl_update",
    "ftrl_weights",
    "group_key",
    "group_multiplicity",
    "poisson_binomial_upper_tail",
    "rank_boundary_gradient",
    "run_rank_ftrl_blocks",
    "weighted_boundary_contributions",
    "write_rank_ftrl_report",
]
