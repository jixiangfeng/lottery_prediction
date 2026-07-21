# -*- coding: utf-8 -*-
"""learned_ranker_v4逐期特征归因与正则化在线梯度研究。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    BEHAVIORAL_FEATURE_NAMES,
    FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import (
    DEFAULT_WEIGHTS,
    rank_candidate_indices,
)
from src.analysis.digit_statistics import classify_digit_shape
from src.lotteries.base import LotteryRule

_UNIFORM_LOG_LOSS = math.log(1000.0)
_UNIFORM_BRIER = 0.999
_CANDIDATES = tuple(f"{value:03d}" for value in range(1000))
_SPARSE_L2_MULTIPLIERS = (
    ("position_frequency", 10.0),
    ("sum_distribution", 5.0),
    ("pair_frequency", 5.0),
    ("pair_trend", 5.0),
    ("position_trend", 5.0),
    ("recent_trend", 5.0),
)
_SPARSE_ZERO_FEATURES = ("shape_transition", "shape_recent_deviation")


@dataclass(frozen=True)
class OnlineGradientConfig:
    development_end: int
    outer_periods: int = 500
    calibration_interval: int = 10
    search_lookback: int = 300
    validation_lookback: int = 100
    warmup_history: int = 150
    learning_rates: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
    shrinkages: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    temperature: float = 1.0
    l2_penalty: float = 0.001
    gradient_clip: float = 1.0
    weight_limit: float = 1.5
    direct_top_k: int = 50
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_l2_multipliers: tuple[tuple[str, float], ...] = _SPARSE_L2_MULTIPLIERS
    zeroed_features: tuple[str, ...] = _SPARSE_ZERO_FEATURES

    def __post_init__(self) -> None:
        if self.development_end <= 0 or self.outer_periods <= 0:
            raise ValueError("development_end和outer_periods必须为正")
        if self.calibration_interval <= 0:
            raise ValueError("calibration_interval必须为正")
        if self.search_lookback <= 0 or self.validation_lookback <= 0:
            raise ValueError("Search/Validation回看必须为正")
        if self.warmup_history < 20:
            raise ValueError("warmup_history至少为20")
        if not self.learning_rates or any(value < 0 for value in self.learning_rates):
            raise ValueError("learning_rates必须是非空非负数组")
        if not self.shrinkages or any(not 0 <= value <= 1 for value in self.shrinkages):
            raise ValueError("shrinkages必须位于0..1")
        if not any(value > 0 for value in self.shrinkages):
            raise ValueError("shrinkages至少包含一个λ>0模型候选")
        if self.temperature <= 0 or self.gradient_clip <= 0 or self.weight_limit <= 0:
            raise ValueError("temperature、gradient_clip和weight_limit必须为正")
        if self.l2_penalty < 0:
            raise ValueError("l2_penalty不得为负")
        if not 1 <= self.direct_top_k <= 1000:
            raise ValueError("direct_top_k必须位于1..1000")
        available_features = set(FEATURE_NAMES) | set(BEHAVIORAL_FEATURE_NAMES)
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature_names必须非空且不得重复")
        if any(name not in available_features for name in self.feature_names):
            raise ValueError("feature_names包含未知特征")
        multiplier_names = [name for name, _ in self.feature_l2_multipliers]
        if len(set(multiplier_names)) != len(multiplier_names):
            raise ValueError("feature_l2_multipliers特征不得重复")
        if any(name not in self.feature_names for name in multiplier_names):
            raise ValueError("feature_l2_multipliers包含未知特征")
        if any(value <= 0 for _, value in self.feature_l2_multipliers):
            raise ValueError("feature_l2_multipliers倍率必须为正")
        if len(set(self.zeroed_features)) != len(self.zeroed_features):
            raise ValueError("zeroed_features特征不得重复")
        if any(name not in self.feature_names for name in self.zeroed_features):
            raise ValueError("zeroed_features包含未知特征")


@dataclass(frozen=True)
class OnlineGradientCandidate:
    learning_rate: float
    uniform_shrinkage: float

    @property
    def key(self) -> str:
        return f"eta={self.learning_rate:g}|lambda={self.uniform_shrinkage:g}"


@dataclass(frozen=True)
class OnlineGradientSelection:
    block_start_index: int
    candidate: OnlineGradientCandidate
    search_mean_log_loss: float
    validation_mean_log_loss: float
    validation_mean_brier: float
    stable_blocks: int
    abstained: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "blockStartIndex": self.block_start_index,
            "candidate": {
                "key": self.candidate.key,
                "learningRate": self.candidate.learning_rate,
                "uniformShrinkage": self.candidate.uniform_shrinkage,
            },
            "searchMeanLogLoss": self.search_mean_log_loss,
            "validationMeanLogLoss": self.validation_mean_log_loss,
            "validationMeanBrier": self.validation_mean_brier,
            "stableBlocks": self.stable_blocks,
            "abstained": self.abstained,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OnlineGradientPeriod:
    target_index: int
    target_issue: str
    history_end_issue: str
    candidate_key: str
    learning_rate: float
    uniform_shrinkage: float
    abstained: bool
    actual_text: str
    research_actual_probability: float
    deployed_actual_probability: float
    research_rank: int
    research_direct_hit: bool
    research_log_loss: float
    deployed_log_loss: float
    research_brier: float
    deployed_brier: float
    gradient_norm: float
    gradients: dict[str, float]
    boundary_contributions: dict[str, float]
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    top50_shape_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "targetIndex": self.target_index,
            "targetIssue": self.target_issue,
            "historyEndIssue": self.history_end_issue,
            "candidateKey": self.candidate_key,
            "learningRate": self.learning_rate,
            "uniformShrinkage": self.uniform_shrinkage,
            "abstained": self.abstained,
            "actual": self.actual_text,
            "researchActualProbability": self.research_actual_probability,
            "deployedActualProbability": self.deployed_actual_probability,
            "researchRank": self.research_rank,
            "researchDirectHit": self.research_direct_hit,
            "researchLogLoss": self.research_log_loss,
            "deployedLogLoss": self.deployed_log_loss,
            "researchBrier": self.research_brier,
            "deployedBrier": self.deployed_brier,
            "gradientNorm": self.gradient_norm,
            "gradients": self.gradients,
            "boundaryContributions": self.boundary_contributions,
            "weightsBefore": self.weights_before,
            "weightsAfter": self.weights_after,
            "top50ShapeCounts": self.top50_shape_counts,
        }


@dataclass(frozen=True)
class OnlineGradientReport:
    lottery: str
    outer_start_index: int
    outer_end_index: int
    frozen_test_read: bool
    evidence_status: str
    config: OnlineGradientConfig
    selections: tuple[OnlineGradientSelection, ...]
    periods: tuple[OnlineGradientPeriod, ...]

    def to_dict(self) -> dict[str, object]:
        active = [item for item in self.periods if not item.abstained]
        research_log = float(np.mean([item.research_log_loss for item in self.periods]))
        deployed_log = float(np.mean([item.deployed_log_loss for item in self.periods]))
        research_brier = float(np.mean([item.research_brier for item in self.periods]))
        deployed_brier = float(np.mean([item.deployed_brier for item in self.periods]))
        top50_hits = sum(item.research_direct_hit for item in self.periods)
        top50_p_value = float(
            binom.sf(
                top50_hits - 1,
                len(self.periods),
                self.config.direct_top_k / 1000,
            )
        )
        block_log_losses = tuple(
            float(block.mean())
            for block in np.array_split(
                np.asarray([item.research_log_loss for item in self.periods]), 3
            )
            if len(block)
        )
        stable_blocks = sum(value < _UNIFORM_LOG_LOSS for value in block_log_losses)
        frozen_reasons = []
        if research_log >= _UNIFORM_LOG_LOSS:
            frozen_reasons.append("Frozen LogLoss未优于均匀")
        if research_brier >= _UNIFORM_BRIER:
            frozen_reasons.append("Frozen Brier未优于均匀")
        if top50_p_value >= 0.05:
            frozen_reasons.append("Frozen Top50未达到p<0.05")
        if stable_blocks < 2:
            frozen_reasons.append("Frozen稳定时间块不足2/3")
        harmful = {
            name: {
                "negativeBoundaryPeriods": sum(
                    item.boundary_contributions[name] < 0 for item in self.periods
                ),
                "meanBoundaryContribution": float(
                    np.mean(
                        [item.boundary_contributions[name] for item in self.periods]
                    )
                ),
                "meanGradient": float(
                    np.mean([item.gradients[name] for item in self.periods])
                ),
            }
            for name in self.config.feature_names
        }
        return {
            "modelVersion": "learned_ranker_v4",
            "evaluationKind": (
                "frozen_sparse_online_gradient"
                if self.frozen_test_read
                else "development_sparse_online_gradient"
            ),
            "evidenceStatus": self.evidence_status,
            "lottery": self.lottery,
            "outerStartIndex": self.outer_start_index,
            "outerEndIndex": self.outer_end_index,
            "frozenTestRead": self.frozen_test_read,
            "config": {
                "outerPeriods": self.config.outer_periods,
                "calibrationInterval": self.config.calibration_interval,
                "searchLookback": self.config.search_lookback,
                "validationLookback": self.config.validation_lookback,
                "learningRates": list(self.config.learning_rates),
                "shrinkages": list(self.config.shrinkages),
                "temperature": self.config.temperature,
                "l2Penalty": self.config.l2_penalty,
                "gradientClip": self.config.gradient_clip,
                "weightLimit": self.config.weight_limit,
                "directTopK": self.config.direct_top_k,
                "featureNames": list(self.config.feature_names),
                "featureL2Multipliers": dict(self.config.feature_l2_multipliers),
                "zeroedFeatures": list(self.config.zeroed_features),
            },
            "metrics": {
                "periods": len(self.periods),
                "activePeriods": len(active),
                "abstainedPeriods": len(self.periods) - len(active),
                "abstentionRate": 1.0 - len(active) / len(self.periods),
                "researchMeanLogLoss": research_log,
                "deployedMeanLogLoss": deployed_log,
                "uniformLogLoss": _UNIFORM_LOG_LOSS,
                "researchMeanBrier": research_brier,
                "deployedMeanBrier": deployed_brier,
                "uniformBrier": _UNIFORM_BRIER,
                "researchTop50HitRate": float(
                    np.mean([item.research_direct_hit for item in self.periods])
                ),
                "researchTop50Hits": top50_hits,
                "researchTop50PValue": top50_p_value,
                "activeTop50HitRate": (
                    float(np.mean([item.research_direct_hit for item in active]))
                    if active
                    else None
                ),
            },
            "frozenGate": (
                {
                    "passed": not frozen_reasons,
                    "blockMeanLogLoss": list(block_log_losses),
                    "stableBlocks": stable_blocks,
                    "requiredStableBlocks": 2,
                    "top50PValueThreshold": 0.05,
                    "reasons": frozen_reasons,
                }
                if self.frozen_test_read
                else None
            ),
            "featureAttribution": harmful,
            "selections": [item.to_dict() for item in self.selections],
            "periods": [item.to_dict() for item in self.periods],
        }


@dataclass
class _CandidateState:
    candidate: OnlineGradientCandidate
    weights: np.ndarray
    log_losses: list[float]
    brier_scores: list[float]


@dataclass(frozen=True)
class _Step:
    model_probabilities: np.ndarray
    final_probabilities: np.ndarray
    gradient: np.ndarray
    clipped_gradient: np.ndarray
    weights_after: np.ndarray
    log_loss: float
    brier: float


def _regularization_multipliers(config: OnlineGradientConfig) -> np.ndarray:
    mapping = dict(config.feature_l2_multipliers)
    return np.asarray(
        [mapping.get(name, 1.0) for name in config.feature_names], dtype=float
    )


def _initial_weights(config: OnlineGradientConfig) -> np.ndarray:
    weights = np.asarray(
        [float(DEFAULT_WEIGHTS.get(name, 0.0)) for name in config.feature_names],
        dtype=float,
    )
    for name in config.zeroed_features:
        weights[config.feature_names.index(name)] = 0.0
    return weights


def online_gradient_step(
    feature_matrix: np.ndarray,
    actual_index: int,
    weights: np.ndarray,
    candidate: OnlineGradientCandidate,
    config: OnlineGradientConfig,
) -> _Step:
    """先给出事前概率，再使用真实结果更新下一期权重。"""

    matrix = np.asarray(feature_matrix, dtype=float)
    scores = matrix @ weights / config.temperature
    shifted = scores - float(scores.max())
    model = np.exp(shifted)
    model /= float(model.sum())
    uniform = np.full(len(model), 1.0 / len(model))
    final = (
        candidate.uniform_shrinkage * model
        + (1.0 - candidate.uniform_shrinkage) * uniform
    )
    actual_probability = float(final[actual_index])
    factor = (
        candidate.uniform_shrinkage
        * float(model[actual_index])
        / max(actual_probability, 1e-15)
    )
    expected_features = model @ matrix
    gradient = factor * (expected_features - matrix[actual_index]) / config.temperature
    norm = float(np.linalg.norm(gradient))
    clipped = gradient * min(1.0, config.gradient_clip / max(norm, 1e-15))
    regularization = _regularization_multipliers(config)
    updated = weights - candidate.learning_rate * (
        clipped + config.l2_penalty * regularization * weights
    )
    updated = np.clip(updated, -config.weight_limit, config.weight_limit)
    if "constraint_penalty" in config.feature_names:
        constraint_index = config.feature_names.index("constraint_penalty")
        updated[constraint_index] = min(0.0, updated[constraint_index])
    for name in config.zeroed_features:
        updated[config.feature_names.index(name)] = 0.0
    brier = float(np.dot(final, final) - 2 * actual_probability + 1)
    return _Step(
        model_probabilities=model,
        final_probabilities=final,
        gradient=gradient,
        clipped_gradient=clipped,
        weights_after=updated,
        log_loss=-math.log(max(actual_probability, 1e-15)),
        brier=brier,
    )


def _select_candidate(
    states: list[_CandidateState], block_start: int, config: OnlineGradientConfig
) -> OnlineGradientSelection:
    required = config.search_lookback + config.validation_lookback
    if any(len(state.log_losses) < required for state in states):
        raise ValueError("候选历史不足以执行Search/Validation校准")
    search_slice = slice(-required, -config.validation_lookback)
    model_states = [state for state in states if state.candidate.uniform_shrinkage > 0]
    if not model_states:
        raise ValueError("在线梯度至少需要一个λ>0的模型候选")
    selected = min(
        model_states,
        key=lambda state: (
            float(np.mean(state.log_losses[search_slice])),
            float(np.mean(state.brier_scores[search_slice])),
            state.candidate.learning_rate,
            state.candidate.uniform_shrinkage,
        ),
    )
    validation_log = np.asarray(selected.log_losses[-config.validation_lookback :])
    validation_brier = np.asarray(selected.brier_scores[-config.validation_lookback :])
    stable_blocks = sum(
        float(chunk.mean()) < _UNIFORM_LOG_LOSS
        for chunk in np.array_split(validation_log, 3)
    )
    search_log = float(np.mean(selected.log_losses[search_slice]))
    mean_log = float(validation_log.mean())
    mean_brier = float(validation_brier.mean())
    reasons = []
    if selected.candidate.uniform_shrinkage <= 0:
        reasons.append("Search选择λ=0")
    if search_log >= _UNIFORM_LOG_LOSS:
        reasons.append("Search LogLoss未优于均匀")
    if mean_log >= _UNIFORM_LOG_LOSS:
        reasons.append("Validation LogLoss未优于均匀")
    if mean_brier > _UNIFORM_BRIER:
        reasons.append("Validation Brier劣于均匀")
    if stable_blocks < 2:
        reasons.append("Validation稳定时间块不足2/3")
    return OnlineGradientSelection(
        block_start_index=block_start,
        candidate=selected.candidate,
        search_mean_log_loss=search_log,
        validation_mean_log_loss=mean_log,
        validation_mean_brier=mean_brier,
        stable_blocks=stable_blocks,
        abstained=bool(reasons),
        reasons=tuple(reasons),
    )


def run_online_gradient_research(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: OnlineGradientConfig,
    feature_config: LearnedFeatureConfig = LearnedFeatureConfig(),
    *,
    frozen_test_read: bool = False,
) -> OnlineGradientReport:
    """按严格逐期顺序执行开发或一次性Frozen在线梯度评估。"""

    if rule.code not in {"fc3d", "pl3"} or rule.draw_count != 3:
        raise ValueError("在线梯度v4只支持fc3d/pl3")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    if config.development_end > len(chronological):
        raise ValueError("development_end超过开发历史")
    outer_start = config.development_end - config.outer_periods
    calibration_history = config.search_lookback + config.validation_lookback
    audit_start = outer_start - calibration_history
    if audit_start < config.warmup_history:
        raise ValueError("开发历史不足以保留warmup和校准窗口")
    indices = tuple(range(audit_start, config.development_end))
    states = iter_rolling_history_states(chronological, rule, indices, feature_config)
    candidates = [
        OnlineGradientCandidate(learning_rate, shrinkage)
        for learning_rate in config.learning_rates
        for shrinkage in config.shrinkages
    ]
    learners = [
        _CandidateState(candidate, _initial_weights(config), [], [])
        for candidate in candidates
    ]
    selections: list[OnlineGradientSelection] = []
    periods: list[OnlineGradientPeriod] = []
    current_selection: OnlineGradientSelection | None = None
    for target_index, history_state in zip(indices, states):
        if target_index >= outer_start and (
            current_selection is None
            or (target_index - outer_start) % config.calibration_interval == 0
        ):
            current_selection = _select_candidate(learners, target_index, config)
            selections.append(current_selection)
        features = build_candidate_features(
            history_state,
            rule,
            include_behavioral_context=any(
                name in BEHAVIORAL_FEATURE_NAMES for name in config.feature_names
            ),
        )
        matrix = features[list(config.feature_names)].to_numpy(dtype=float)
        actual_row = chronological.iloc[target_index]
        actual_text = "".join(
            str(int(actual_row[column])) for column in rule.number_columns
        )
        actual_index = int(actual_text)
        steps = [
            online_gradient_step(
                matrix,
                actual_index,
                learner.weights,
                learner.candidate,
                config,
            )
            for learner in learners
        ]
        if target_index >= outer_start:
            if current_selection is None:
                raise RuntimeError("外层预测缺少参数选择")
            selected_index = candidates.index(current_selection.candidate)
            selected_learner = learners[selected_index]
            selected_step = steps[selected_index]
            probabilities = selected_step.final_probabilities
            order = rank_candidate_indices(probabilities, _CANDIDATES)
            rank = int(np.flatnonzero(order == actual_index)[0]) + 1
            boundary_index = int(order[min(config.direct_top_k, len(order)) - 1])
            contributions = selected_learner.weights * (
                matrix[actual_index] - matrix[boundary_index]
            )
            top50_shape_counts = {"组六": 0, "组三": 0, "豹子": 0}
            for candidate_index in order[: config.direct_top_k]:
                digits = tuple(
                    int(value) for value in _CANDIDATES[int(candidate_index)]
                )
                top50_shape_counts[classify_digit_shape(digits)] += 1
            deployed_probability = (
                1.0 / 1000.0
                if current_selection.abstained
                else float(probabilities[actual_index])
            )
            deployed_brier = (
                _UNIFORM_BRIER if current_selection.abstained else selected_step.brier
            )
            periods.append(
                OnlineGradientPeriod(
                    target_index=target_index,
                    target_issue=str(actual_row["期数"]),
                    history_end_issue=str(history_state.history_end_issue),
                    candidate_key=current_selection.candidate.key,
                    learning_rate=current_selection.candidate.learning_rate,
                    uniform_shrinkage=current_selection.candidate.uniform_shrinkage,
                    abstained=current_selection.abstained,
                    actual_text=actual_text,
                    research_actual_probability=float(probabilities[actual_index]),
                    deployed_actual_probability=deployed_probability,
                    research_rank=rank,
                    research_direct_hit=rank <= config.direct_top_k,
                    research_log_loss=selected_step.log_loss,
                    deployed_log_loss=-math.log(deployed_probability),
                    research_brier=selected_step.brier,
                    deployed_brier=deployed_brier,
                    gradient_norm=float(np.linalg.norm(selected_step.gradient)),
                    gradients={
                        name: float(value)
                        for name, value in zip(
                            config.feature_names, selected_step.gradient
                        )
                    },
                    boundary_contributions={
                        name: float(value)
                        for name, value in zip(config.feature_names, contributions)
                    },
                    weights_before={
                        name: float(value)
                        for name, value in zip(
                            config.feature_names, selected_learner.weights
                        )
                    },
                    weights_after={
                        name: float(value)
                        for name, value in zip(
                            config.feature_names, selected_step.weights_after
                        )
                    },
                    top50_shape_counts=top50_shape_counts,
                )
            )
        for learner, step in zip(learners, steps):
            learner.log_losses.append(step.log_loss)
            learner.brier_scores.append(step.brier)
            learner.weights = step.weights_after
    return OnlineGradientReport(
        lottery=rule.code,
        outer_start_index=outer_start,
        outer_end_index=config.development_end,
        frozen_test_read=frozen_test_read,
        evidence_status=(
            "independent_frozen_test"
            if frozen_test_read
            else "exploratory_reused_development"
        ),
        config=config,
        selections=tuple(selections),
        periods=tuple(periods),
    )


def write_online_gradient_report(
    report: OnlineGradientReport, path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "OnlineGradientCandidate",
    "OnlineGradientConfig",
    "OnlineGradientPeriod",
    "OnlineGradientReport",
    "OnlineGradientSelection",
    "online_gradient_step",
    "run_online_gradient_research",
    "write_online_gradient_report",
]
