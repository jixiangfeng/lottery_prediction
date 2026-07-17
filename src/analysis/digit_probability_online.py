# -*- coding: utf-8 -*-
"""三位彩在线概率集成与严格逐期开奖反馈评估。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_candidates import (
    ENSEMBLE_MODEL_NAMES,
    DigitCandidate,
    DigitCandidateConfig,
    DigitCandidateResult,
    DigitGroupCandidate,
)
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_probability import (
    _probability_pool,
    _profile_scores,
    _scoring_config,
    _softmax,
    _tie_key,
    _validate_probability_rule,
)
from src.analysis.digit_statistics import DigitStatisticsResult, analyze_digit_history
from src.analysis.prediction_viability import (
    PredictionViabilityReport,
    build_prediction_viability_report,
    calculate_group_random_probability,
)
from src.lotteries.base import LotteryRule

ONLINE_MODEL_PROFILES = ENSEMBLE_MODEL_NAMES[:14]


@dataclass(frozen=True)
class DigitOnlineProbabilityConfig:
    """固定在线更新规则，避免逐日重新搜索超参数。"""

    min_train_size: int = 100
    temperature: float = 0.2
    uniform_prior_weight: float = 0.5
    learning_rate: float = 1.0
    fixed_share: float = 0.01
    model_profiles: tuple[str, ...] = ONLINE_MODEL_PROFILES

    def __post_init__(self) -> None:
        if self.min_train_size <= 0:
            raise ValueError("在线概率最小训练期数必须为正整数")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("温度和在线学习率必须为正数")
        if not 0.0 < self.uniform_prior_weight < 1.0:
            raise ValueError("均匀基线先验权重必须位于 (0, 1)")
        if not 0.0 <= self.fixed_share < 1.0:
            raise ValueError("固定份额收缩必须位于 [0, 1)")
        if not self.model_profiles or len(set(self.model_profiles)) != len(
            self.model_profiles
        ):
            raise ValueError("在线模型列表不能为空或重复")
        if any(profile not in ONLINE_MODEL_PROFILES for profile in self.model_profiles):
            raise ValueError("在线模型列表包含未知或非统计模型")

    @property
    def expert_names(self) -> tuple[str, ...]:
        """返回均匀基线和参与在线调权的模型名。"""

        return ("uniform", *self.model_profiles)

    def prior_weights(self) -> np.ndarray:
        """返回保守先验：均匀基线占一半，其余由统计模型均分。"""

        model_weight = (1.0 - self.uniform_prior_weight) / len(self.model_profiles)
        return np.asarray(
            [self.uniform_prior_weight] + [model_weight] * len(self.model_profiles),
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minTrainSize": self.min_train_size,
            "temperature": self.temperature,
            "uniformPriorWeight": self.uniform_prior_weight,
            "learningRate": self.learning_rate,
            "fixedShare": self.fixed_share,
            "modelProfiles": list(self.model_profiles),
        }

    @property
    def signature(self) -> str:
        """返回影响在线状态兼容性的配置指纹。"""

        serialized = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class DigitOnlineProbabilityState:
    """已经消费完实际开奖结果、可直接预测下一期的在线权重。"""

    rule_code: str
    config_signature: str
    processed_periods: int
    feedback_periods: int
    latest_issue: str
    history_fingerprint: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "ruleCode": self.rule_code,
            "configSignature": self.config_signature,
            "processedPeriods": self.processed_periods,
            "feedbackPeriods": self.feedback_periods,
            "latestIssue": self.latest_issue,
            "historyFingerprint": self.history_fingerprint,
            "weights": dict(self.weights),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DigitOnlineProbabilityState":
        if int(payload.get("schemaVersion", 0)) != 1:
            raise ValueError("在线概率状态版本不兼容")
        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, dict):
            raise ValueError("在线概率状态缺少模型权重")
        return cls(
            rule_code=str(payload["ruleCode"]),
            config_signature=str(payload["configSignature"]),
            processed_periods=int(payload["processedPeriods"]),
            feedback_periods=int(payload["feedbackPeriods"]),
            latest_issue=str(payload["latestIssue"]),
            history_fingerprint=str(payload["historyFingerprint"]),
            weights={str(name): float(value) for name, value in raw_weights.items()},
        )


@dataclass(frozen=True)
class DigitOnlineProbabilityStateUpdate:
    """本次日报对在线状态执行的更新证据。"""

    mode: str
    processed_rows: int
    feedback_updates: int
    rebuild_reason: str | None
    state_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "processedRows": self.processed_rows,
            "feedbackUpdates": self.feedback_updates,
            "rebuildReason": self.rebuild_reason,
            "statePath": self.state_path,
        }


@dataclass(frozen=True)
class DigitOnlineProbabilityDistribution:
    """在线专家权重合并后的完整三位号码概率。"""

    probabilities: Mapping[str, float]
    probability_sum: float
    squared_probability_sum: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probabilities", MappingProxyType(dict(self.probabilities))
        )

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            dict(self.probabilities),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class DigitOnlineProbabilityPlan:
    """使用最新在线权重生成的下一期直选、组选概率候选。"""

    rule_code: str
    display_name: str
    direct_candidates: list[DigitCandidate]
    group_candidates: list[DigitGroupCandidate]
    config: DigitCandidateConfig
    distribution: DigitOnlineProbabilityDistribution
    online_config: DigitOnlineProbabilityConfig
    state: DigitOnlineProbabilityState
    state_update: DigitOnlineProbabilityStateUpdate
    model_candidates: dict[str, list[str]]

    @property
    def candidates(self) -> list[DigitCandidate]:
        return self.direct_candidates

    def to_dict(self) -> dict[str, Any]:
        legacy = DigitCandidateResult(
            self.rule_code,
            self.display_name,
            self.direct_candidates,
            self.config,
            self.model_candidates,
        ).to_dict()
        legacy["directCandidates"] = [
            candidate.to_dict() for candidate in self.direct_candidates
        ]
        legacy["groupCandidates"] = [
            candidate.to_dict() for candidate in self.group_candidates
        ]
        legacy["probabilityModel"] = "online_expert_mixture_v3"
        legacy["onlineProbability"] = {
            "config": self.online_config.to_dict(),
            "state": self.state.to_dict(),
            "stateUpdate": self.state_update.to_dict(),
        }
        legacy["probabilitySum"] = self.distribution.probability_sum
        legacy["probabilityDistributionFingerprint"] = self.distribution.fingerprint
        return legacy


@dataclass(frozen=True)
class DigitOnlineProbabilityIssue:
    """一个评估期在开奖前后的概率、候选与权重证据。"""

    issue: str
    train_end_issue: str
    train_size: int
    actual_text: str
    actual_probability: float
    probability_sum: float
    log_loss: float
    brier_score: float
    actual_midrank: float
    actual_rank_percentile: float
    direct_candidates: list[str]
    group_candidates: list[str]
    direct_hit: bool
    group_hit: bool
    direct_random_probability: float
    group_random_probability: float
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    expert_actual_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "trainEndIssue": self.train_end_issue,
            "trainSize": self.train_size,
            "actualText": self.actual_text,
            "actualProbability": self.actual_probability,
            "probabilitySum": self.probability_sum,
            "logLoss": self.log_loss,
            "brierScore": self.brier_score,
            "actualMidrank": self.actual_midrank,
            "actualRankPercentile": self.actual_rank_percentile,
            "directCandidates": self.direct_candidates,
            "groupCandidates": self.group_candidates,
            "directHit": self.direct_hit,
            "groupHit": self.group_hit,
            "directRandomProbability": self.direct_random_probability,
            "groupRandomProbability": self.group_random_probability,
            "weightsBefore": self.weights_before,
            "weightsAfter": self.weights_after,
            "expertActualProbabilities": self.expert_actual_probabilities,
        }


@dataclass(frozen=True)
class DigitOnlineProbabilityReport:
    """固定更新规则后，逐期先预测再反馈的500期开发报告。"""

    rule_code: str
    display_name: str
    development_only: bool
    config: DigitOnlineProbabilityConfig
    training_periods: int
    pretraining_feedback_periods: int
    initial_evaluation_weights: dict[str, float]
    final_weights: dict[str, float]
    issues: list[DigitOnlineProbabilityIssue]
    mean_log_loss: float
    uniform_log_loss: float
    mean_brier_score: float
    uniform_brier_score: float
    mean_actual_rank_percentile: float
    mean_reciprocal_rank: float
    expert_mean_log_losses: dict[str, float]
    viability: PredictionViabilityReport

    @property
    def log_loss_improvement(self) -> float:
        return self.uniform_log_loss - self.mean_log_loss

    @property
    def brier_improvement(self) -> float:
        return self.uniform_brier_score - self.mean_brier_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "experimentModel": "digit_probability_online_v3",
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "developmentOnly": self.development_only,
            "config": self.config.to_dict(),
            "trainingPeriods": self.training_periods,
            "pretrainingFeedbackPeriods": self.pretraining_feedback_periods,
            "periods": len(self.issues),
            "initialEvaluationWeights": self.initial_evaluation_weights,
            "finalWeights": self.final_weights,
            "meanLogLoss": self.mean_log_loss,
            "uniformLogLoss": self.uniform_log_loss,
            "logLossImprovement": self.log_loss_improvement,
            "meanBrierScore": self.mean_brier_score,
            "uniformBrierScore": self.uniform_brier_score,
            "brierImprovement": self.brier_improvement,
            "meanActualRankPercentile": self.mean_actual_rank_percentile,
            "meanReciprocalRank": self.mean_reciprocal_rank,
            "directHits": sum(issue.direct_hit for issue in self.issues),
            "groupHits": sum(issue.group_hit for issue in self.issues),
            "expertMeanLogLosses": self.expert_mean_log_losses,
            "viability": self.viability.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def update_online_weights(
    weights: Sequence[float] | np.ndarray,
    expert_actual_probabilities: Sequence[float] | np.ndarray,
    config: DigitOnlineProbabilityConfig,
) -> np.ndarray:
    """按当期 Log Loss 更新专家权重，再向固定先验收缩。

    示例：``update_online_weights([0.5, 0.5], [0.001, 0.002], config)``。
    """

    current = np.asarray(weights, dtype=float)
    likelihoods = np.asarray(expert_actual_probabilities, dtype=float)
    prior = config.prior_weights()
    if current.shape != likelihoods.shape or current.shape != prior.shape:
        raise ValueError("权重、专家概率与配置中的专家数量必须一致")
    if np.any(current < 0) or not math.isclose(float(current.sum()), 1.0):
        raise ValueError("在线权重必须非负且总和为1")
    if np.any(likelihoods < 0) or not np.all(np.isfinite(likelihoods)):
        raise ValueError("专家对真实开奖号的概率必须是有限非负数")
    log_posterior = np.log(np.clip(current, 1e-300, None)) + (
        config.learning_rate * np.log(np.clip(likelihoods, 1e-300, None))
    )
    log_posterior -= float(log_posterior.max())
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()
    updated = (1.0 - config.fixed_share) * posterior + config.fixed_share * prior
    return updated / updated.sum()


def _weight_mapping(names: Sequence[str], weights: np.ndarray) -> dict[str, float]:
    return {name: float(weight) for name, weight in zip(names, weights)}


def _expert_distributions(
    train: pd.DataFrame,
    rule: LotteryRule,
    config: DigitOnlineProbabilityConfig,
) -> tuple[Any, np.ndarray]:
    candidate_config = DigitCandidateConfig(
        ranking_mode="ensemble",
        exclude_latest=False,
        constraint_mode="off",
    )
    scoring = _scoring_config(candidate_config, len(train))
    stats = analyze_digit_history(
        train, rule, frequency_windows=scoring.frequency_windows
    )
    return _expert_distributions_from_stats(stats, rule, config)


def _expert_distributions_from_stats(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitOnlineProbabilityConfig,
) -> tuple[Any, np.ndarray]:
    """复用已计算统计，生成均匀专家和全部统计专家概率。"""

    candidate_config = DigitCandidateConfig(
        ranking_mode="ensemble",
        exclude_latest=False,
        constraint_mode="off",
    )
    pool, _ = _probability_pool(stats, rule, candidate_config)
    distributions = [np.full(len(pool), 1.0 / len(pool), dtype=float)]
    distributions.extend(
        _softmax(_profile_scores(pool, profile), config.temperature)
        for profile in config.model_profiles
    )
    return pool, np.vstack(distributions)


def _history_fingerprint(history: pd.DataFrame, rule: LotteryRule) -> str:
    rows = [
        [
            str(row["期数"]),
            *[int(row[column]) for column in rule.number_columns],
        ]
        for _, row in history.iterrows()
    ]
    serialized = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _load_online_state(path: Path) -> DigitOnlineProbabilityState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("在线概率状态必须是 JSON 对象")
    return DigitOnlineProbabilityState.from_dict(payload)


def _state_is_compatible(
    state: DigitOnlineProbabilityState,
    history: pd.DataFrame,
    rule: LotteryRule,
    config: DigitOnlineProbabilityConfig,
) -> tuple[bool, str | None]:
    if state.rule_code != rule.code or state.config_signature != config.signature:
        return False, "config_changed"
    if not config.min_train_size <= state.processed_periods <= len(history):
        return False, "history_changed"
    prefix = history.iloc[: state.processed_periods]
    if (
        str(prefix.iloc[-1]["期数"]) != state.latest_issue
        or _history_fingerprint(prefix, rule) != state.history_fingerprint
    ):
        return False, "history_changed"
    if set(state.weights) != set(config.expert_names):
        return False, "config_changed"
    if not math.isclose(sum(state.weights.values()), 1.0, abs_tol=1e-12):
        return False, "state_invalid"
    return True, None


def _update_state_from_history(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: DigitOnlineProbabilityConfig,
    state_path: Path | None,
    rebuild: bool,
) -> tuple[DigitOnlineProbabilityState, DigitOnlineProbabilityStateUpdate]:
    existing: DigitOnlineProbabilityState | None = None
    rebuild_reason: str | None = "requested" if rebuild else None
    if state_path is not None and state_path.exists() and not rebuild:
        try:
            candidate = _load_online_state(state_path)
            compatible, rebuild_reason = _state_is_compatible(
                candidate, history, rule, config
            )
            if compatible:
                existing = candidate
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            rebuild_reason = "state_invalid"
    elif not rebuild:
        rebuild_reason = "missing"

    if existing is None:
        weights = config.prior_weights()
        start_index = config.min_train_size
        feedback_periods = 0
        mode = "full_rebuild"
    else:
        weights = np.asarray(
            [existing.weights[name] for name in config.expert_names], dtype=float
        )
        start_index = existing.processed_periods
        feedback_periods = existing.feedback_periods
        mode = "cache_hit" if start_index == len(history) else "incremental"
        rebuild_reason = None

    feedback_updates = 0
    for target_index in range(start_index, len(history)):
        weights = _feedback(
            history.iloc[:target_index],
            history.iloc[target_index],
            rule,
            config,
            weights,
        )
        feedback_updates += 1
    feedback_periods += feedback_updates
    state = DigitOnlineProbabilityState(
        rule_code=rule.code,
        config_signature=config.signature,
        processed_periods=len(history),
        feedback_periods=feedback_periods,
        latest_issue=str(history.iloc[-1]["期数"]),
        history_fingerprint=_history_fingerprint(history, rule),
        weights=_weight_mapping(config.expert_names, weights),
    )
    update = DigitOnlineProbabilityStateUpdate(
        mode=mode,
        processed_rows=(len(history) if mode == "full_rebuild" else feedback_updates),
        feedback_updates=feedback_updates,
        rebuild_reason=rebuild_reason,
        state_path=str(state_path) if state_path is not None else None,
    )
    if state_path is not None:
        _atomic_write(
            state_path,
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
    return state, update


def _actual_text(target: pd.Series, rule: LotteryRule) -> str:
    return "".join(str(int(target[column])) for column in rule.number_columns)


def _midrank(probabilities: np.ndarray, actual_index: int) -> float:
    actual = float(probabilities[actual_index])
    tolerance = 1e-15
    greater = int(np.count_nonzero(probabilities > actual + tolerance))
    tied = int(np.count_nonzero(np.abs(probabilities - actual) <= tolerance))
    return greater + (tied + 1) / 2


def _select_candidates(
    pool: Any,
    probabilities: np.ndarray,
    candidate_count: int,
    seed: str,
) -> tuple[list[str], list[str]]:
    ordered = sorted(
        range(len(pool)),
        key=lambda index: (
            -float(probabilities[index]),
            _tie_key(seed, pool.text_at(index)),
        ),
    )
    direct = [pool.text_at(index) for index in ordered[:candidate_count]]
    group_mass: dict[str, float] = {}
    for index in range(len(pool)):
        key = "".join(sorted(pool.text_at(index)))
        if len(set(key)) == 1:
            continue
        group_mass[key] = group_mass.get(key, 0.0) + float(probabilities[index])
    groups = sorted(
        group_mass,
        key=lambda key: (-group_mass[key], _tie_key(seed, key)),
    )[:candidate_count]
    return direct, groups


def build_digit_online_probability_plan(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    candidate_count: int = 10,
    online_config: DigitOnlineProbabilityConfig | None = None,
    state_path: str | Path | None = None,
    rebuild_state: bool = False,
) -> DigitOnlineProbabilityPlan:
    """消费截至最新期的反馈状态，并生成下一期完整概率与纯TopK候选。"""

    _validate_probability_rule(rule)
    if candidate_count <= 0:
        raise ValueError("候选数量必须为正整数")
    config = online_config or DigitOnlineProbabilityConfig()
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    if len(chronological) < config.min_train_size:
        raise ValueError(f"在线概率至少需要 {config.min_train_size} 期历史后才能预测")
    effective_state_path = Path(state_path) if state_path is not None else None
    state, state_update = _update_state_from_history(
        chronological, rule, config, effective_state_path, rebuild_state
    )
    scoring = _scoring_config(
        DigitCandidateConfig(
            ranking_mode="ensemble",
            exclude_latest=False,
            constraint_mode="off",
        ),
        len(chronological),
    )
    current_stats = analyze_digit_history(
        chronological, rule, frequency_windows=scoring.frequency_windows
    )
    if str(current_stats.latest_issue) != str(chronological.iloc[-1]["期数"]):
        raise ValueError("在线概率统计截止期与历史最新期不一致")
    pool, experts = _expert_distributions_from_stats(current_stats, rule, config)
    weights = np.asarray(
        [state.weights[name] for name in config.expert_names], dtype=float
    )
    probabilities = np.asarray(weights @ experts, dtype=float)
    probabilities /= probabilities.sum()
    source_issue = str(current_stats.latest_issue)
    direct_texts, group_keys = _select_candidates(
        pool, probabilities, candidate_count, source_issue
    )
    direct_candidates: list[DigitCandidate] = []
    for text in direct_texts:
        index = pool.rank_for_text(text)
        if index is None:
            raise ValueError("在线概率直选候选不在完整空间")
        probability = float(probabilities[index])
        direct_candidates.append(
            replace(
                pool.candidate_at(index),
                score=math.log(max(probability, 1e-300)),
                joint_probability=probability,
                predicted_probability=probability,
            )
        )

    group_mass: dict[str, float] = {}
    group_permutations: Counter[str] = Counter()
    probability_by_text: dict[str, float] = {}
    for index in range(len(pool)):
        text = pool.text_at(index)
        probability = float(probabilities[index])
        probability_by_text[text] = probability
        key = "".join(sorted(text))
        if len(set(key)) == 1:
            continue
        group_mass[key] = group_mass.get(key, 0.0) + probability
        group_permutations[key] += 1
    group_candidates = [
        DigitGroupCandidate(
            group_key=key,
            numbers=[int(value) for value in key],
            shape="组三" if len(set(key)) == 2 else "组六",
            probability_mass=group_mass[key],
            score=math.log(max(group_mass[key], 1e-300)),
            permutations=group_permutations[key],
            predicted_probability=group_mass[key],
            ranking_model="exact_permutation_online_probability_v3",
        )
        for key in group_keys
    ]
    plan_config = replace(
        scoring,
        count=candidate_count,
        ranking_mode="online_probability",
    )
    return DigitOnlineProbabilityPlan(
        rule_code=rule.code,
        display_name=rule.display_name,
        direct_candidates=direct_candidates,
        group_candidates=group_candidates,
        config=plan_config,
        distribution=DigitOnlineProbabilityDistribution(
            probabilities=probability_by_text,
            probability_sum=float(probabilities.sum()),
            squared_probability_sum=float(np.square(probabilities).sum()),
        ),
        online_config=config,
        state=state,
        state_update=state_update,
        model_candidates={},
    )


def _feedback(
    train: pd.DataFrame,
    target: pd.Series,
    rule: LotteryRule,
    config: DigitOnlineProbabilityConfig,
    weights: np.ndarray,
) -> np.ndarray:
    pool, experts = _expert_distributions(train, rule, config)
    actual_index = pool.rank_for_text(_actual_text(target, rule))
    if actual_index is None:
        raise ValueError("真实开奖号不在完整概率空间")
    return update_online_weights(weights, experts[:, actual_index], config)


def run_digit_online_probability_walk_forward(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    periods: int = 500,
    candidate_count: int = 10,
    online_config: DigitOnlineProbabilityConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> DigitOnlineProbabilityReport:
    """用开发段预训练权重，再在目标段执行预测后反馈的严格前推。"""

    _validate_probability_rule(rule)
    if periods <= 0 or candidate_count <= 0:
        raise ValueError("periods 和 candidate_count 必须为正整数")
    config = online_config or DigitOnlineProbabilityConfig()
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    first_target_index = len(chronological) - periods
    if first_target_index <= config.min_train_size:
        raise ValueError("历史不足：评估段之前必须留出超过最小训练期数的开发数据")

    weights = config.prior_weights()
    pretraining_indexes = range(config.min_train_size, first_target_index)
    pretraining_total = first_target_index - config.min_train_size
    for feedback_number, target_index in enumerate(pretraining_indexes, start=1):
        weights = _feedback(
            chronological.iloc[:target_index],
            chronological.iloc[target_index],
            rule,
            config,
            weights,
        )
        if progress_callback and (
            feedback_number % 50 == 0 or feedback_number == pretraining_total
        ):
            progress_callback(f"预训练反馈 {feedback_number}/{pretraining_total}")

    names = config.expert_names
    initial_evaluation_weights = _weight_mapping(names, weights)
    issues: list[DigitOnlineProbabilityIssue] = []
    expert_losses: dict[str, list[float]] = {name: [] for name in names}
    for evaluation_number, target_index in enumerate(
        range(first_target_index, len(chronological)), start=1
    ):
        train = chronological.iloc[:target_index]
        target = chronological.iloc[target_index]
        pool, experts = _expert_distributions(train, rule, config)
        probabilities = np.asarray(weights @ experts, dtype=float)
        probabilities /= probabilities.sum()
        actual_text = _actual_text(target, rule)
        actual_index = pool.rank_for_text(actual_text)
        if actual_index is None:
            raise ValueError("真实开奖号不在完整概率空间")
        actual_probability = float(probabilities[actual_index])
        expert_actual = experts[:, actual_index]
        weights_before = _weight_mapping(names, weights)
        weights = update_online_weights(weights, expert_actual, config)
        weights_after = _weight_mapping(names, weights)
        for name, probability in zip(names, expert_actual):
            expert_losses[name].append(-math.log(max(float(probability), 1e-300)))
        direct, groups = _select_candidates(
            pool, probabilities, candidate_count, str(train.iloc[-1]["期数"])
        )
        actual_group = "".join(sorted(actual_text))
        midrank = _midrank(probabilities, actual_index)
        issues.append(
            DigitOnlineProbabilityIssue(
                issue=str(target["期数"]),
                train_end_issue=str(train.iloc[-1]["期数"]),
                train_size=len(train),
                actual_text=actual_text,
                actual_probability=actual_probability,
                probability_sum=float(probabilities.sum()),
                log_loss=-math.log(max(actual_probability, 1e-300)),
                brier_score=float(np.square(probabilities).sum())
                - 2.0 * actual_probability
                + 1.0,
                actual_midrank=midrank,
                actual_rank_percentile=midrank / len(probabilities),
                direct_candidates=direct,
                group_candidates=groups,
                direct_hit=actual_text in direct,
                group_hit=actual_group in groups,
                direct_random_probability=len(set(direct)) / len(probabilities),
                group_random_probability=calculate_group_random_probability(groups),
                weights_before=weights_before,
                weights_after=weights_after,
                expert_actual_probabilities={
                    name: float(probability)
                    for name, probability in zip(names, expert_actual)
                },
            )
        )
        if progress_callback and (
            evaluation_number % 50 == 0 or evaluation_number == periods
        ):
            progress_callback(f"在线评估 {evaluation_number}/{periods}")

    uniform_probability = 1.0 / (10**rule.draw_count)
    uniform_log_loss = -math.log(uniform_probability)
    uniform_brier = 1.0 - uniform_probability
    viability = build_prediction_viability_report(
        [issue.direct_hit for issue in issues],
        [issue.direct_random_probability for issue in issues],
        group_hits=[issue.group_hit for issue in issues],
        group_random_probabilities=[issue.group_random_probability for issue in issues],
    )
    return DigitOnlineProbabilityReport(
        rule_code=rule.code,
        display_name=rule.display_name,
        development_only=True,
        config=config,
        training_periods=first_target_index,
        pretraining_feedback_periods=pretraining_total,
        initial_evaluation_weights=initial_evaluation_weights,
        final_weights=_weight_mapping(names, weights),
        issues=issues,
        mean_log_loss=mean(issue.log_loss for issue in issues),
        uniform_log_loss=uniform_log_loss,
        mean_brier_score=mean(issue.brier_score for issue in issues),
        uniform_brier_score=uniform_brier,
        mean_actual_rank_percentile=mean(
            issue.actual_rank_percentile for issue in issues
        ),
        mean_reciprocal_rank=mean(1.0 / issue.actual_midrank for issue in issues),
        expert_mean_log_losses={
            name: mean(losses) for name, losses in expert_losses.items()
        },
        viability=viability,
    )


def build_digit_online_probability_markdown(
    report: DigitOnlineProbabilityReport,
) -> str:
    """生成在线概率500期开发评估报告。"""

    direct_hits = sum(issue.direct_hit for issue in report.issues)
    group_hits = sum(issue.group_hit for issue in report.issues)
    probability_passed = (
        report.log_loss_improvement > 0 and report.brier_improvement > 0
    )
    lines = [
        f"# {report.display_name} 在线概率 v3 严格前推开发评估",
        "",
        "## 边界",
        "",
        "- 更新纪律：先预测、后开奖、再更新；每期预测只读取此前历史。",
        "- 更新规则和超参数在评估开始前固定，评估段只更新模型状态与权重。",
        "- 本区间已用于开发研究，不能重新表述为未来未见验证。",
        "",
        "## 配置",
        "",
        f"- 开发训练：`{report.training_periods}` 期；其中在线预训练反馈："
        f"`{report.pretraining_feedback_periods}` 期。",
        f"- 评估：`{len(report.issues)}` 期；温度：`{report.config.temperature:.2f}`；"
        f"学习率：`{report.config.learning_rate:.2f}`；固定份额："
        f"`{report.config.fixed_share:.3f}`。",
        f"- 专家：均匀基线 + {len(report.config.model_profiles)} 个统计模型；"
        "不使用固定 ensemble、蒙特卡洛或机器学习专家。",
        "",
        "## 概率质量",
        "",
        "| 指标 | 在线 v3 | 均匀分布 | 改善（正数更好） |",
        "|---|---:|---:|---:|",
        f"| Log Loss | {report.mean_log_loss:.6f} | {report.uniform_log_loss:.6f} | "
        f"{report.log_loss_improvement:+.6f} |",
        f"| Brier Score | {report.mean_brier_score:.6f} | "
        f"{report.uniform_brier_score:.6f} | {report.brier_improvement:+.6f} |",
        "",
        f"- 平均真实开奖号排名分位：`{report.mean_actual_rank_percentile:.2%}`",
        f"- MRR：`{report.mean_reciprocal_rank:.6f}`",
        f"- 概率质量：`{'通过' if probability_passed else '不通过'}`",
        "",
        "## 在线权重与单模型",
        "",
        "| 专家 | 评估初始权重 | 最终权重 | 评估期平均 Log Loss |",
        "|---|---:|---:|---:|",
    ]
    for name in report.config.expert_names:
        lines.append(
            f"| {name} | {report.initial_evaluation_weights[name]:.6f} | "
            f"{report.final_weights[name]:.6f} | "
            f"{report.expert_mean_log_losses[name]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 命中与随机闸门",
            "",
            f"- 直选：`{direct_hits}/{len(report.issues)}`；组选："
            f"`{group_hits}/{len(report.issues)}`",
            f"- 整体统计可行性：`{'通过' if report.viability.viable else '不通过'}`",
            f"- 原因：{report.viability.reason}",
            "",
            "在线调权只是一种严格反馈机制；如果历史特征没有稳定信息，"
            "结果仍应回到均匀基线，不保证中奖或盈利。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_digit_online_probability_reports(
    report: DigitOnlineProbabilityReport,
    output_dir: str | Path,
    *,
    prefix: str = "digit_probability_online_v3",
) -> tuple[Path, Path]:
    """原子写入在线概率 Markdown 与完整逐期 JSON。"""

    directory = Path(output_dir)
    markdown_path = directory / f"{prefix}_{report.rule_code}.md"
    json_path = directory / f"{prefix}_{report.rule_code}.json"
    _atomic_write(markdown_path, build_digit_online_probability_markdown(report))
    _atomic_write(
        json_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    return markdown_path, json_path


__all__ = [
    "DigitOnlineProbabilityConfig",
    "DigitOnlineProbabilityDistribution",
    "DigitOnlineProbabilityIssue",
    "DigitOnlineProbabilityPlan",
    "DigitOnlineProbabilityReport",
    "DigitOnlineProbabilityState",
    "DigitOnlineProbabilityStateUpdate",
    "build_digit_online_probability_plan",
    "build_digit_online_probability_markdown",
    "run_digit_online_probability_walk_forward",
    "update_online_weights",
    "write_digit_online_probability_reports",
]
