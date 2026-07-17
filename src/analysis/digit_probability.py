# -*- coding: utf-8 -*-
"""三位数字彩精确概率分布、校准与纯 TopK 候选。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from statistics import mean
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_candidates import (
    ENSEMBLE_MODEL_NAMES,
    DigitCandidate,
    DigitCandidateConfig,
    DigitCandidateResult,
    DigitGroupCandidate,
    _score_candidate_space,
    with_all_history_window,
)
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_statistics import DigitStatisticsResult, analyze_digit_history
from src.lotteries.base import LotteryRule

PROBABILITY_MODEL_PROFILES = ("ensemble", *ENSEMBLE_MODEL_NAMES[:14])


@dataclass(frozen=True)
class DigitProbabilityConfig:
    """概率校准使用的严格历史窗口与保守候选网格。"""

    validation_periods: int = 180
    min_train_size: int = 100
    minimum_validation_periods: int = 90
    temperatures: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
    learned_weights: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    model_profiles: tuple[str, ...] = PROBABILITY_MODEL_PROFILES
    minimum_log_loss_improvement: float = 0.001

    def __post_init__(self) -> None:
        if self.validation_periods <= 0 or self.min_train_size <= 0:
            raise ValueError("概率校准窗口与最小训练期数必须为正整数")
        if self.minimum_validation_periods < 3:
            raise ValueError("概率校准至少需要 3 个验证期")
        if not self.temperatures or any(value <= 0 for value in self.temperatures):
            raise ValueError("温度参数必须为正数")
        if not self.learned_weights or any(
            not 0.0 < value <= 1.0 for value in self.learned_weights
        ):
            raise ValueError("学习分布权重必须位于 (0, 1]")
        if self.minimum_log_loss_improvement < 0:
            raise ValueError("最小 Log Loss 改善不得为负数")
        if not self.model_profiles or any(
            value not in PROBABILITY_MODEL_PROFILES for value in self.model_profiles
        ):
            raise ValueError("概率模型剖面包含未知模型")


@dataclass(frozen=True)
class DigitProbabilityCalibration:
    """仅由目标期之前数据得到的概率混合参数与闸门证据。"""

    applied_learned_weight: float
    selected_learned_weight: float
    selected_model: str
    temperature: float
    validation_periods: int
    selection_periods: int
    holdout_periods: int
    uniform_log_loss: float
    selection_log_loss: float | None
    holdout_log_loss: float | None
    block_log_losses: tuple[float, ...]
    passed: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "appliedLearnedWeight": self.applied_learned_weight,
            "selectedLearnedWeight": self.selected_learned_weight,
            "selectedModel": self.selected_model,
            "temperature": self.temperature,
            "validationPeriods": self.validation_periods,
            "selectionPeriods": self.selection_periods,
            "holdoutPeriods": self.holdout_periods,
            "uniformLogLoss": self.uniform_log_loss,
            "selectionLogLoss": self.selection_log_loss,
            "holdoutLogLoss": self.holdout_log_loss,
            "blockLogLosses": list(self.block_log_losses),
            "passed": self.passed,
            "fallbackReason": self.fallback_reason,
        }


@dataclass(frozen=True)
class DigitProbabilityDistribution:
    """完整三位号码空间上的归一化概率。"""

    probabilities: Mapping[str, float]
    calibration: DigitProbabilityCalibration
    probability_sum: float
    squared_probability_sum: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probabilities", MappingProxyType(dict(self.probabilities))
        )

    def probability_for(self, text: str) -> float:
        return float(self.probabilities.get(str(text), 0.0))

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            dict(self.probabilities),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()

    def midrank(self, text: str) -> float:
        """返回概率降序下处理并列后的 1-based 中位排名。"""

        actual = self.probability_for(text)
        tolerance = 1e-15
        greater = sum(
            value > actual + tolerance for value in self.probabilities.values()
        )
        tied = sum(
            abs(value - actual) <= tolerance for value in self.probabilities.values()
        )
        return greater + (tied + 1) / 2


@dataclass(frozen=True)
class DigitProbabilityPlan:
    """概率模式的直选、组选与完整校准证据。"""

    rule_code: str
    display_name: str
    direct_candidates: list[DigitCandidate]
    group_candidates: list[DigitGroupCandidate]
    config: DigitCandidateConfig
    distribution: DigitProbabilityDistribution
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
        legacy["probabilityModel"] = "uniform_shrunk_rank_softmax_v2"
        legacy["probabilityCalibration"] = self.distribution.calibration.to_dict()
        legacy["probabilitySum"] = self.distribution.probability_sum
        legacy["probabilityDistributionFingerprint"] = self.distribution.fingerprint
        return legacy


def _validate_probability_rule(rule: LotteryRule) -> None:
    if rule.category != "digit" or rule.draw_count != 3:
        raise ValueError("概率 v2 当前只支持福彩3D和排列三")


def _validate_complete_space(config: DigitCandidateConfig) -> None:
    if any(
        value is not None
        for value in (
            config.sum_min,
            config.sum_max,
            config.span_min,
            config.span_max,
            config.allowed_shapes,
        )
    ):
        raise ValueError("概率模式必须保留完整 000-999 空间，不支持硬过滤")


def _scoring_config(
    config: DigitCandidateConfig, total_issues: int
) -> DigitCandidateConfig:
    _validate_complete_space(config)
    model_weights = tuple(config.ensemble_model_weights[:14]) + (0.0, 0.0)
    effective = replace(
        config,
        ranking_mode="ensemble",
        exclude_latest=False,
        constraint_mode="off",
        ensemble_model_weights=model_weights,
    )
    return with_all_history_window(effective, total_issues)


def _probability_pool(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitCandidateConfig,
):
    effective = _scoring_config(config, stats.total_issues)
    pool = _score_candidate_space(stats, rule, effective, None)
    expected = 10**rule.draw_count
    if len(pool) != expected:
        raise ValueError(f"概率模式要求完整 {expected} 个号码，实际得到 {len(pool)} 个")
    return pool, effective


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = (values - float(values.max())) / float(temperature)
    weights = np.exp(scaled)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    return weights / total


def _profile_scores(pool: Any, profile: str) -> np.ndarray:
    if profile == "ensemble":
        return pool.ensemble_scores
    model_index = ENSEMBLE_MODEL_NAMES.index(profile)
    return pool.model_percentiles[:, model_index]


def _block_means(values: Sequence[float], count: int = 3) -> tuple[float, ...]:
    blocks = []
    for index in range(count):
        start = index * len(values) // count
        end = (index + 1) * len(values) // count
        blocks.append(mean(values[start:end]))
    return tuple(blocks)


def _fallback_calibration(
    rule: LotteryRule,
    validation_periods: int,
    reason: str,
    *,
    selected_weight: float = 0.0,
    selected_model: str = "ensemble",
    temperature: float = 1.0,
    selection_periods: int = 0,
    holdout_periods: int = 0,
    selection_log_loss: float | None = None,
    holdout_log_loss: float | None = None,
    block_log_losses: tuple[float, ...] = (),
) -> DigitProbabilityCalibration:
    return DigitProbabilityCalibration(
        applied_learned_weight=0.0,
        selected_learned_weight=selected_weight,
        selected_model=selected_model,
        temperature=temperature,
        validation_periods=validation_periods,
        selection_periods=selection_periods,
        holdout_periods=holdout_periods,
        uniform_log_loss=math.log(10**rule.draw_count),
        selection_log_loss=selection_log_loss,
        holdout_log_loss=holdout_log_loss,
        block_log_losses=block_log_losses,
        passed=False,
        fallback_reason=reason,
    )


def fit_digit_probability_calibration(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    candidate_config: DigitCandidateConfig | None = None,
    probability_config: DigitProbabilityConfig | None = None,
) -> DigitProbabilityCalibration:
    """用历史末段的前2/3选参、后1/3守门，失败时回退均匀分布。"""

    _validate_probability_rule(rule)
    base_config = candidate_config or DigitCandidateConfig(ranking_mode="ensemble")
    _validate_complete_space(base_config)
    calibration_config = probability_config or DigitProbabilityConfig()
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    available = len(chronological) - calibration_config.min_train_size
    validation_periods = min(calibration_config.validation_periods, max(0, available))
    if validation_periods < calibration_config.minimum_validation_periods:
        return _fallback_calibration(
            rule,
            validation_periods,
            f"严格历史验证期不足 {calibration_config.minimum_validation_periods} 期",
        )

    target_indexes = range(len(chronological) - validation_periods, len(chronological))
    parameter_losses: dict[tuple[str, float, float], list[float]] = {
        (profile, temperature, learned_weight): []
        for profile in calibration_config.model_profiles
        for temperature in calibration_config.temperatures
        for learned_weight in calibration_config.learned_weights
    }
    uniform_probability = 1.0 / (10**rule.draw_count)
    for target_index in target_indexes:
        train = chronological.iloc[:target_index]
        target = chronological.iloc[target_index]
        scoring = _scoring_config(base_config, len(train))
        stats = analyze_digit_history(
            train, rule, frequency_windows=scoring.frequency_windows
        )
        pool, _ = _probability_pool(stats, rule, base_config)
        actual_text = "".join(
            str(int(target[column])) for column in rule.number_columns
        )
        actual_rank = pool.rank_for_text(actual_text)
        if actual_rank is None:
            raise ValueError("真实开奖号不在完整概率空间")
        for profile in calibration_config.model_profiles:
            profile_scores = _profile_scores(pool, profile)
            for temperature in calibration_config.temperatures:
                learned = _softmax(profile_scores, temperature)
                learned_actual = float(learned[actual_rank])
                for learned_weight in calibration_config.learned_weights:
                    probability = (
                        1.0 - learned_weight
                    ) * uniform_probability + learned_weight * learned_actual
                    parameter_losses[(profile, temperature, learned_weight)].append(
                        -math.log(max(probability, 1e-300))
                    )

    selection_periods = validation_periods * 2 // 3
    holdout_periods = validation_periods - selection_periods
    selected = min(
        parameter_losses,
        key=lambda item: (
            mean(parameter_losses[item][:selection_periods]),
            item[2],
            item[0] != "ensemble",
            -item[1],
        ),
    )
    selected_model, temperature, selected_weight = selected
    losses = parameter_losses[selected]
    selection_loss = mean(losses[:selection_periods])
    holdout_loss = mean(losses[selection_periods:])
    block_losses = _block_means(losses)
    uniform_loss = math.log(10**rule.draw_count)
    required = calibration_config.minimum_log_loss_improvement
    selection_passed = selection_loss < uniform_loss - required
    holdout_passed = holdout_loss < uniform_loss - required
    blocks_passed = all(value < uniform_loss for value in block_losses)
    passed = selection_passed and holdout_passed and blocks_passed
    if not passed:
        failures = []
        if not selection_passed:
            failures.append("选参段 Log Loss 未优于均匀分布")
        if not holdout_passed:
            failures.append("独立守门段 Log Loss 未优于均匀分布")
        if not blocks_passed:
            failures.append("三个时间块未全部优于均匀分布")
        return _fallback_calibration(
            rule,
            validation_periods,
            "；".join(failures),
            selected_weight=selected_weight,
            selected_model=selected_model,
            temperature=temperature,
            selection_periods=selection_periods,
            holdout_periods=holdout_periods,
            selection_log_loss=selection_loss,
            holdout_log_loss=holdout_loss,
            block_log_losses=block_losses,
        )
    return DigitProbabilityCalibration(
        applied_learned_weight=selected_weight,
        selected_learned_weight=selected_weight,
        selected_model=selected_model,
        temperature=temperature,
        validation_periods=validation_periods,
        selection_periods=selection_periods,
        holdout_periods=holdout_periods,
        uniform_log_loss=uniform_loss,
        selection_log_loss=selection_loss,
        holdout_log_loss=holdout_loss,
        block_log_losses=block_losses,
        passed=True,
        fallback_reason=None,
    )


def _tie_key(seed: str, text: str) -> bytes:
    return hashlib.sha256(f"{seed}:{text}".encode("ascii")).digest()


def build_digit_probability_plan(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    candidate_count: int = 10,
    candidate_config: DigitCandidateConfig | None = None,
    probability_config: DigitProbabilityConfig | None = None,
    calibration: DigitProbabilityCalibration | None = None,
    stats: DigitStatisticsResult | None = None,
) -> DigitProbabilityPlan:
    """生成完整概率分布，并按概率纯 TopK 选择直选和组选。"""

    _validate_probability_rule(rule)
    if candidate_count <= 0:
        raise ValueError("候选数量必须为正整数")
    base_config = candidate_config or DigitCandidateConfig(ranking_mode="ensemble")
    _validate_complete_space(base_config)
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    fitted = calibration or fit_digit_probability_calibration(
        chronological,
        rule,
        candidate_config=base_config,
        probability_config=probability_config,
    )
    scoring = _scoring_config(base_config, len(chronological))
    current_stats = stats or analyze_digit_history(
        chronological, rule, frequency_windows=scoring.frequency_windows
    )
    pool, effective = _probability_pool(current_stats, rule, base_config)
    learned = _softmax(_profile_scores(pool, fitted.selected_model), fitted.temperature)
    uniform = 1.0 / len(pool)
    probabilities = (
        1.0 - fitted.applied_learned_weight
    ) * uniform + fitted.applied_learned_weight * learned
    probabilities /= probabilities.sum()
    probability_by_text = {
        pool.text_at(index): float(probabilities[index]) for index in range(len(pool))
    }
    source_issue = str(current_stats.latest_issue)
    ordered_indexes = sorted(
        range(len(pool)),
        key=lambda index: (
            -float(probabilities[index]),
            _tie_key(source_issue, pool.text_at(index)),
        ),
    )
    direct_candidates = []
    for index in ordered_indexes[:candidate_count]:
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
    for text, probability in probability_by_text.items():
        key = "".join(sorted(text))
        if len(set(key)) == 1:
            continue
        group_mass[key] = group_mass.get(key, 0.0) + probability
        group_permutations[key] += 1
    group_keys = sorted(
        group_mass,
        key=lambda key: (-group_mass[key], _tie_key(source_issue, key)),
    )[:candidate_count]
    group_candidates = [
        DigitGroupCandidate(
            group_key=key,
            numbers=[int(value) for value in key],
            shape="组三" if len(set(key)) == 2 else "组六",
            probability_mass=group_mass[key],
            score=math.log(max(group_mass[key], 1e-300)),
            permutations=group_permutations[key],
            predicted_probability=group_mass[key],
            ranking_model="exact_permutation_probability_v2",
        )
        for key in group_keys
    ]
    distribution = DigitProbabilityDistribution(
        probabilities=probability_by_text,
        calibration=fitted,
        probability_sum=float(probabilities.sum()),
        squared_probability_sum=float(np.square(probabilities).sum()),
    )
    plan_config = replace(effective, count=candidate_count, ranking_mode="probability")
    return DigitProbabilityPlan(
        rule_code=rule.code,
        display_name=rule.display_name,
        direct_candidates=direct_candidates,
        group_candidates=group_candidates,
        config=plan_config,
        distribution=distribution,
        model_candidates={},
    )


__all__ = [
    "DigitProbabilityCalibration",
    "DigitProbabilityConfig",
    "DigitProbabilityDistribution",
    "DigitProbabilityPlan",
    "build_digit_probability_plan",
    "fit_digit_probability_calibration",
]
