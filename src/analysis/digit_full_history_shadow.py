# -*- coding: utf-8 -*-
"""全历史连续预训练的稀疏v4影子状态。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.analysis.digit_daily_policy import select_daily_candidates
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
from src.analysis.digit_learned_ranker import (
    learned_ranker_source_fingerprint,
    rank_candidate_indices,
)
from src.analysis.digit_online_gradient import (
    OnlineGradientCandidate,
    OnlineGradientConfig,
    OnlineGradientSelection,
    _CandidateState,
    _initial_weights,
    _select_candidate,
    online_gradient_step,
)
from src.lotteries.base import LotteryRule

_CANDIDATES = tuple(f"{value:03d}" for value in range(1000))


@dataclass(frozen=True)
class FullHistoryShadowConfig:
    warmup_history: int = 150
    calibration_interval: int = 10
    search_lookback: int = 300
    validation_lookback: int = 100
    prospective_periods: int = 500
    learning_rates: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
    shrinkages: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    weight_half_life: float = 300.0
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
        if self.warmup_history <= 0 or self.calibration_interval <= 0:
            raise ValueError("warmup_history和calibration_interval必须大于零")
        if self.search_lookback <= 0 or self.validation_lookback <= 0:
            raise ValueError("Search/Validation窗口必须大于零")
        if self.prospective_periods <= 0 or self.weight_half_life <= 0:
            raise ValueError("前瞻期数和权重半衰期必须大于零")

    def online_config(self, history_size: int) -> OnlineGradientConfig:
        return OnlineGradientConfig(
            development_end=history_size,
            outer_periods=1,
            calibration_interval=self.calibration_interval,
            search_lookback=self.search_lookback,
            validation_lookback=self.validation_lookback,
            warmup_history=self.warmup_history,
            learning_rates=self.learning_rates,
            shrinkages=self.shrinkages,
        )


@dataclass(frozen=True)
class FullHistoryShadowResult:
    lottery: str
    training_start_index: int
    training_end_index: int
    updates_per_candidate: int
    latest_history_issue: str
    data_sha256: str
    source_fingerprint: str
    config: FullHistoryShadowConfig
    selection: OnlineGradientSelection
    candidate_states: tuple[dict[str, object], ...]
    research_top50: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "modelVersion": "learned_ranker_v4",
            "evaluationKind": "full_history_shadow_pretraining",
            "evidenceStatus": "prospective_only",
            "lottery": self.lottery,
            "trainingStartIndex": self.training_start_index,
            "trainingEndIndex": self.training_end_index,
            "updatesPerCandidate": self.updates_per_candidate,
            "latestHistoryIssue": self.latest_history_issue,
            "dataSha256": self.data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "historicalReplacementFrozenAllowed": False,
            "formalPredictionActivated": False,
            "config": {
                "warmupHistory": self.config.warmup_history,
                "calibrationInterval": self.config.calibration_interval,
                "searchLookback": self.config.search_lookback,
                "validationLookback": self.config.validation_lookback,
                "prospectivePeriods": self.config.prospective_periods,
                "learningRates": list(self.config.learning_rates),
                "shrinkages": list(self.config.shrinkages),
                "weightHalfLife": self.config.weight_half_life,
                "featureWindows": list(self.config.feature_config.windows),
            },
            "currentSelection": self.selection.to_dict(),
            "candidateStates": list(self.candidate_states),
            "nextPrediction": {
                "historyEndIssue": self.latest_history_issue,
                "researchTop50": list(self.research_top50),
                "formalRecommendation": None,
                "shadowOnly": True,
            },
            "prospectiveValidation": {
                "status": "collecting",
                "requiredPeriods": self.config.prospective_periods,
                "observedPeriods": 0,
                "startAfterIssue": self.latest_history_issue,
                "parameterChangesAllowed": False,
            },
        }


def _data_sha256(history: pd.DataFrame) -> str:
    return hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()


def shadow_state_sha256(payload: Mapping[str, Any]) -> str:
    """计算不包含自校验字段的影子状态指纹。"""

    document = dict(payload)
    document.pop("stateSha256", None)
    serialized = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_locked_shadow_state(
    payload: Mapping[str, Any], *, expected_lottery: str
) -> dict[str, Any]:
    """校验影子状态身份、源码版本和内容完整性。"""

    document = dict(payload)
    if document.get("modelVersion") != "learned_ranker_v4":
        raise ValueError("影子状态不是learned_ranker_v4")
    if document.get("evaluationKind") != "full_history_shadow_pretraining":
        raise ValueError("影子状态类型不正确")
    if document.get("lottery") != expected_lottery:
        raise ValueError("影子状态彩种与请求不一致")
    current_source = learned_ranker_source_fingerprint()
    if document.get("sourceFingerprint") != current_source:
        raise ValueError("影子状态源码指纹不匹配，禁止用旧状态增量预测")
    claimed_state = document.get("stateSha256")
    if not isinstance(claimed_state, str) or claimed_state != shadow_state_sha256(
        document
    ):
        raise ValueError("影子状态内容指纹不匹配，文件可能被修改或损坏")
    return document


def decay_shadow_weights(
    weights: np.ndarray,
    candidate: OnlineGradientCandidate,
    half_life: float,
    zeroed_features: tuple[str, ...],
) -> np.ndarray:
    if candidate.learning_rate <= 0:
        return weights
    decayed = weights * math.exp(-math.log(2.0) / half_life)
    for name in zeroed_features:
        decayed[FEATURE_NAMES.index(name)] = 0.0
    decayed[FEATURE_NAMES.index("constraint_penalty")] = min(
        0.0, decayed[FEATURE_NAMES.index("constraint_penalty")]
    )
    return decayed


def train_full_history_shadow(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: FullHistoryShadowConfig = FullHistoryShadowConfig(),
) -> FullHistoryShadowResult:
    if rule.code not in {"fc3d", "pl3"} or rule.draw_count != 3:
        raise ValueError("全历史稀疏v4只支持fc3d/pl3")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    required = config.search_lookback + config.validation_lookback
    if len(chronological) <= config.warmup_history + required:
        raise ValueError("历史不足以完成全历史预训练和首次校准")
    online_config = config.online_config(len(chronological))
    candidates = [
        OnlineGradientCandidate(rate, shrinkage)
        for rate in config.learning_rates
        for shrinkage in config.shrinkages
    ]
    learners = [
        _CandidateState(candidate, _initial_weights(online_config), [], [])
        for candidate in candidates
    ]
    indices = tuple(range(config.warmup_history, len(chronological)))
    states = iter_rolling_history_states(
        chronological, rule, indices, config.feature_config
    )
    selection: OnlineGradientSelection | None = None
    last_selection_index: int | None = None
    for target_index, history_state in zip(indices, states):
        if len(learners[0].log_losses) >= required and (
            selection is None
            or last_selection_index is None
            or target_index - last_selection_index >= config.calibration_interval
        ):
            selection = _select_candidate(learners, target_index, online_config)
            last_selection_index = target_index
        matrix = build_candidate_features(history_state, rule)[
            list(FEATURE_NAMES)
        ].to_numpy(dtype=float)
        row = chronological.iloc[target_index]
        actual_index = int(
            "".join(str(int(row[column])) for column in rule.number_columns)
        )
        for learner in learners:
            step = online_gradient_step(
                matrix,
                actual_index,
                learner.weights,
                learner.candidate,
                online_config,
            )
            learner.weights = decay_shadow_weights(
                step.weights_after,
                learner.candidate,
                config.weight_half_life,
                online_config.zeroed_features,
            )
            learner.log_losses.append(step.log_loss)
            learner.brier_scores.append(step.brier)
            if len(learner.log_losses) > required:
                learner.log_losses.pop(0)
                learner.brier_scores.pop(0)
    if selection is None:
        raise RuntimeError("全历史训练结束但未形成参数选择")
    next_state = build_history_state(chronological, rule, config.feature_config)
    next_matrix = build_candidate_features(next_state, rule)[
        list(FEATURE_NAMES)
    ].to_numpy(dtype=float)
    selected_index = candidates.index(selection.candidate)
    selected = learners[selected_index]
    scores = next_matrix @ selected.weights / online_config.temperature
    shifted = scores - float(scores.max())
    model_probabilities = np.exp(shifted)
    model_probabilities /= float(model_probabilities.sum())
    final_probabilities = (
        selection.candidate.uniform_shrinkage * model_probabilities
        + (1.0 - selection.candidate.uniform_shrinkage) / 1000.0
    )
    order = rank_candidate_indices(final_probabilities, _CANDIDATES)
    latest_row = chronological.iloc[-1]
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
    research_top50 = (
        select_daily_candidates(
            (_CANDIDATES[int(index)] for index in order),
            latest_exact=latest_exact,
            top_k=50,
            maximum_triples=1,
        )
        if selection.candidate.uniform_shrinkage > 0
        else ()
    )
    candidate_states = tuple(
        {
            "candidate": {
                "key": learner.candidate.key,
                "learningRate": learner.candidate.learning_rate,
                "uniformShrinkage": learner.candidate.uniform_shrinkage,
            },
            "weights": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, learner.weights)
            },
            "rollingLogLoss": list(learner.log_losses),
            "rollingBrier": list(learner.brier_scores),
        }
        for learner in learners
    )
    return FullHistoryShadowResult(
        lottery=rule.code,
        training_start_index=config.warmup_history,
        training_end_index=len(chronological),
        updates_per_candidate=len(indices),
        latest_history_issue=str(chronological.iloc[-1]["期数"]),
        data_sha256=_data_sha256(chronological),
        source_fingerprint=learned_ranker_source_fingerprint(),
        config=config,
        selection=selection,
        candidate_states=candidate_states,
        research_top50=research_top50,
    )


def write_locked_shadow_state(
    result: FullHistoryShadowResult, path: str | Path
) -> Path:
    payload = result.to_dict()
    payload["stateSha256"] = shadow_state_sha256(payload)
    return write_locked_shadow_payload(payload, path)


def write_locked_shadow_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    """以只写一次方式保存已带自校验指纹的影子状态。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    if document.get("stateSha256") != shadow_state_sha256(document):
        raise ValueError("待写入影子状态的内容指纹不匹配")
    content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o444)
    except FileExistsError:
        raise RuntimeError("影子模型锁定状态已存在，禁止覆盖") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    return destination


__all__ = [
    "FullHistoryShadowConfig",
    "FullHistoryShadowResult",
    "decay_shadow_weights",
    "shadow_state_sha256",
    "train_full_history_shadow",
    "validate_locked_shadow_state",
    "write_locked_shadow_payload",
    "write_locked_shadow_state",
]
