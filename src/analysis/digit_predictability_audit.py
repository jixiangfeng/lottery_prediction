# -*- coding: utf-8 -*-
"""三位彩开发区可预测性审计：时序置换与简单基线逐期对照。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.lotteries.base import LotteryRule

_CANDIDATES = np.asarray(
    [(value // 100, (value // 10) % 10, value % 10) for value in range(1000)],
    dtype=int,
)
_CANDIDATE_SUMS = _CANDIDATES.sum(axis=1)
_CANDIDATE_SPANS = _CANDIDATES.max(axis=1) - _CANDIDATES.min(axis=1)
_CANDIDATE_SHAPES = np.where(
    (_CANDIDATES[:, 0] == _CANDIDATES[:, 1]) & (_CANDIDATES[:, 1] == _CANDIDATES[:, 2]),
    2,
    np.where(
        (_CANDIDATES[:, 0] == _CANDIDATES[:, 1])
        | (_CANDIDATES[:, 0] == _CANDIDATES[:, 2])
        | (_CANDIDATES[:, 1] == _CANDIDATES[:, 2]),
        1,
        0,
    ),
)
_UNIFORM_LOG_LOSS = math.log(1000.0)
_UNIFORM_BRIER = 0.999


@dataclass(frozen=True)
class PredictabilityAuditConfig:
    min_train_size: int = 150
    permutation_trials: int = 499
    block_size: int = 10
    seed: int = 20260719
    fdr_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.min_train_size < 20:
            raise ValueError("min_train_size至少为20")
        if self.permutation_trials < 19:
            raise ValueError("permutation_trials至少为19")
        if self.block_size <= 0:
            raise ValueError("block_size必须大于零")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError("fdr_alpha必须位于0和1之间")


@dataclass(frozen=True)
class SequenceTest:
    name: str
    statistic: float
    p_value: float
    q_value: float
    passed_fdr: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "pValue": self.p_value,
            "qValue": self.q_value,
            "passedFdr": self.passed_fdr,
        }


@dataclass(frozen=True)
class BaselineAudit:
    name: str
    observations: int
    mean_log_loss: float
    mean_brier_score: float
    mean_rank: float
    top50_hit_rate: float
    log_loss_improvement: float
    brier_improvement: float
    log_loss_p_value: float
    brier_p_value: float
    log_loss_q_value: float
    brier_q_value: float
    passed_fdr: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "observations": self.observations,
            "meanLogLoss": self.mean_log_loss,
            "meanBrierScore": self.mean_brier_score,
            "meanRank": self.mean_rank,
            "top50HitRate": self.top50_hit_rate,
            "logLossImprovementVsUniform": self.log_loss_improvement,
            "brierImprovementVsUniform": self.brier_improvement,
            "logLossPValue": self.log_loss_p_value,
            "brierPValue": self.brier_p_value,
            "logLossQValue": self.log_loss_q_value,
            "brierQValue": self.brier_q_value,
            "passedFdr": self.passed_fdr,
        }


@dataclass(frozen=True)
class _RawBaseline:
    name: str
    observations: int
    mean_log_loss: float
    mean_brier_score: float
    mean_rank: float
    top50_hit_rate: float
    log_loss_improvement: float
    brier_improvement: float
    log_p: float
    brier_p: float


@dataclass(frozen=True)
class PredictabilityAuditReport:
    lottery: str
    observations: int
    evaluated_targets: int
    frozen_test_read: bool
    config: PredictabilityAuditConfig
    sequence_tests: tuple[SequenceTest, ...]
    baselines: tuple[BaselineAudit, ...]
    predictable_signal_found: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "modelVersion": "learned_ranker_v4",
            "auditKind": "development_predictability",
            "lottery": self.lottery,
            "observations": self.observations,
            "evaluatedTargets": self.evaluated_targets,
            "frozenTestRead": self.frozen_test_read,
            "config": {
                "minTrainSize": self.config.min_train_size,
                "permutationTrials": self.config.permutation_trials,
                "blockSize": self.config.block_size,
                "seed": self.config.seed,
                "fdrAlpha": self.config.fdr_alpha,
            },
            "uniformReference": {
                "meanLogLoss": _UNIFORM_LOG_LOSS,
                "meanBrierScore": _UNIFORM_BRIER,
                "meanRank": 500.5,
                "top50HitRate": 0.05,
            },
            "sequenceTests": [item.to_dict() for item in self.sequence_tests],
            "baselines": [item.to_dict() for item in self.baselines],
            "predictableSignalFound": self.predictable_signal_found,
            "decision": (
                "发现经FDR校正且在LogLoss/Brier同时优于均匀的简单信号"
                if self.predictable_signal_found
                else "未发现可复现预测信号；保持均匀基线和研究状态"
            ),
        }


def _shape_codes(digits: np.ndarray) -> np.ndarray:
    return np.where(
        (digits[:, 0] == digits[:, 1]) & (digits[:, 1] == digits[:, 2]),
        2,
        np.where(
            (digits[:, 0] == digits[:, 1])
            | (digits[:, 0] == digits[:, 2])
            | (digits[:, 1] == digits[:, 2]),
            1,
            0,
        ),
    )


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    pairs, counts = np.unique(
        np.column_stack((left, right)), axis=0, return_counts=True
    )
    total = float(counts.sum())
    left_values, left_counts = np.unique(left, return_counts=True)
    right_values, right_counts = np.unique(right, return_counts=True)
    left_probability = dict(zip(left_values.tolist(), (left_counts / total).tolist()))
    right_probability = dict(
        zip(right_values.tolist(), (right_counts / total).tolist())
    )
    result = 0.0
    for pair, count in zip(pairs, counts):
        joint = float(count / total)
        result += joint * math.log(
            joint / (left_probability[int(pair[0])] * right_probability[int(pair[1])])
        )
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return abs(float(np.corrcoef(left, right)[0, 1]))


def _permutation_p_value(
    values: np.ndarray,
    lag: int,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    rng: np.random.Generator,
    trials: int,
) -> tuple[float, float]:
    observed = statistic(values[:-lag], values[lag:])
    exceedances = 0
    for _ in range(trials):
        shuffled = rng.permutation(values)
        exceedances += int(statistic(shuffled[:-lag], shuffled[lag:]) >= observed)
    return observed, (exceedances + 1) / (trials + 1)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        original_index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, p_values[original_index] * len(order) / rank)
        adjusted[original_index] = running
    return [float(min(1.0, value)) for value in adjusted]


def _sequence_tests(
    digits: np.ndarray, config: PredictabilityAuditConfig
) -> tuple[SequenceTest, ...]:
    sequences: list[
        tuple[str, np.ndarray, Callable[[np.ndarray, np.ndarray], float]]
    ] = [
        ("百位", digits[:, 0], _mutual_information),
        ("十位", digits[:, 1], _mutual_information),
        ("个位", digits[:, 2], _mutual_information),
        ("和值", digits.sum(axis=1), _correlation),
        ("跨度", digits.max(axis=1) - digits.min(axis=1), _correlation),
        ("形态", _shape_codes(digits), _mutual_information),
    ]
    rng = np.random.default_rng(config.seed)
    raw: list[tuple[str, float, float]] = []
    for name, values, statistic in sequences:
        for lag in (1, 2, 5):
            observed, p_value = _permutation_p_value(
                values, lag, statistic, rng, config.permutation_trials
            )
            raw.append((f"{name}_lag{lag}", observed, p_value))
    q_values = _benjamini_hochberg([item[2] for item in raw])
    return tuple(
        SequenceTest(name, statistic, p_value, q_value, q_value <= config.fdr_alpha)
        for (name, statistic, p_value), q_value in zip(raw, q_values)
    )


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    total = float(probabilities.sum())
    if total <= 0 or not np.isfinite(probabilities).all():
        raise ValueError("审计基线概率无效")
    return probabilities / total


def _position_probability(
    history: np.ndarray, window: int, alpha: float = 2.0
) -> np.ndarray:
    recent = history[-window:]
    per_position = np.empty((3, 10), dtype=float)
    for position in range(3):
        counts = np.bincount(recent[:, position], minlength=10).astype(float)
        per_position[position] = (counts + alpha / 10.0) / (len(recent) + alpha)
    return _normalize_probability(
        np.prod(per_position[np.arange(3)[:, None], _CANDIDATES.T], axis=0)
    )


def _omission_probability(history: np.ndarray, window: int = 150) -> np.ndarray:
    recent = history[-window:]
    per_position = np.empty((3, 10), dtype=float)
    for position in range(3):
        gaps = np.empty(10, dtype=float)
        reversed_values = recent[::-1, position]
        for digit in range(10):
            matches = np.flatnonzero(reversed_values == digit)
            gaps[digit] = float(matches[0] + 1 if len(matches) else len(recent) + 1)
        per_position[position] = gaps / gaps.sum()
    return _normalize_probability(
        np.prod(per_position[np.arange(3)[:, None], _CANDIDATES.T], axis=0)
    )


def _markov_probability(
    history: np.ndarray, window: int = 150, alpha: float = 5.0
) -> np.ndarray:
    recent = history[-window:]
    per_position = np.empty((3, 10), dtype=float)
    for position in range(3):
        previous = recent[-1, position]
        left = recent[:-1, position]
        right = recent[1:, position]
        counts = np.bincount(right[left == previous], minlength=10).astype(float)
        per_position[position] = (counts + alpha / 10.0) / (counts.sum() + alpha)
    return _normalize_probability(
        np.prod(per_position[np.arange(3)[:, None], _CANDIDATES.T], axis=0)
    )


def _shape_transition_probability(history: np.ndarray, window: int = 150) -> np.ndarray:
    shapes = _shape_codes(history[-window:])
    previous = shapes[-1]
    targets = shapes[1:][shapes[:-1] == previous]
    prior = np.asarray([0.72, 0.27, 0.01])
    counts = np.bincount(targets, minlength=3).astype(float)
    shape_probability = (counts + 20.0 * prior) / (len(targets) + 20.0)
    shape_counts = np.asarray([720.0, 270.0, 10.0])
    return _normalize_probability(
        shape_probability[_CANDIDATE_SHAPES] / shape_counts[_CANDIDATE_SHAPES]
    )


def _sum_span_probability(history: np.ndarray, window: int = 150) -> np.ndarray:
    recent = history[-window:]
    sums = recent.sum(axis=1)
    spans = recent.max(axis=1) - recent.min(axis=1)
    sum_probability = (np.bincount(sums, minlength=28) + 2.0 / 28.0) / (
        len(recent) + 2.0
    )
    span_probability = (np.bincount(spans, minlength=10) + 2.0 / 10.0) / (
        len(recent) + 2.0
    )
    return _normalize_probability(
        sum_probability[_CANDIDATE_SUMS] * span_probability[_CANDIDATE_SPANS]
    )


def _paired_block_p_value(
    differences: np.ndarray, rng: np.random.Generator, trials: int, block_size: int
) -> float:
    blocks = [
        float(differences[start : start + block_size].mean())
        for start in range(0, len(differences), block_size)
    ]
    values = np.asarray(blocks)
    observed = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(trials, len(values)))
    null_means = np.mean(signs * values[None, :], axis=1)
    return float((1 + np.sum(null_means >= observed)) / (trials + 1))


def _baseline_audits(
    digits: np.ndarray, config: PredictabilityAuditConfig
) -> tuple[BaselineAudit, ...]:
    factories: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "position_20": lambda history: _position_probability(history, 20),
        "position_50": lambda history: _position_probability(history, 50),
        "position_150": lambda history: _position_probability(history, 150),
        "omission_150": _omission_probability,
        "markov_150": _markov_probability,
        "shape_transition_150": _shape_transition_probability,
        "sum_span_150": _sum_span_probability,
    }
    records: dict[str, dict[str, list[float]]] = {
        name: {"log": [], "brier": [], "rank": [], "hit": []} for name in factories
    }
    for target in range(config.min_train_size, len(digits)):
        history = digits[:target]
        actual = int(
            digits[target, 0] * 100 + digits[target, 1] * 10 + digits[target, 2]
        )
        for name, factory in factories.items():
            probabilities = factory(history)
            actual_probability = float(probabilities[actual])
            rank = 1.0 + float(np.sum(probabilities > actual_probability))
            rank += 0.5 * float(np.sum(probabilities == actual_probability) - 1)
            records[name]["log"].append(-math.log(max(actual_probability, 1e-15)))
            records[name]["brier"].append(
                float(np.dot(probabilities, probabilities) - 2 * actual_probability + 1)
            )
            records[name]["rank"].append(rank)
            records[name]["hit"].append(float(rank <= 50))
    rng = np.random.default_rng(config.seed + 1)
    raw: list[_RawBaseline] = []
    p_values: list[float] = []
    for name, metrics in records.items():
        log_values = np.asarray(metrics["log"])
        brier_values = np.asarray(metrics["brier"])
        log_difference = _UNIFORM_LOG_LOSS - log_values
        brier_difference = _UNIFORM_BRIER - brier_values
        log_p = _paired_block_p_value(
            log_difference, rng, config.permutation_trials, config.block_size
        )
        brier_p = _paired_block_p_value(
            brier_difference, rng, config.permutation_trials, config.block_size
        )
        p_values.extend((log_p, brier_p))
        raw.append(
            _RawBaseline(
                name=name,
                observations=len(log_values),
                mean_log_loss=float(log_values.mean()),
                mean_brier_score=float(brier_values.mean()),
                mean_rank=float(np.mean(metrics["rank"])),
                top50_hit_rate=float(np.mean(metrics["hit"])),
                log_loss_improvement=float(log_difference.mean()),
                brier_improvement=float(brier_difference.mean()),
                log_p=log_p,
                brier_p=brier_p,
            )
        )
    q_values = _benjamini_hochberg(p_values)
    output = []
    for index, values in enumerate(raw):
        log_q = q_values[index * 2]
        brier_q = q_values[index * 2 + 1]
        passed = (
            values.log_loss_improvement > 0
            and values.brier_improvement > 0
            and log_q <= config.fdr_alpha
            and brier_q <= config.fdr_alpha
        )
        output.append(
            BaselineAudit(
                name=values.name,
                observations=values.observations,
                mean_log_loss=values.mean_log_loss,
                mean_brier_score=values.mean_brier_score,
                mean_rank=values.mean_rank,
                top50_hit_rate=values.top50_hit_rate,
                log_loss_improvement=values.log_loss_improvement,
                brier_improvement=values.brier_improvement,
                log_loss_p_value=values.log_p,
                brier_p_value=values.brier_p,
                log_loss_q_value=log_q,
                brier_q_value=brier_q,
                passed_fdr=passed,
            )
        )
    return tuple(output)


def run_predictability_audit(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: PredictabilityAuditConfig = PredictabilityAuditConfig(),
) -> PredictabilityAuditReport:
    """对已截断开发区执行可预测性审计，不接受Frozen数据。"""

    if rule.code not in {"fc3d", "pl3"} or rule.draw_count != 3:
        raise ValueError("可预测性审计只支持fc3d/pl3")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    if len(chronological) <= config.min_train_size:
        raise ValueError("审计历史必须长于min_train_size")
    digits = chronological[list(rule.number_columns)].to_numpy(dtype=int)
    sequence_tests = _sequence_tests(digits, config)
    baselines = _baseline_audits(digits, config)
    predictable = any(item.passed_fdr for item in baselines)
    return PredictabilityAuditReport(
        lottery=rule.code,
        observations=len(digits),
        evaluated_targets=len(digits) - config.min_train_size,
        frozen_test_read=False,
        config=config,
        sequence_tests=sequence_tests,
        baselines=baselines,
        predictable_signal_found=predictable,
    )


def write_predictability_audit(
    report: PredictabilityAuditReport, path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "BaselineAudit",
    "PredictabilityAuditConfig",
    "PredictabilityAuditReport",
    "SequenceTest",
    "run_predictability_audit",
    "write_predictability_audit",
]
