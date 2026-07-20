# -*- coding: utf-8 -*-
"""三位彩统一概率、排名和校准指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    observations: int
    mean_probability: float
    empirical_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "observations": self.observations,
            "meanProbability": self.mean_probability,
            "empiricalRate": self.empirical_rate,
        }


@dataclass(frozen=True)
class UnifiedEvaluation:
    observations: int
    mean_log_loss: float
    mean_brier_score: float
    mean_rank: float
    mean_rank_percentile: float
    top_k_hit_rates: dict[int, float]
    expected_calibration_error: float
    calibration: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "meanLogLoss": self.mean_log_loss,
            "meanBrierScore": self.mean_brier_score,
            "meanRank": self.mean_rank,
            "meanRankPercentile": self.mean_rank_percentile,
            "topKHitRates": {
                str(key): value for key, value in self.top_k_hit_rates.items()
            },
            "expectedCalibrationError": self.expected_calibration_error,
            "calibration": [item.to_dict() for item in self.calibration],
        }


def _normalized_probability_matrix(
    probability_rows: Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = np.asarray(probability_rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 1000 or matrix.shape[0] == 0:
        raise ValueError("概率矩阵必须是非空的 observations×1000")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("概率必须是有限非负数")
    totals = matrix.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("每期概率总和必须大于零")
    return matrix / totals[:, None]


def evaluate_binary_calibration(
    probabilities: Sequence[float],
    outcomes: Sequence[int | bool],
    *,
    calibration_bins: int = 10,
) -> tuple[tuple[CalibrationBin, ...], float]:
    """计算二元事件的校准分箱和ECE。"""

    values = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if values.ndim != 1 or observed.shape != values.shape or values.size == 0:
        raise ValueError("校准概率和结果必须是同长度非空一维数组")
    if np.any((values < 0) | (values > 1)) or np.any((observed < 0) | (observed > 1)):
        raise ValueError("校准概率和结果必须位于0-1")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins必须大于零")
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    calibration: list[CalibrationBin] = []
    expected_calibration_error = 0.0
    for index in range(calibration_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        mask = (
            (values >= lower) & (values <= upper)
            if index == calibration_bins - 1
            else (values >= lower) & (values < upper)
        )
        count = int(mask.sum())
        if not count:
            continue
        mean_probability = float(values[mask].mean())
        empirical_rate = float(observed[mask].mean())
        expected_calibration_error += (count / values.size) * abs(
            mean_probability - empirical_rate
        )
        calibration.append(
            CalibrationBin(
                lower,
                upper,
                count,
                mean_probability,
                empirical_rate,
            )
        )
    return tuple(calibration), float(expected_calibration_error)


def evaluate_probability_history(
    probability_rows: Sequence[Sequence[float]],
    actual_indices: Sequence[int],
    *,
    top_ks: tuple[int, ...] = (10, 20, 50, 100),
    calibration_bins: int = 10,
) -> UnifiedEvaluation:
    """按统一口径评估多期1000候选概率。

    ``actual_indices`` 使用 ``000`` 到 ``999`` 的整数索引。
    """

    matrix = _normalized_probability_matrix(probability_rows)
    actual = np.asarray(actual_indices, dtype=int)
    if actual.shape != (matrix.shape[0],) or np.any((actual < 0) | (actual >= 1000)):
        raise ValueError("actual_indices数量必须匹配且位于0-999")
    if not top_ks or any(value <= 0 or value > 1000 for value in top_ks):
        raise ValueError("TopK必须位于1-1000")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins必须大于零")

    rows = np.arange(matrix.shape[0])
    actual_probabilities = matrix[rows, actual]
    ranks = np.empty(matrix.shape[0], dtype=float)
    top_hits = {value: 0 for value in top_ks}
    for row_index, probabilities in enumerate(matrix):
        actual_probability = probabilities[actual[row_index]]
        ranks[row_index] = 1.0 + float(np.sum(probabilities > actual_probability))
        ranks[row_index] += 0.5 * float(np.sum(probabilities == actual_probability) - 1)
        for top_k in top_ks:
            top_hits[top_k] += int(ranks[row_index] <= top_k)

    outcomes = np.zeros_like(matrix)
    outcomes[rows, actual] = 1.0
    brier = np.sum((matrix - outcomes) ** 2, axis=1)

    flat_probabilities = matrix.ravel()
    flat_outcomes = outcomes.ravel()
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    calibration: list[CalibrationBin] = []
    expected_calibration_error = 0.0
    for index in range(calibration_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == calibration_bins - 1:
            mask = (flat_probabilities >= lower) & (flat_probabilities <= upper)
        else:
            mask = (flat_probabilities >= lower) & (flat_probabilities < upper)
        count = int(mask.sum())
        if not count:
            continue
        mean_probability = float(flat_probabilities[mask].mean())
        empirical_rate = float(flat_outcomes[mask].mean())
        expected_calibration_error += (count / flat_probabilities.size) * abs(
            mean_probability - empirical_rate
        )
        calibration.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                observations=count,
                mean_probability=mean_probability,
                empirical_rate=empirical_rate,
            )
        )

    observations = matrix.shape[0]
    return UnifiedEvaluation(
        observations=observations,
        mean_log_loss=float(-np.mean(np.log(np.maximum(actual_probabilities, 1e-15)))),
        mean_brier_score=float(np.mean(brier)),
        mean_rank=float(ranks.mean()),
        mean_rank_percentile=float(np.mean((ranks - 1.0) / 999.0)),
        top_k_hit_rates={
            top_k: hits / observations for top_k, hits in top_hits.items()
        },
        expected_calibration_error=float(expected_calibration_error),
        calibration=tuple(calibration),
    )


def uniform_reference() -> dict[str, float]:
    """返回1000候选均匀分布的理论指标。"""

    return {
        "logLoss": math.log(1000.0),
        "brierScore": 0.999,
        "meanRank": 500.5,
        "meanRankPercentile": 0.5,
    }


__all__ = [
    "CalibrationBin",
    "UnifiedEvaluation",
    "evaluate_binary_calibration",
    "evaluate_probability_history",
    "uniform_reference",
]
