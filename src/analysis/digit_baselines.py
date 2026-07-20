# -*- coding: utf-8 -*-
"""三位彩独立基线层；不依赖 learned ranker 参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_statistics import classify_digit_shape
from src.lotteries.base import LotteryRule

SHAPE_PRIORS: Mapping[str, float] = {"组六": 0.72, "组三": 0.27, "豹子": 0.01}
SHAPE_COUNTS: Mapping[str, int] = {"组六": 720, "组三": 270, "豹子": 10}
_CANDIDATE_DIGITS = np.asarray(
    [(number // 100, (number // 10) % 10, number % 10) for number in range(1000)],
    dtype=int,
)
_CANDIDATE_SHAPES = tuple(
    classify_digit_shape(tuple(int(value) for value in row))
    for row in _CANDIDATE_DIGITS
)


@dataclass(frozen=True)
class BaselinePrediction:
    name: str
    probabilities: tuple[float, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "probabilities": list(self.probabilities),
            "metadata": self.metadata,
        }


def _history_digits(
    history: pd.DataFrame, rule: LotteryRule, *, window: int
) -> np.ndarray:
    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("基线层只支持fc3d/pl3")
    if window <= 0:
        raise ValueError("window必须大于零")
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True).tail(
        window
    )
    if chronological.empty:
        raise ValueError("基线至少需要一期历史")
    return chronological[list(rule.number_columns)].to_numpy(dtype=int)


def _prediction(
    name: str, probabilities: np.ndarray, metadata: dict[str, object]
) -> BaselinePrediction:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (1000,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("基线概率必须是1000个有限非负数")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("基线概率总和必须大于零")
    values /= total
    return BaselinePrediction(name, tuple(float(value) for value in values), metadata)


def uniform_baseline() -> BaselinePrediction:
    """所有直选候选等概率。"""

    return _prediction(
        "uniform", np.full(1000, 0.001), {"candidateSpace": 1000, "window": None}
    )


def position_frequency_baseline(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    window: int = 150,
    alpha: float = 2.0,
) -> BaselinePrediction:
    """最近窗口位置独立频率基线。"""

    if alpha <= 0:
        raise ValueError("alpha必须大于零")
    digits = _history_digits(history, rule, window=window)
    position_probabilities = np.empty((3, 10), dtype=float)
    for position in range(3):
        counts = np.bincount(digits[:, position], minlength=10).astype(float)
        position_probabilities[position] = (counts + alpha / 10.0) / (
            len(digits) + alpha
        )
    probabilities = np.prod(
        position_probabilities[np.arange(3)[:, None], _CANDIDATE_DIGITS.T], axis=0
    )
    return _prediction(
        "position_frequency",
        probabilities,
        {"window": int(len(digits)), "alpha": alpha, "independentPositions": True},
    )


def shape_transition_baseline(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    window: int = 150,
    smoothing: float = 20.0,
) -> BaselinePrediction:
    """形态转移基线，后验向72/27/1理论先验收缩。"""

    if smoothing <= 0:
        raise ValueError("smoothing必须大于零")
    digits = _history_digits(history, rule, window=window)
    shapes = [
        classify_digit_shape(tuple(int(value) for value in row)) for row in digits
    ]
    previous = shapes[-1]
    transition_targets = [
        right for left, right in zip(shapes[:-1], shapes[1:]) if left == previous
    ]
    posterior = {
        name: (transition_targets.count(name) + smoothing * prior)
        / (len(transition_targets) + smoothing)
        for name, prior in SHAPE_PRIORS.items()
    }
    probabilities = np.asarray(
        [posterior[shape] / SHAPE_COUNTS[shape] for shape in _CANDIDATE_SHAPES],
        dtype=float,
    )
    return _prediction(
        "shape_transition",
        probabilities,
        {
            "window": int(len(digits)),
            "previousShape": previous,
            "transitionObservations": len(transition_targets),
            "smoothing": smoothing,
            "shapeProbabilities": posterior,
            "theoreticalShapePrior": dict(SHAPE_PRIORS),
        },
    )


def shape_prior_baseline() -> BaselinePrediction:
    """按72/27/1理论形态质量分配到每个直选候选。"""

    probabilities = np.asarray(
        [SHAPE_PRIORS[shape] / SHAPE_COUNTS[shape] for shape in _CANDIDATE_SHAPES],
        dtype=float,
    )
    return _prediction(
        "shape_prior",
        probabilities,
        {"shapeProbabilities": dict(SHAPE_PRIORS), "source": "enumeration"},
    )


def sum_span_baseline(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    window: int = 150,
    alpha: float = 2.0,
) -> BaselinePrediction:
    """最近窗口和值与跨度联合朴素基线。"""

    if alpha <= 0:
        raise ValueError("alpha必须大于零")
    digits = _history_digits(history, rule, window=window)
    sums = digits.sum(axis=1)
    spans = digits.max(axis=1) - digits.min(axis=1)
    sum_probabilities = (np.bincount(sums, minlength=28) + alpha / 28.0) / (
        len(digits) + alpha
    )
    span_probabilities = (np.bincount(spans, minlength=10) + alpha / 10.0) / (
        len(digits) + alpha
    )
    candidate_sums = _CANDIDATE_DIGITS.sum(axis=1)
    candidate_spans = _CANDIDATE_DIGITS.max(axis=1) - _CANDIDATE_DIGITS.min(axis=1)
    probabilities = (
        sum_probabilities[candidate_sums] * span_probabilities[candidate_spans]
    )
    return _prediction(
        "sum_span",
        probabilities,
        {"window": int(len(digits)), "alpha": alpha},
    )


def build_baseline_suite(
    history: pd.DataFrame, rule: LotteryRule
) -> dict[str, BaselinePrediction]:
    """构建三层设计固定基线矩阵。"""

    total_window = len(history)
    suite = {
        "uniform": uniform_baseline(),
        "shape_prior": shape_prior_baseline(),
        "shape_transition_150": shape_transition_baseline(history, rule, window=150),
        "sum_span_150": sum_span_baseline(history, rule, window=150),
    }
    for window in (20, 50, 150, total_window):
        name = "position_all" if window == total_window else f"position_{window}"
        prediction = position_frequency_baseline(history, rule, window=window)
        suite[name] = BaselinePrediction(
            name, prediction.probabilities, prediction.metadata
        )
    return suite


__all__ = [
    "BaselinePrediction",
    "SHAPE_COUNTS",
    "SHAPE_PRIORS",
    "build_baseline_suite",
    "position_frequency_baseline",
    "shape_prior_baseline",
    "shape_transition_baseline",
    "sum_span_baseline",
    "uniform_baseline",
]
