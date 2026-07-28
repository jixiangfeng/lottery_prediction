# -*- coding: utf-8 -*-
"""DLT v1 四候选原始 ridge-logit 分数与严格区块前向预测。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Protocol, Sequence

import numpy as np

C1_LONG_RIDGE = "C1_LONG_RIDGE"
C2_MULTISCALE_RIDGE = "C2_MULTISCALE_RIDGE"
C3_PAIR_GRAPH_RIDGE = "C3_PAIR_GRAPH_RIDGE"
C4_EQUAL_LOGPOOL = "C4_EQUAL_LOGPOOL"
CANDIDATE_IDS = (
    C1_LONG_RIDGE,
    C2_MULTISCALE_RIDGE,
    C3_PAIR_GRAPH_RIDGE,
    C4_EQUAL_LOGPOOL,
)

FRONT_SIZE = 35
BACK_SIZE = 12
BLOCK_SIZE = 25
RIDGE_L2 = 10.0
SOLVER = "deterministic_newton_raphson"
RANDOM_SEED = 0
_MAX_NEWTON_ITERATIONS = 30
_NEWTON_TOLERANCE = 1e-10

_C1_NAMES = ("frequency500", "ewma365", "gap365", "sqrt_gap365")
_C2_NAMES = (
    "frequency20",
    "frequency60",
    "frequency200",
    "ewma30",
    "ewma120",
    "gap200",
    "frequency20_minus_frequency200",
    "ewma30_minus_ewma120",
)
_GRAPH_NAMES = (
    "graph_weighted_degree",
    "graph_mean_cooccurrence",
    "graph_pagerank_0_85_50",
    "graph_degree_minus_zone_average",
)


class DLTDrawLike(Protocol):
    """建模所需的最小开奖接口。"""

    @property
    def front(self) -> Sequence[int]:
        """前区开奖号码。"""
        ...

    @property
    def back(self) -> Sequence[int]:
        """后区开奖号码。"""
        ...


@dataclass(frozen=True)
class CandidateFeatures:
    """在一个历史前缀末端、按号码升序排列的候选特征。"""

    names: tuple[str, ...]
    front: tuple[tuple[float, ...], ...]
    back: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CandidateScores:
    """前区 35 个、后区 12 个未经校准的 logit。"""

    front: tuple[float, ...]
    back: tuple[float, ...]


@dataclass(frozen=True)
class BlockPrediction:
    """一个目标行的严格区块前向分数及其独占式拟合截止点。"""

    target_index: int
    fit_cutoff: int
    candidate_id: str
    scores: CandidateScores


def _validate_candidate(candidate_id: str) -> None:
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"未知 DLT 候选：{candidate_id}")


def _validated_zone(values: Sequence[int], size: int, count: int) -> tuple[int, ...]:
    normalized = tuple(values)
    if (
        len(normalized) != count
        or len(set(normalized)) != count
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in normalized
        )
        or any(value < 1 or value > size for value in normalized)
    ):
        raise ValueError("DLT 开奖号码不符合固定分区")
    return normalized


def _validated_history(
    draw_prefix: Sequence[DLTDrawLike],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return tuple(
        (
            _validated_zone(draw.front, FRONT_SIZE, 5),
            _validated_zone(draw.back, BACK_SIZE, 2),
        )
        for draw in draw_prefix
    )


def _frequency(history: Sequence[tuple[int, ...]], number: int, window: int) -> float:
    selected = history[-window:]
    if not selected:
        return 0.0
    return sum(number in values for values in selected) / len(selected)


def _ewma(history: Sequence[tuple[int, ...]], number: int, half_life: int) -> float:
    if not history:
        return 0.0
    weighted = 0.0
    weight_sum = 0.0
    for age, values in enumerate(reversed(history)):
        weight = 2.0 ** (-age / half_life)
        weight_sum += weight
        if number in values:
            weighted += weight
    return weighted / weight_sum


def _gap(history: Sequence[tuple[int, ...]], number: int, cap: int) -> float:
    for gap, values in enumerate(reversed(history)):
        if number in values:
            return float(min(gap, cap))
    return float(min(len(history), cap))


def _base_rows(
    history: Sequence[tuple[int, ...]], size: int, candidate_id: str
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for number in range(1, size + 1):
        if candidate_id == C1_LONG_RIDGE:
            gap = _gap(history, number, 365)
            rows.append(
                (
                    _frequency(history, number, 500),
                    _ewma(history, number, 365),
                    gap,
                    math.sqrt(gap),
                )
            )
        else:
            frequency20 = _frequency(history, number, 20)
            frequency60 = _frequency(history, number, 60)
            frequency200 = _frequency(history, number, 200)
            ewma30 = _ewma(history, number, 30)
            ewma120 = _ewma(history, number, 120)
            rows.append(
                (
                    frequency20,
                    frequency60,
                    frequency200,
                    ewma30,
                    ewma120,
                    _gap(history, number, 200),
                    frequency20 - frequency200,
                    ewma30 - ewma120,
                )
            )
    return tuple(rows)


def _graph_rows(
    history: Sequence[tuple[int, ...]], size: int
) -> tuple[tuple[float, ...], ...]:
    adjacency = [[0.0] * size for _ in range(size)]
    for values in history[-200:]:
        for left, right in combinations(values, 2):
            adjacency[left - 1][right - 1] += 1.0
            adjacency[right - 1][left - 1] += 1.0
    degrees = [sum(row) for row in adjacency]
    ranks = [1.0 / size] * size
    damping = 0.85
    for _ in range(50):
        updated = [(1.0 - damping) / size] * size
        dangling = sum(
            ranks[index] for index, degree in enumerate(degrees) if not degree
        )
        dangling_share = damping * dangling / size
        for target in range(size):
            updated[target] += dangling_share
        for source, degree in enumerate(degrees):
            if degree:
                scale = damping * ranks[source] / degree
                for target, weight in enumerate(adjacency[source]):
                    updated[target] += scale * weight
        ranks = updated
    zone_average = sum(degrees) / size
    denominator = size - 1
    return tuple(
        (
            degrees[index],
            degrees[index] / denominator,
            ranks[index],
            degrees[index] - zone_average,
        )
        for index in range(size)
    )


def _features_from_validated(
    history: Sequence[tuple[tuple[int, ...], tuple[int, ...]]], candidate_id: str
) -> CandidateFeatures:
    front_history = tuple(item[0] for item in history)
    back_history = tuple(item[1] for item in history)
    front = _base_rows(front_history, FRONT_SIZE, candidate_id)
    back = _base_rows(back_history, BACK_SIZE, candidate_id)
    if candidate_id == C3_PAIR_GRAPH_RIDGE:
        front_graph = _graph_rows(front_history, FRONT_SIZE)
        back_graph = _graph_rows(back_history, BACK_SIZE)
        front = tuple(base + graph for base, graph in zip(front, front_graph))
        back = tuple(base + graph for base, graph in zip(back, back_graph))
        return CandidateFeatures(_C2_NAMES + _GRAPH_NAMES, front, back)
    return CandidateFeatures(
        _C1_NAMES if candidate_id == C1_LONG_RIDGE else _C2_NAMES, front, back
    )


def build_candidate_features(
    draw_prefix: Sequence[DLTDrawLike], candidate_id: str
) -> CandidateFeatures:
    """仅用给定前缀计算下一目标行的冻结候选特征。"""

    _validate_candidate(candidate_id)
    if candidate_id == C4_EQUAL_LOGPOOL:
        raise ValueError("C4 是三个原始 logit 的均值，不定义独立特征")
    return _features_from_validated(_validated_history(draw_prefix), candidate_id)


def _standardizer(
    rows: Sequence[tuple[float, ...]], feature_count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not rows:
        return (0.0,) * feature_count, (0.0,) * feature_count
    means = tuple(
        sum(row[column] for row in rows) / len(rows) for column in range(feature_count)
    )
    variances = tuple(
        sum((row[column] - means[column]) ** 2 for row in rows) / len(rows)
        for column in range(feature_count)
    )
    scales = tuple(math.sqrt(value) if value > 0.0 else 0.0 for value in variances)
    return means, scales


def _standardize(
    row: tuple[float, ...], means: tuple[float, ...], scales: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(
        0.0 if scale == 0.0 else (value - mean) / scale
        for value, mean, scale in zip(row, means, scales)
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] += 1e-8
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [augmented[row][-1] for row in range(size)]


def _fit_ridge_logit_reference(
    rows: Sequence[tuple[float, ...]], labels: Sequence[int]
) -> tuple[float, ...]:
    if not rows:
        return (0.0,)
    width = len(rows[0]) + 1
    coefficients = [0.0] * width
    design = [(1.0,) + row for row in rows]
    for _ in range(_MAX_NEWTON_ITERATIONS):
        gradient = [0.0] * width
        hessian = [[0.0] * width for _ in range(width)]
        for row, label in zip(design, labels):
            probability = _sigmoid(
                sum(
                    value * coefficient for value, coefficient in zip(row, coefficients)
                )
            )
            residual = probability - label
            curvature = max(probability * (1.0 - probability), 1e-12)
            for left in range(width):
                gradient[left] += residual * row[left]
                for right in range(left + 1):
                    hessian[left][right] += curvature * row[left] * row[right]
        for index in range(1, width):
            gradient[index] += RIDGE_L2 * coefficients[index]
            hessian[index][index] += RIDGE_L2
        for left in range(width):
            for right in range(left):
                hessian[right][left] = hessian[left][right]
        step = _solve(hessian, gradient)
        coefficients = [value - change for value, change in zip(coefficients, step)]
        if max(abs(change) for change in step) < _NEWTON_TOLERANCE:
            break
    return tuple(coefficients)


def _fit_ridge_logit(
    rows: Sequence[tuple[float, ...]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[float, ...]:
    """用确定性 NumPy Newton/IRLS 拟合不惩罚截距的 ridge logit。"""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.size == 0:
        return (0.0,)
    if matrix.ndim != 2:
        raise ValueError("ridge 设计矩阵必须是二维")
    response = np.asarray(labels, dtype=np.float64)
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.zeros(design.shape[1], dtype=np.float64)
    penalty[1:] = RIDGE_L2
    for _ in range(_MAX_NEWTON_ITERATIONS):
        logits = np.clip(design @ coefficients, -700.0, 700.0)
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponential = np.exp(logits[~positive])
        probabilities[~positive] = exponential / (1.0 + exponential)
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        gradient = design.T @ (probabilities - response) + penalty * coefficients
        hessian = (design.T * curvature) @ design
        hessian.flat[:: hessian.shape[0] + 1] += penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            repaired = hessian.copy()
            repaired.flat[:: repaired.shape[0] + 1] += 1e-8
            step = np.linalg.solve(repaired, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < _NEWTON_TOLERANCE:
            break
    return tuple(float(value) for value in coefficients)


def _fit_zone(
    snapshots: Sequence[tuple[tuple[float, ...], ...]],
    labels_by_draw: Sequence[tuple[int, ...]],
    prediction_rows: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    if not snapshots:
        return (0.0,) * len(prediction_rows)
    feature_count = len(prediction_rows[0])
    rows = [row for snapshot in snapshots for row in snapshot]
    labels = [
        int(number in winning)
        for snapshot, winning in zip(snapshots, labels_by_draw)
        for number in range(1, len(snapshot) + 1)
    ]
    means, scales = _standardizer(rows, feature_count)
    standardized = [_standardize(row, means, scales) for row in rows]
    coefficients = _fit_ridge_logit(standardized, labels)
    return tuple(
        coefficients[0]
        + sum(
            coefficient * value
            for coefficient, value in zip(
                coefficients[1:], _standardize(row, means, scales)
            )
        )
        for row in prediction_rows
    )


def _indicator_matrix(history: Sequence[tuple[int, ...]], size: int) -> np.ndarray:
    result = np.zeros((len(history), size), dtype=np.float64)
    for index, winning in enumerate(history):
        result[index, np.asarray(winning, dtype=np.int64) - 1] = 1.0
    return result


def _rolling_frequency(cumulative: np.ndarray, target: int, window: int) -> np.ndarray:
    start = max(0, target - window)
    count = target - start
    if count == 0:
        return np.zeros(cumulative.shape[1], dtype=np.float64)
    return (cumulative[target] - cumulative[start]) / count


def _precompute_zone_features(
    history: Sequence[tuple[int, ...]], size: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """一次前向扫描生成目标 0..T 的全部无泄漏特征快照。"""

    indicators = _indicator_matrix(history, size)
    total = len(history)
    cumulative = np.vstack(
        (np.zeros((1, size), dtype=np.float64), np.cumsum(indicators, axis=0))
    )
    c1 = np.empty((total + 1, size, len(_C1_NAMES)), dtype=np.float64)
    c2 = np.empty((total + 1, size, len(_C2_NAMES)), dtype=np.float64)
    c3 = np.empty(
        (total + 1, size, len(_C2_NAMES) + len(_GRAPH_NAMES)), dtype=np.float64
    )
    gaps = np.zeros(size, dtype=np.float64)
    adjacency = np.zeros((size, size), dtype=np.float64)
    ewma_numerators = {
        half_life: np.zeros(size, dtype=np.float64) for half_life in (30, 120, 365)
    }
    ewma_denominators = {half_life: 0.0 for half_life in (30, 120, 365)}
    decays = {half_life: 2.0 ** (-1.0 / half_life) for half_life in (30, 120, 365)}
    for target in range(total + 1):
        f20 = _rolling_frequency(cumulative, target, 20)
        f60 = _rolling_frequency(cumulative, target, 60)
        f200 = _rolling_frequency(cumulative, target, 200)
        f500 = _rolling_frequency(cumulative, target, 500)

        def ewma(half_life: int) -> np.ndarray:
            denominator = ewma_denominators[half_life]
            if denominator == 0.0:
                return np.zeros(size, dtype=np.float64)
            return ewma_numerators[half_life] / denominator

        e30, e120, e365 = ewma(30), ewma(120), ewma(365)
        gap365 = np.minimum(gaps, 365.0)
        gap200 = np.minimum(gaps, 200.0)
        c1[target] = np.column_stack((f500, e365, gap365, np.sqrt(gap365)))
        base = np.column_stack(
            (f20, f60, f200, e30, e120, gap200, f20 - f200, e30 - e120)
        )
        c2[target] = base
        degrees = adjacency.sum(axis=1)
        ranks = np.full(size, 1.0 / size, dtype=np.float64)
        nonzero = degrees > 0.0
        transition = np.zeros_like(adjacency)
        transition[nonzero] = adjacency[nonzero] / degrees[nonzero, None]
        for _ in range(50):
            dangling = float(ranks[~nonzero].sum())
            ranks = (
                (1.0 - 0.85) / size
                + 0.85 * dangling / size
                + 0.85 * (ranks @ transition)
            )
        average = float(degrees.mean())
        graph = np.column_stack(
            (degrees, degrees / (size - 1), ranks, degrees - average)
        )
        c3[target] = np.column_stack((base, graph))
        if target == total:
            continue
        if target >= 200:
            old = np.flatnonzero(indicators[target - 200])
            adjacency[np.ix_(old, old)] -= 1.0
            adjacency[old, old] += 1.0
        selected = np.flatnonzero(indicators[target])
        adjacency[np.ix_(selected, selected)] += 1.0
        adjacency[selected, selected] -= 1.0
        for half_life, decay in decays.items():
            ewma_numerators[half_life] *= decay
            ewma_numerators[half_life] += indicators[target]
            ewma_denominators[half_life] = decay * ewma_denominators[half_life] + 1.0
        gaps += 1.0
        gaps[selected] = 0.0
    return {
        C1_LONG_RIDGE: c1,
        C2_MULTISCALE_RIDGE: c2,
        C3_PAIR_GRAPH_RIDGE: c3,
    }, indicators


class FeatureSnapshotCache:
    """仅消费 ``draws[0:stop_index]`` 的不可变、前缀安全特征缓存。"""

    def __init__(self, draws: Sequence[DLTDrawLike], *, stop_index: int) -> None:
        if stop_index < 0:
            raise ValueError("stop_index 不得为负")
        history = tuple(
            (
                _validated_zone(draws[index].front, FRONT_SIZE, 5),
                _validated_zone(draws[index].back, BACK_SIZE, 2),
            )
            for index in range(stop_index)
        )
        self.stop_index = stop_index
        self.history = history
        self._front, self.front_labels = _precompute_zone_features(
            tuple(item[0] for item in history), FRONT_SIZE
        )
        self._back, self.back_labels = _precompute_zone_features(
            tuple(item[1] for item in history), BACK_SIZE
        )

    def features(self, candidate_id: str, target_index: int) -> CandidateFeatures:
        _validate_candidate(candidate_id)
        if candidate_id == C4_EQUAL_LOGPOOL:
            raise ValueError("C4 是三个原始 logit 的均值，不定义独立特征")
        if not 0 <= target_index <= self.stop_index:
            raise ValueError("缓存目标索引越界")
        names: tuple[str, ...] = (
            _C1_NAMES if candidate_id == C1_LONG_RIDGE else _C2_NAMES
        )
        if candidate_id == C3_PAIR_GRAPH_RIDGE:
            names += _GRAPH_NAMES
        return CandidateFeatures(
            names,
            tuple(
                tuple(float(value) for value in row)
                for row in self._front[candidate_id][target_index]
            ),
            tuple(
                tuple(float(value) for value in row)
                for row in self._back[candidate_id][target_index]
            ),
        )


def _fit_cached_zone(
    snapshots: np.ndarray, labels: np.ndarray, prediction_rows: np.ndarray
) -> tuple[float, ...]:
    if snapshots.shape[0] == 0:
        return (0.0,) * prediction_rows.shape[0]
    rows = snapshots.reshape(-1, snapshots.shape[-1])
    response = labels.reshape(-1)
    means = rows.mean(axis=0)
    scales = rows.std(axis=0)
    standardized = np.divide(
        rows - means, scales, out=np.zeros_like(rows), where=scales != 0.0
    )
    coefficients = np.asarray(_fit_ridge_logit(standardized, response))
    prediction = np.divide(
        prediction_rows - means,
        scales,
        out=np.zeros_like(prediction_rows),
        where=scales != 0.0,
    )
    return tuple(
        float(value) for value in coefficients[0] + prediction @ coefficients[1:]
    )


def fit_candidate_scores_from_cache(
    cache: FeatureSnapshotCache, fit_cutoff: int, candidate_id: str
) -> CandidateScores:
    """只消费 ``[0, fit_cutoff)`` 标签并用截止点快照预测。"""

    _validate_candidate(candidate_id)
    if not 0 <= fit_cutoff <= cache.stop_index:
        raise ValueError("拟合截止点越过缓存")
    if candidate_id == C4_EQUAL_LOGPOOL:
        components = tuple(
            fit_candidate_scores_from_cache(cache, fit_cutoff, item)
            for item in CANDIDATE_IDS[:3]
        )
        return CandidateScores(
            tuple(
                sum(item.front[index] for item in components) / 3.0
                for index in range(FRONT_SIZE)
            ),
            tuple(
                sum(item.back[index] for item in components) / 3.0
                for index in range(BACK_SIZE)
            ),
        )
    return CandidateScores(
        _fit_cached_zone(
            cache._front[candidate_id][:fit_cutoff],
            cache.front_labels[:fit_cutoff],
            cache._front[candidate_id][fit_cutoff],
        ),
        _fit_cached_zone(
            cache._back[candidate_id][:fit_cutoff],
            cache.back_labels[:fit_cutoff],
            cache._back[candidate_id][fit_cutoff],
        ),
    )


def _fit_base_candidate(
    history: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], candidate_id: str
) -> CandidateScores:
    snapshots = [
        _features_from_validated(history[:target_index], candidate_id)
        for target_index in range(len(history))
    ]
    prediction = _features_from_validated(history, candidate_id)
    return CandidateScores(
        _fit_zone(
            [snapshot.front for snapshot in snapshots],
            [item[0] for item in history],
            prediction.front,
        ),
        _fit_zone(
            [snapshot.back for snapshot in snapshots],
            [item[1] for item in history],
            prediction.back,
        ),
    )


def fit_candidate_scores(
    draw_prefix: Sequence[DLTDrawLike], candidate_id: str
) -> CandidateScores:
    """只用 ``draw_prefix`` 拟合并返回下一行的前/后区原始 logit。"""

    _validate_candidate(candidate_id)
    cache = FeatureSnapshotCache(draw_prefix, stop_index=len(draw_prefix))
    return fit_candidate_scores_from_cache(cache, len(draw_prefix), candidate_id)


def iter_block_walk_forward_predictions(
    draws: Sequence[DLTDrawLike],
    candidate_id: str,
    *,
    start_index: int = 0,
    stop_index: int | None = None,
) -> Iterator[BlockPrediction]:
    """按全局零基目标索引每 25 行重拟合，并在区块内冻结同一分数。"""

    _validate_candidate(candidate_id)
    end = len(draws) if stop_index is None else stop_index
    if start_index < 0 or end < start_index or end > len(draws):
        raise ValueError("walk-forward 目标索引范围非法")
    active_cutoff: int | None = None
    scores: CandidateScores | None = None
    for target_index in range(start_index, end):
        fit_cutoff = target_index - target_index % BLOCK_SIZE
        if fit_cutoff != active_cutoff:
            scores = fit_candidate_scores(draws[:fit_cutoff], candidate_id)
            active_cutoff = fit_cutoff
        if scores is None:  # pragma: no cover - 由非空 range 与上面分支保证
            raise RuntimeError("区块分数未初始化")
        yield BlockPrediction(target_index, fit_cutoff, candidate_id, scores)


def predict_block_walk_forward(
    draws: Sequence[DLTDrawLike],
    candidate_id: str,
    *,
    start_index: int = 0,
    stop_index: int | None = None,
) -> tuple[BlockPrediction, ...]:
    """返回 :func:`iter_block_walk_forward_predictions` 的不可变结果。"""

    return tuple(
        iter_block_walk_forward_predictions(
            draws, candidate_id, start_index=start_index, stop_index=stop_index
        )
    )
