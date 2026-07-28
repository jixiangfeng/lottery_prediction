# -*- coding: utf-8 -*-
"""大乐透7+2 C5：严格前序的在线Hedge log-pool核心。"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

from src.analysis.dlt_fixed_cardinality_v1 import FixedCardinalityDistribution
from src.analysis.dlt_ridge_candidates_v1 import (
    BACK_SIZE,
    C1_LONG_RIDGE,
    C2_MULTISCALE_RIDGE,
    C3_PAIR_GRAPH_RIDGE,
    FRONT_SIZE,
    CandidateScores,
    FeatureSnapshotCache,
    fit_candidate_scores_from_cache,
)

C5_ONLINE_HEDGE_LOGPOOL = "C5_ONLINE_HEDGE_LOGPOOL"
EXPERT_IDS = (C1_LONG_RIDGE, C2_MULTISCALE_RIDGE, C3_PAIR_GRAPH_RIDGE)
HEDGE_LOOKBACK = 200
HEDGE_ETA = 8.0
WEIGHT_FLOOR = 0.10
EXPERT_LOSS_TAU = 1.0
EXPERT_LOSS_EPSILON = 0.10
C5_FRONT_TAU = 1.0
C5_FRONT_EPSILON = 0.10
C5_BACK_TAU = 2.0
C5_BACK_EPSILON = 0.20


class DLTObservedLike(Protocol):
    @property
    def front(self) -> Sequence[int]: ...

    @property
    def back(self) -> Sequence[int]: ...


@dataclass(frozen=True)
class C5Prediction:
    target_offset: int
    weights: tuple[float, float, float]
    scores: CandidateScores


@dataclass(frozen=True)
class C5BlockPrediction:
    target_index: int
    fit_cutoff: int
    weights: tuple[float, float, float]
    scores: CandidateScores
    expert_scores: tuple[CandidateScores, CandidateScores, CandidateScores]


@dataclass(frozen=True)
class C5Output:
    front: tuple[int, ...]
    back: tuple[int, ...]
    tickets: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    front_marginals: tuple[float, ...]
    back_marginals: tuple[float, ...]


def _validate_observed(
    values: Sequence[int], *, size: int, count: int
) -> tuple[int, ...]:
    result = tuple(values)
    if (
        len(result) != count
        or len(set(result)) != count
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in result
        )
        or any(value < 1 or value > size for value in result)
    ):
        raise ValueError("大乐透开奖号不符合固定分区")
    return result


def normalized_expert_loss(
    scores: CandidateScores,
    observed_front: Sequence[int],
    observed_back: Sequence[int],
) -> float:
    """计算固定专家评价分布的归一化前后区集合LogLoss。"""

    front = _validate_observed(observed_front, size=FRONT_SIZE, count=5)
    back = _validate_observed(observed_back, size=BACK_SIZE, count=2)
    front_distribution = FixedCardinalityDistribution(
        scores.front, 5, tau=EXPERT_LOSS_TAU, epsilon=EXPERT_LOSS_EPSILON
    )
    back_distribution = FixedCardinalityDistribution(
        scores.back, 2, tau=EXPERT_LOSS_TAU, epsilon=EXPERT_LOSS_EPSILON
    )
    front_loss = -front_distribution.log_probability(
        tuple(value - 1 for value in front)
    )
    back_loss = -back_distribution.log_probability(tuple(value - 1 for value in back))
    return 0.5 * (
        front_loss / math.log(math.comb(FRONT_SIZE, 5))
        + back_loss / math.log(math.comb(BACK_SIZE, 2))
    )


def hedge_weights(
    loss_rows: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """从最近200期三专家损失计算带10%下限的确定性Hedge权重。"""

    if not loss_rows:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    recent = loss_rows[-HEDGE_LOOKBACK:]
    normalized_rows: list[tuple[float, float, float]] = []
    for row in recent:
        values = tuple(float(value) for value in row)
        if len(values) != len(EXPERT_IDS) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("每期必须提供三项有限专家损失")
        normalized_rows.append((values[0], values[1], values[2]))
    means = tuple(
        math.fsum(row[index] for row in normalized_rows) / len(normalized_rows)
        for index in range(len(EXPERT_IDS))
    )
    best = min(means)
    exponentials = tuple(math.exp(-HEDGE_ETA * (value - best)) for value in means)
    denominator = math.fsum(exponentials)
    free_mass = 1.0 - len(EXPERT_IDS) * WEIGHT_FLOOR
    first = WEIGHT_FLOOR + free_mass * exponentials[0] / denominator
    second = WEIGHT_FLOOR + free_mass * exponentials[1] / denominator
    third = 1.0 - first - second
    return (first, second, third)


def weighted_logpool(
    experts: Mapping[str, CandidateScores], weights: Sequence[float]
) -> CandidateScores:
    """逐号码加权平均三位专家的原始logit。"""

    values = tuple(float(value) for value in weights)
    if (
        tuple(experts) != EXPERT_IDS
        or len(values) != len(EXPERT_IDS)
        or any(not math.isfinite(value) or value < 0.0 for value in values)
        or not math.isclose(math.fsum(values), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("专家或权重不符合冻结协议")
    components = tuple(experts[expert_id] for expert_id in EXPERT_IDS)
    if any(
        len(component.front) != FRONT_SIZE or len(component.back) != BACK_SIZE
        for component in components
    ):
        raise ValueError("专家分数维度不符合大乐透分区")
    return CandidateScores(
        tuple(
            math.fsum(
                values[expert_index] * components[expert_index].front[number]
                for expert_index in range(len(EXPERT_IDS))
            )
            for number in range(FRONT_SIZE)
        ),
        tuple(
            math.fsum(
                values[expert_index] * components[expert_index].back[number]
                for expert_index in range(len(EXPERT_IDS))
            )
            for number in range(BACK_SIZE)
        ),
    )


def iter_hedge_predictions(
    score_rows: Sequence[Mapping[str, CandidateScores]],
    observed_rows: Sequence[DLTObservedLike],
) -> Iterable[C5Prediction]:
    """逐期先输出C5预测，再将当期三专家损失加入Hedge窗口。"""

    if len(score_rows) != len(observed_rows):
        raise ValueError("专家分数与开奖号行数必须一致")
    loss_rows: list[tuple[float, float, float]] = []
    for target_offset, (experts, observed) in enumerate(
        zip(score_rows, observed_rows, strict=True)
    ):
        weights = hedge_weights(loss_rows)
        yield C5Prediction(target_offset, weights, weighted_logpool(experts, weights))
        loss_rows.append(
            (
                normalized_expert_loss(
                    experts[EXPERT_IDS[0]], observed.front, observed.back
                ),
                normalized_expert_loss(
                    experts[EXPERT_IDS[1]], observed.front, observed.back
                ),
                normalized_expert_loss(
                    experts[EXPERT_IDS[2]], observed.front, observed.back
                ),
            )
        )


ScoreFitter = Callable[[FeatureSnapshotCache, int, str], CandidateScores]


def _expert_scores(
    cache: FeatureSnapshotCache,
    cutoff: int,
    score_fitter: ScoreFitter,
) -> dict[str, CandidateScores]:
    return {
        expert_id: score_fitter(cache, cutoff, expert_id) for expert_id in EXPERT_IDS
    }


def _append_losses(
    loss_rows: list[tuple[float, float, float]],
    experts: Mapping[str, CandidateScores],
    observed: DLTObservedLike,
) -> None:
    loss_rows.append(
        (
            normalized_expert_loss(
                experts[EXPERT_IDS[0]], observed.front, observed.back
            ),
            normalized_expert_loss(
                experts[EXPERT_IDS[1]], observed.front, observed.back
            ),
            normalized_expert_loss(
                experts[EXPERT_IDS[2]], observed.front, observed.back
            ),
        )
    )
    if len(loss_rows) > HEDGE_LOOKBACK:
        del loss_rows[: len(loss_rows) - HEDGE_LOOKBACK]


def walk_forward_c5_predictions(
    draws: Sequence[DLTObservedLike],
    *,
    start_index: int,
    stop_index: int,
    score_fitter: ScoreFitter = fit_candidate_scores_from_cache,
) -> tuple[C5BlockPrediction, ...]:
    """严格前序重放C5；只返回请求区间，权重最多预热200期。"""

    if start_index < 600 or stop_index < start_index or stop_index > len(draws):
        raise ValueError("C5 walk-forward区间非法")
    cache = FeatureSnapshotCache(draws, stop_index=stop_index)
    seed_start = max(600, start_index - HEDGE_LOOKBACK)
    loss_rows: list[tuple[float, float, float]] = []
    by_cutoff: dict[int, dict[str, CandidateScores]] = {}
    output: list[C5BlockPrediction] = []
    for target_index in range(seed_start, stop_index):
        fit_cutoff = target_index - target_index % 25
        if fit_cutoff not in by_cutoff:
            by_cutoff[fit_cutoff] = _expert_scores(cache, fit_cutoff, score_fitter)
        experts = by_cutoff[fit_cutoff]
        weights = hedge_weights(loss_rows)
        if target_index >= start_index:
            output.append(
                C5BlockPrediction(
                    target_index,
                    fit_cutoff,
                    weights,
                    weighted_logpool(experts, weights),
                    (
                        experts[EXPERT_IDS[0]],
                        experts[EXPERT_IDS[1]],
                        experts[EXPERT_IDS[2]],
                    ),
                )
            )
        _append_losses(loss_rows, experts, draws[target_index])
    return tuple(output)


def current_c5_prediction(
    draws: Sequence[DLTObservedLike],
    *,
    score_fitter: ScoreFitter = fit_candidate_scores_from_cache,
) -> C5BlockPrediction:
    """使用全部已结算前缀生成下一目标的C5研究预测。"""

    target_index = len(draws)
    if target_index < 600:
        raise ValueError("C5当前预测至少需要600期历史")
    cache = FeatureSnapshotCache(draws, stop_index=target_index)
    seed_start = max(600, target_index - HEDGE_LOOKBACK)
    loss_rows: list[tuple[float, float, float]] = []
    by_cutoff: dict[int, dict[str, CandidateScores]] = {}
    for observed_index in range(seed_start, target_index):
        cutoff = observed_index - observed_index % 25
        if cutoff not in by_cutoff:
            by_cutoff[cutoff] = _expert_scores(cache, cutoff, score_fitter)
        _append_losses(loss_rows, by_cutoff[cutoff], draws[observed_index])
    fit_cutoff = target_index - target_index % 25
    if fit_cutoff not in by_cutoff:
        by_cutoff[fit_cutoff] = _expert_scores(cache, fit_cutoff, score_fitter)
    experts = by_cutoff[fit_cutoff]
    weights = hedge_weights(loss_rows)
    return C5BlockPrediction(
        target_index,
        fit_cutoff,
        weights,
        weighted_logpool(experts, weights),
        (
            experts[EXPERT_IDS[0]],
            experts[EXPERT_IDS[1]],
            experts[EXPERT_IDS[2]],
        ),
    )


def build_c5_output(scores: CandidateScores) -> C5Output:
    """用冻结校准参数生成固定Top7+Top2及21张唯一票。"""

    front_distribution = FixedCardinalityDistribution(
        scores.front, 5, tau=C5_FRONT_TAU, epsilon=C5_FRONT_EPSILON
    )
    back_distribution = FixedCardinalityDistribution(
        scores.back, 2, tau=C5_BACK_TAU, epsilon=C5_BACK_EPSILON
    )
    front = tuple(
        sorted(
            range(1, FRONT_SIZE + 1),
            key=lambda number: (-front_distribution.marginals[number - 1], number),
        )[:7]
    )
    back = tuple(
        sorted(
            range(1, BACK_SIZE + 1),
            key=lambda number: (-back_distribution.marginals[number - 1], number),
        )[:2]
    )
    tickets = tuple((tuple(values), back) for values in combinations(sorted(front), 5))
    return C5Output(
        tuple(sorted(front)),
        tuple(sorted(back)),
        tickets,
        front_distribution.marginals,
        back_distribution.marginals,
    )


__all__ = [
    "C5_BACK_EPSILON",
    "C5_BACK_TAU",
    "C5_FRONT_EPSILON",
    "C5_FRONT_TAU",
    "C5_ONLINE_HEDGE_LOGPOOL",
    "C5BlockPrediction",
    "C5Output",
    "C5Prediction",
    "EXPERT_IDS",
    "HEDGE_ETA",
    "HEDGE_LOOKBACK",
    "WEIGHT_FLOOR",
    "build_c5_output",
    "current_c5_prediction",
    "hedge_weights",
    "iter_hedge_predictions",
    "normalized_expert_loss",
    "walk_forward_c5_predictions",
    "weighted_logpool",
]
