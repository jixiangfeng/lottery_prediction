# -*- coding: utf-8 -*-
"""固定基数加性分数集合分布 v1。

索引统一为零基；彩票号码到索引的转换由调用方负责。实现只依赖标准库，
以 log-domain 基本对称多项式动态规划精确归一化合法的 ``K-of-N`` 集合。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real

_NEGATIVE_INFINITY = float("-inf")


def _logaddexp(left: float, right: float) -> float:
    """不依赖 NumPy 的稳定 ``log(exp(left) + exp(right))``。"""

    if left == _NEGATIVE_INFINITY:
        return right
    if right == _NEGATIVE_INFINITY:
        return left
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(-abs(left - right)))


def _validate_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name}必须是实数")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name}必须是有限实数")
    return normalized


def _elementary_log_table(log_weights: Sequence[float], k: int) -> list[list[float]]:
    """返回所有前缀的 0..k 阶基本对称多项式对数。"""

    table = [[_NEGATIVE_INFINITY] * (k + 1) for _ in range(len(log_weights) + 1)]
    table[0][0] = 0.0
    for length, log_weight in enumerate(log_weights, start=1):
        table[length][0] = 0.0
        for degree in range(1, min(length, k) + 1):
            included = table[length - 1][degree - 1] + log_weight
            excluded = table[length - 1][degree]
            table[length][degree] = _logaddexp(excluded, included)
    return table


def _suffix_elementary_log_table(
    log_weights: Sequence[float], k: int
) -> list[list[float]]:
    """返回所有后缀的 0..k 阶基本对称多项式对数。"""

    n = len(log_weights)
    table = [[_NEGATIVE_INFINITY] * (k + 1) for _ in range(n + 1)]
    table[n][0] = 0.0
    for start in range(n - 1, -1, -1):
        table[start][0] = 0.0
        available = n - start
        for degree in range(1, min(available, k) + 1):
            included = table[start + 1][degree - 1] + log_weights[start]
            excluded = table[start + 1][degree]
            table[start][degree] = _logaddexp(excluded, included)
    return table


def _correct_cardinality_sum(probabilities: list[float], k: int) -> tuple[float, ...]:
    """消除浮点舍入造成的微小和约束误差，不改变数学定义。"""

    residual = float(k) - sum(probabilities)
    if residual == 0.0:
        return tuple(probabilities)
    if abs(residual) > 1e-9:
        raise ArithmeticError("固定基数边际概率数值误差超出容限")
    candidates = range(len(probabilities))
    if residual > 0.0:
        index = max(candidates, key=lambda item: 1.0 - probabilities[item])
    else:
        index = max(candidates, key=probabilities.__getitem__)
    probabilities[index] += residual
    return tuple(probabilities)


class FixedCardinalityDistribution:
    """大小恰为 ``k`` 的集合上的加性分数分布与均匀混合。

    ``scores`` 必须全部有限，``tau`` 必须有限且严格为正，``epsilon``
    必须位于闭区间 ``[0, 1]``。公开的集合索引为零基。
    """

    scores: tuple[float, ...]
    n: int
    k: int
    tau: float
    epsilon: float
    log_partition: float
    marginals: tuple[float, ...]
    _centered_logits: tuple[float, ...]
    _centered_log_partition: float
    _uniform_log_probability: float

    def __init__(
        self,
        scores: Sequence[float],
        k: int,
        *,
        tau: float = 1.0,
        epsilon: float = 0.0,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k必须是整数")
        normalized_scores = tuple(
            _validate_real(score, f"scores[{index}]")
            for index, score in enumerate(scores)
        )
        n = len(normalized_scores)
        normalized_k = int(k)
        if n == 0:
            raise ValueError("scores不得为空")
        if normalized_k < 1 or normalized_k > n:
            raise ValueError("k必须满足1 <= k <= N")
        normalized_tau = _validate_real(tau, "tau")
        if normalized_tau <= 0.0:
            raise ValueError("tau必须严格为正")
        normalized_epsilon = _validate_real(epsilon, "epsilon")
        if not 0.0 <= normalized_epsilon <= 1.0:
            raise ValueError("epsilon必须位于[0, 1]")

        center = max(normalized_scores)
        centered_logits = tuple(
            (score - center) / normalized_tau for score in normalized_scores
        )
        if not all(math.isfinite(value) for value in centered_logits):
            raise ValueError("scores/tau超出可表示的有限范围")
        prefix = _elementary_log_table(centered_logits, normalized_k)
        centered_log_partition = prefix[n][normalized_k]
        center_term = normalized_k * (center / normalized_tau)
        log_partition = centered_log_partition + center_term
        if not math.isfinite(log_partition):
            raise ValueError("log partition超出可表示的有限范围")
        suffix = _suffix_elementary_log_table(centered_logits, normalized_k)

        base_marginals: list[float] = []
        for index, log_weight in enumerate(centered_logits):
            excluded = _NEGATIVE_INFINITY
            for left_degree in range(normalized_k):
                right_degree = normalized_k - 1 - left_degree
                term = prefix[index][left_degree] + suffix[index + 1][right_degree]
                excluded = _logaddexp(excluded, term)
            log_marginal = log_weight + excluded - centered_log_partition
            base_marginals.append(min(1.0, max(0.0, math.exp(log_marginal))))
        base = _correct_cardinality_sum(base_marginals, normalized_k)
        uniform_marginal = normalized_k / n
        mixed = [
            (1.0 - normalized_epsilon) * probability
            + normalized_epsilon * uniform_marginal
            for probability in base
        ]

        self.scores = normalized_scores
        self.n = n
        self.k = normalized_k
        self.tau = normalized_tau
        self.epsilon = normalized_epsilon
        self.log_partition = log_partition
        self.marginals = _correct_cardinality_sum(mixed, normalized_k)
        self._centered_logits = centered_logits
        self._centered_log_partition = centered_log_partition
        self._uniform_log_probability = -math.log(math.comb(n, normalized_k))

    def log_probability(self, observed: Iterable[int]) -> float:
        """返回一个完整、唯一、零基观测集合的混合 log probability。"""

        indices = tuple(observed)
        if len(indices) != self.k:
            raise ValueError(f"观测集合必须恰含{self.k}个索引")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in indices
        ):
            raise TypeError("观测集合索引必须是整数且不得为bool")
        normalized = tuple(int(index) for index in indices)
        if len(set(normalized)) != self.k:
            raise ValueError("观测集合索引不得重复")
        if any(index < 0 or index >= self.n for index in normalized):
            raise ValueError("观测集合索引越界")

        base_log_probability = (
            sum(self._centered_logits[index] for index in normalized)
            - self._centered_log_partition
        )
        if self.epsilon == 0.0:
            return base_log_probability
        if self.epsilon == 1.0:
            return self._uniform_log_probability
        return _logaddexp(
            math.log1p(-self.epsilon) + base_log_probability,
            math.log(self.epsilon) + self._uniform_log_probability,
        )


def fixed_cardinality_log_partition(
    scores: Sequence[float], k: int, *, tau: float = 1.0
) -> float:
    """返回固定基数加性分数分布的精确 log partition。"""

    return FixedCardinalityDistribution(scores, k, tau=tau).log_partition


def fixed_cardinality_marginals(
    scores: Sequence[float],
    k: int,
    *,
    tau: float = 1.0,
    epsilon: float = 0.0,
) -> tuple[float, ...]:
    """返回每个零基元素的精确边际入选概率，其和为 ``k``。"""

    return FixedCardinalityDistribution(scores, k, tau=tau, epsilon=epsilon).marginals


def fixed_cardinality_observed_log_probability(
    scores: Sequence[float],
    observed: Iterable[int],
    k: int,
    *,
    tau: float = 1.0,
    epsilon: float = 0.0,
) -> float:
    """返回观测固定基数集合的混合 log probability。"""

    return FixedCardinalityDistribution(
        scores, k, tau=tau, epsilon=epsilon
    ).log_probability(observed)


__all__ = [
    "FixedCardinalityDistribution",
    "fixed_cardinality_log_partition",
    "fixed_cardinality_marginals",
    "fixed_cardinality_observed_log_probability",
]
