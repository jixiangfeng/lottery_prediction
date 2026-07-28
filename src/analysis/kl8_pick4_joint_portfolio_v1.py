# -*- coding: utf-8 -*-
"""快乐8 Pick4 同成本联合概率票组挑战器 v1。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, cast

import numpy as np
from numpy.typing import NDArray

Ticket = tuple[int, int, int, int]
Portfolio = tuple[Ticket, Ticket, Ticket, Ticket, Ticket]
Objective = tuple[float, float, float]
ProgressCallback = Callable[[dict[str, object]], None]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TicketDistribution:
    """单张 Pick4 票在条件 Poisson 联合分布下的精确命中分布。"""

    ticket: Ticket
    hit_pmf: tuple[float, float, float, float, float]

    @property
    def exact4(self) -> float:
        """返回恰中 4 个号码的概率。"""

        return self.hit_pmf[4]

    @property
    def at_least3(self) -> float:
        """返回至少命中 3 个号码的概率。"""

        return self.hit_pmf[3] + self.hit_pmf[4]

    @property
    def at_least2(self) -> float:
        """返回至少命中 2 个号码的概率。"""

        return self.hit_pmf[2] + self.hit_pmf[3] + self.hit_pmf[4]


@dataclass(frozen=True)
class PortfolioEvaluation:
    """五张同成本 Pick4 票的逐票分布与求和目标。"""

    tickets: Portfolio
    ticket_distributions: tuple[TicketDistribution, ...]
    objective: Objective


def validate_probabilities80(probabilities80: Sequence[float]) -> FloatArray:
    """校验 v2 输出的已审计边际概率，不宣称联合模型保持这些边际。"""

    probabilities = np.asarray(probabilities80, dtype=np.float64)
    if probabilities.shape != (80,) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities80必须是80维有限向量")
    if not np.all((probabilities > 0.0) & (probabilities < 1.0)):
        raise ValueError("probabilities80必须逐项严格满足0<p<1")
    if not math.isclose(
        math.fsum(float(value) for value in probabilities), 20.0, abs_tol=1e-10
    ):
        raise ValueError("probabilities80总和必须精确为20（容许1e-10浮点误差）")
    return probabilities


def probabilities_to_odds(probabilities80: Sequence[float]) -> FloatArray:
    """将已审计概率转换为条件 Poisson 分布使用的正赔率。"""

    probabilities = validate_probabilities80(probabilities80)
    odds = probabilities / (1.0 - probabilities)
    if not np.isfinite(odds).all() or not np.all(odds > 0.0):
        raise ValueError("概率转换后的赔率必须为正有限值")
    return cast(FloatArray, odds)


def _log_elementary_symmetric(weights: FloatArray, degree: int) -> FloatArray:
    """用 log-domain DP 计算 0..degree 阶初等对称多项式。"""

    if degree < 0:
        raise ValueError("degree不得为负数")
    logs = np.full(degree + 1, -np.inf, dtype=np.float64)
    logs[0] = 0.0
    populated = 0
    for weight in weights:
        populated = min(populated + 1, degree)
        log_weight = math.log(float(weight))
        for order in range(populated, 0, -1):
            logs[order] = np.logaddexp(logs[order], logs[order - 1] + log_weight)
    return logs


def _canonical_ticket(ticket: Sequence[int]) -> Ticket:
    values = tuple(sorted(int(value) for value in ticket))
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("每张Pick4票必须恰好包含4个唯一号码")
    if values[0] < 1 or values[-1] > 80:
        raise ValueError("Pick4号码必须位于1..80")
    return values


def canonical_portfolio(tickets: Sequence[Sequence[int]]) -> Portfolio:
    """规范化并校验五张互不重叠的 Pick4 票。"""

    canonical = tuple(sorted(_canonical_ticket(ticket) for ticket in tickets))
    if len(canonical) != 5:
        raise ValueError("票组必须恰好包含5张Pick4票")
    flat = [number for ticket in canonical for number in ticket]
    if len(set(flat)) != 20:
        raise ValueError("五张Pick4票必须互不重叠并覆盖20个号码")
    return canonical


class ConditionalPoisson20:
    """精确的 20-of-80 条件 Poisson 联合分布计算器。

    输入概率仅用于构造赔率 ``w=p/(1-p)``。条件化到恰选20个号码后，
    模型不保证继续逐项保持输入边际概率。
    """

    def __init__(self, probabilities80: Sequence[float]) -> None:
        self.audited_marginals = validate_probabilities80(probabilities80)
        self.odds = probabilities_to_odds(self.audited_marginals.tolist())
        self._ticket_cache: dict[Ticket, tuple[float, float, float, float, float]] = {}
        self._log_normalizer = float(_log_elementary_symmetric(self.odds, 20)[20])
        if not math.isfinite(self._log_normalizer):
            raise ValueError("20-of-80联合分布归一化常数无效")

    def ticket_hit_pmf(
        self, ticket: Sequence[int]
    ) -> tuple[float, float, float, float, float]:
        """返回指定 Pick4 票命中 0..4 个号码的精确概率。"""

        canonical = _canonical_ticket(ticket)
        cached = self._ticket_cache.get(canonical)
        if cached is not None:
            return cached
        indices = np.asarray([number - 1 for number in canonical], dtype=np.int64)
        mask = np.ones(80, dtype=bool)
        mask[indices] = False
        ticket_esps = _log_elementary_symmetric(self.odds[indices], 4)
        rest_esps = _log_elementary_symmetric(self.odds[mask], 20)
        log_probabilities = np.asarray(
            [
                ticket_esps[hits] + rest_esps[20 - hits] - self._log_normalizer
                for hits in range(5)
            ],
            dtype=np.float64,
        )
        probabilities = np.exp(log_probabilities)
        total = math.fsum(float(value) for value in probabilities)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("票面命中分布无法归一化")
        probabilities /= total
        if not math.isclose(
            math.fsum(float(value) for value in probabilities), 1.0, abs_tol=1e-12
        ):
            raise RuntimeError("票面命中分布概率和不为1")
        result = (
            float(probabilities[0]),
            float(probabilities[1]),
            float(probabilities[2]),
            float(probabilities[3]),
            float(probabilities[4]),
        )
        self._ticket_cache[canonical] = result
        return result

    def evaluate_portfolio(
        self, tickets: Sequence[Sequence[int]]
    ) -> PortfolioEvaluation:
        """计算五票逐票 PMF 及 exact4/atLeast3/atLeast2 求和目标。"""

        portfolio = canonical_portfolio(tickets)
        distributions = tuple(
            TicketDistribution(ticket=ticket, hit_pmf=self.ticket_hit_pmf(ticket))
            for ticket in portfolio
        )
        objective = (
            math.fsum(item.exact4 for item in distributions),
            math.fsum(item.at_least3 for item in distributions),
            math.fsum(item.at_least2 for item in distributions),
        )
        return PortfolioEvaluation(portfolio, distributions, objective)


def _swapped_portfolios(portfolio: Portfolio) -> list[Portfolio]:
    candidates: set[Portfolio] = set()
    for left_index in range(4):
        for right_index in range(left_index + 1, 5):
            for left_number in portfolio[left_index]:
                for right_number in portfolio[right_index]:
                    mutable = [list(ticket) for ticket in portfolio]
                    mutable[left_index].remove(left_number)
                    mutable[left_index].append(right_number)
                    mutable[right_index].remove(right_number)
                    mutable[right_index].append(left_number)
                    candidates.add(canonical_portfolio(mutable))
    return sorted(candidates)


def best_improvement_partition(
    probabilities80: Sequence[float],
    control_tickets: Sequence[Sequence[int]],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[PortfolioEvaluation, PortfolioEvaluation]:
    """从控制票组出发，执行无参数、确定性的最佳改进两票换号搜索。"""

    distribution = ConditionalPoisson20(probabilities80)
    control = distribution.evaluate_portfolio(control_tickets)
    current = control
    iteration = 0
    while True:
        evaluated = [
            distribution.evaluate_portfolio(candidate)
            for candidate in _swapped_portfolios(current.tickets)
        ]
        best_objective = max(candidate.objective for candidate in evaluated)
        best = min(
            (
                candidate
                for candidate in evaluated
                if candidate.objective == best_objective
            ),
            key=lambda item: item.tickets,
        )
        if best.objective <= current.objective:
            break
        iteration += 1
        current = best
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "acceptedPairSwap",
                    "iteration": iteration,
                    "objective": list(current.objective),
                    "tickets": [list(ticket) for ticket in current.tickets],
                }
            )
    return control, current


def has_improving_pair_swap(
    probabilities80: Sequence[float], tickets: Sequence[Sequence[int]]
) -> bool:
    """判断当前票组是否仍存在目标严格更优的两票换号。"""

    distribution = ConditionalPoisson20(probabilities80)
    current = distribution.evaluate_portfolio(tickets)
    return any(
        distribution.evaluate_portfolio(candidate).objective > current.objective
        for candidate in _swapped_portfolios(current.tickets)
    )


def evaluation_to_dict(evaluation: PortfolioEvaluation) -> dict[str, object]:
    """将票组评估转换为稳定 JSON 结构。"""

    return {
        "tickets": [list(ticket) for ticket in evaluation.tickets],
        "ticketHitPmfs": [
            {"ticket": list(item.ticket), "hitPmf0To4": list(item.hit_pmf)}
            for item in evaluation.ticket_distributions
        ],
        "objective": {
            "sumExact4": evaluation.objective[0],
            "sumAtLeast3": evaluation.objective[1],
            "sumAtLeast2": evaluation.objective[2],
        },
    }


__all__ = [
    "ConditionalPoisson20",
    "PortfolioEvaluation",
    "TicketDistribution",
    "best_improvement_partition",
    "canonical_portfolio",
    "evaluation_to_dict",
    "has_improving_pair_swap",
    "probabilities_to_odds",
    "validate_probabilities80",
]
