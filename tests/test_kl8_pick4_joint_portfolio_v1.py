# -*- coding: utf-8 -*-
"""KL8 Pick4 条件 Poisson 联合概率与固定分区搜索测试。"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import hypergeom

from src.analysis.kl8_pick4_joint_portfolio_v1 import (
    ConditionalPoisson20,
    best_improvement_partition,
    has_improving_pair_swap,
)
from src.analysis.kl8_pick4_rank_challenger import ranked_pick4_portfolio


def _probabilities() -> list[float]:
    raw = np.linspace(0.08, 0.42, 80, dtype=np.float64)
    return [float(value) for value in raw * (20.0 / raw.sum())]


def test_uniform_joint_ticket_pmf_equals_hypergeometric() -> None:
    model = ConditionalPoisson20([0.25] * 80)
    actual = model.ticket_hit_pmf([1, 2, 3, 4])
    expected = hypergeom.pmf(np.arange(5), 80, 20, 4)
    assert np.allclose(actual, expected, atol=1e-13, rtol=1e-13)
    assert math.isclose(math.fsum(actual), 1.0, abs_tol=1e-14)


def test_arbitrary_joint_ticket_pmf_sums_to_one() -> None:
    model = ConditionalPoisson20(_probabilities())
    pmf = model.ticket_hit_pmf([3, 17, 44, 79])
    assert all(0.0 <= value <= 1.0 for value in pmf)
    assert math.isclose(math.fsum(pmf), 1.0, abs_tol=1e-13)


def test_challenger_keeps_same_union_cost_and_is_local_optimum() -> None:
    probabilities = _probabilities()
    scores = np.linspace(1.0, 0.0, 80, dtype=np.float64)
    control_tickets = ranked_pick4_portfolio(scores)
    progress: list[dict[str, object]] = []
    control, challenger = best_improvement_partition(
        probabilities,
        control_tickets,
        progress_callback=progress.append,
    )
    assert len(control.tickets) == len(challenger.tickets) == 5
    assert all(len(ticket) == 4 for ticket in challenger.tickets)
    assert len({number for ticket in challenger.tickets for number in ticket}) == 20
    assert {number for ticket in control.tickets for number in ticket} == {
        number for ticket in challenger.tickets for number in ticket
    }
    assert challenger.objective >= control.objective
    assert not has_improving_pair_swap(probabilities, challenger.tickets)
    assert [item["iteration"] for item in progress] == list(range(1, len(progress) + 1))


def test_partition_search_is_deterministic() -> None:
    probabilities = _probabilities()
    control = ranked_pick4_portfolio(np.linspace(0.0, 1.0, 80, dtype=np.float64))
    first = best_improvement_partition(probabilities, control)[1]
    second = best_improvement_partition(probabilities, control)[1]
    assert first == second
