# -*- coding: utf-8 -*-

from dataclasses import dataclass

from src.analysis.betting_plan import (
    build_kl8_betting_plan,
    build_digit_betting_plan,
    kl8_select10_compound_cost,
)


@dataclass(frozen=True)
class DummyGroup:
    numbers: list[int]
    score: float = 1.0
    odd_count: int = 5
    big_count: int = 5
    repeat_last_count: int = 0
    zone_distribution: list[int] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DummyCandidate:
    text: str
    numbers: list[int]
    score: float = 1.0
    shape: str = "组六"


@dataclass(frozen=True)
class DummyCandidateResult:
    rule_code: str
    display_name: str
    candidates: list[DummyCandidate]


def test_kl8_select10_compound_cost_controls_explosive_growth():
    assert kl8_select10_compound_cost(10).bet_count == 1
    assert kl8_select10_compound_cost(12).bet_count == 66
    assert kl8_select10_compound_cost(12).cost == 132
    assert kl8_select10_compound_cost(13).risk_level == "高"


def test_build_kl8_betting_plan_extracts_core_and_budget_plans():
    groups = [
        DummyGroup([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        DummyGroup([1, 2, 3, 4, 11, 12, 13, 14, 15, 16]),
        DummyGroup([1, 2, 5, 6, 17, 18, 19, 20, 21, 22]),
    ] + [DummyGroup([start + offset for offset in range(10)]) for start in range(23, 93, 10)]

    plan = build_kl8_betting_plan(groups, budgets=(20, 150))

    assert plan.play == "kl8_select10"
    assert plan.core_numbers[:4] == [1, 2, 3, 4]
    assert any(item.kind == "复式" and item.number_count == 12 and item.cost == 132 for item in plan.plans)
    assert any(item.kind == "单式" and item.cost == 20 for item in plan.plans)


def test_build_digit_betting_plan_returns_direct_and_group_or_position_compound():
    fc3d = DummyCandidateResult(
        "fc3d",
        "福彩3D",
        [
            DummyCandidate("123", [1, 2, 3], shape="组六"),
            DummyCandidate("133", [1, 3, 3], shape="组三"),
            DummyCandidate("263", [2, 6, 3], shape="组六"),
            DummyCandidate("662", [6, 6, 2], shape="组三"),
        ],
    )
    fc3d_plan = build_digit_betting_plan(fc3d)

    assert fc3d_plan.play == "fc3d"
    assert any(item.kind == "直选复式" for item in fc3d_plan.plans)
    assert any(item.kind == "组选复式" for item in fc3d_plan.plans)

    pl5 = DummyCandidateResult(
        "pl5",
        "排列五",
        [
            DummyCandidate("31340", [3, 1, 3, 4, 0]),
            DummyCandidate("94338", [9, 4, 3, 3, 8]),
            DummyCandidate("41281", [4, 1, 2, 8, 1]),
        ],
    )
    pl5_plan = build_digit_betting_plan(pl5)

    assert pl5_plan.play == "pl5"
    assert any(item.kind == "定位复式" and item.cost > 0 for item in pl5_plan.plans)
