# -*- coding: utf-8 -*-

from dataclasses import dataclass

from src.analysis.betting_plan import (
    build_digit_betting_plan,
)


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
