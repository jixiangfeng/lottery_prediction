# -*- coding: utf-8 -*-

import pytest

from src.lotteries import get_lottery_rule, list_lottery_rules
from src.lotteries.base import BallSpec, LotteryRule, validate_numbers


def test_registry_contains_primary_lotteries():
    rules = {rule.code: rule for rule in list_lottery_rules()}

    assert set(rules) >= {"kl8", "fc3d", "pl3", "pl5"}
    assert rules["kl8"].display_name == "快乐8"
    assert rules["fc3d"].display_name == "福彩3D"
    assert rules["pl3"].display_name == "排列三"
    assert rules["pl5"].display_name == "排列五"


def test_kl8_rule_matches_existing_select10_flow():
    rule = get_lottery_rule("kl8")

    assert rule.draw_count == 20
    assert rule.default_pick_count == 10
    assert rule.number_columns == [f"红球_{idx}" for idx in range(1, 21)]
    assert validate_numbers(rule, list(range(1, 21))) == list(range(1, 21))


def test_digit_lotteries_use_position_columns_and_allow_repeated_digits():
    fc3d = get_lottery_rule("fc3d")
    pl5 = get_lottery_rule("pl5")

    assert fc3d.number_columns == ["百位", "十位", "个位"]
    assert pl5.number_columns == ["万位", "千位", "百位", "十位", "个位"]
    assert validate_numbers(fc3d, [1, 1, 1]) == [1, 1, 1]
    assert validate_numbers(pl5, [0, 0, 0, 0, 0]) == [0, 0, 0, 0, 0]


def test_validate_numbers_rejects_invalid_shape_and_range():
    kl8 = get_lottery_rule("kl8")
    fc3d = get_lottery_rule("fc3d")

    with pytest.raises(ValueError, match="不允许重复"):
        validate_numbers(kl8, [1] * 20)
    with pytest.raises(ValueError, match="数量必须为"):
        validate_numbers(fc3d, [1, 2])
    with pytest.raises(ValueError, match="范围"):
        validate_numbers(fc3d, [1, 2, 10])


def test_lottery_rule_to_dict_is_frontend_friendly():
    custom = LotteryRule(
        code="demo",
        display_name="演示",
        category="digit",
        source_name="demo-source",
        draw_count=2,
        default_pick_count=2,
        ball_specs=(BallSpec("十位", 0, 9), BallSpec("个位", 0, 9)),
        allow_repeated=True,
        prize_mode="demo",
    )

    payload = custom.to_dict()

    assert payload["code"] == "demo"
    assert payload["numberColumns"] == ["十位", "个位"]
    assert payload["ballSpecs"][0]["minNumber"] == 0
