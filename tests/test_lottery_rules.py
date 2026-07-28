# -*- coding: utf-8 -*-

import pytest

from src.lotteries import get_lottery_rule, list_lottery_rules
from src.lotteries.base import BallSpec, LotteryRule
from src.lotteries.dlt import DLT_RULE
from src.lotteries.ssq import SSQ_RULE


def test_registry_contains_only_retained_lotteries():
    rules = {rule.code: rule for rule in list_lottery_rules()}

    assert set(rules) == {"ssq", "dlt", "kl8"}
    assert rules["ssq"].display_name == "双色球"
    assert rules["dlt"].display_name == "超级大乐透"
    assert rules["kl8"].display_name == "快乐8"


def test_retained_zone_rules_validate_draws():
    assert SSQ_RULE.validate_draw((1, 2, 3, 4, 5, 6), 1) == (
        (1, 2, 3, 4, 5, 6),
        1,
    )
    assert DLT_RULE.validate_draw((1, 2, 3, 4, 5), (1, 2)) == (
        (1, 2, 3, 4, 5),
        (1, 2),
    )
    with pytest.raises(ValueError, match="未知彩票玩法"):
        get_lottery_rule("removed")


def test_lottery_rule_to_dict_is_frontend_friendly():
    custom = LotteryRule(
        code="demo",
        display_name="演示",
        category="multi_label",
        source_name="demo-source",
        draw_count=2,
        default_pick_count=2,
        ball_specs=(BallSpec("号码", 1, 9),),
        allow_repeated=False,
        prize_mode="demo",
    )

    payload = custom.to_dict()

    assert payload["code"] == "demo"
    assert payload["numberColumns"] == ["号码"]
    assert payload["ballSpecs"][0]["minNumber"] == 1
