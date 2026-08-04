# -*- coding: utf-8 -*-

import pytest

from src.lotteries import get_lottery_rule, list_lottery_rules
from src.lotteries.base import BallSpec, LotteryRule
from src.lotteries.ssq import SSQ_RULE


def test_registry_contains_only_ssq() -> None:
    rules = {rule.code: rule for rule in list_lottery_rules()}
    assert set(rules) == {"ssq"}
    assert rules["ssq"].display_name == "双色球"
    with pytest.raises(ValueError, match="未知彩票玩法"):
        get_lottery_rule("removed")


def test_ssq_rule_validates_draw() -> None:
    assert SSQ_RULE.validate_draw((1, 2, 3, 4, 5, 6), 1) == ((1, 2, 3, 4, 5, 6), 1)


def test_lottery_rule_to_dict_is_frontend_friendly() -> None:
    custom = LotteryRule(
        "demo",
        "演示",
        "multi_label",
        "demo-source",
        2,
        2,
        (BallSpec("号码", 1, 9),),
        False,
        "demo",
    )
    assert custom.to_dict()["numberColumns"] == ["号码"]
