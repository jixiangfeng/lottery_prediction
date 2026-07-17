# -*- coding: utf-8 -*-
"""多彩种规则注册表。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule, validate_numbers
from src.lotteries.fc3d import FC3D_RULE
from src.lotteries.pl3 import PL3_RULE
from src.lotteries.pl5 import PL5_RULE

LOTTERY_RULES: dict[str, LotteryRule] = {
    rule.code: rule
    for rule in (
        FC3D_RULE,
        PL3_RULE,
        PL5_RULE,
    )
}


def get_lottery_rule(code: str) -> LotteryRule:
    """根据玩法代码获取规则。"""

    normalized = code.lower().strip()
    if normalized not in LOTTERY_RULES:
        raise ValueError(f"未知彩票玩法：{code}")
    return LOTTERY_RULES[normalized]


def list_lottery_rules() -> list[LotteryRule]:
    """返回已注册玩法规则。"""

    return list(LOTTERY_RULES.values())


__all__ = [
    "BallSpec",
    "LotteryRule",
    "LOTTERY_RULES",
    "get_lottery_rule",
    "list_lottery_rules",
    "validate_numbers",
]
