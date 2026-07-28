# -*- coding: utf-8 -*-
"""双色球、大乐透、快乐8玩法注册表。"""

from __future__ import annotations

from typing import TypeAlias

from src.lotteries.base import BallSpec, LotteryRule, validate_numbers
from src.lotteries.dlt import DLT_RULE, DLTRule
from src.lotteries.kl8 import KL8_RULE
from src.lotteries.ssq import SSQ_RULE, SSQRule

RegisteredRule: TypeAlias = LotteryRule | SSQRule | DLTRule

LOTTERY_RULES: dict[str, RegisteredRule] = {
    rule.code: rule for rule in (SSQ_RULE, DLT_RULE, KL8_RULE)
}


def get_lottery_rule(code: str) -> RegisteredRule:
    """根据玩法代码获取规则。"""

    normalized = code.lower().strip()
    if normalized not in LOTTERY_RULES:
        raise ValueError(f"未知彩票玩法：{code}")
    return LOTTERY_RULES[normalized]


def list_lottery_rules() -> list[RegisteredRule]:
    """返回双色球、大乐透和快乐8规则。"""

    return list(LOTTERY_RULES.values())


__all__ = [
    "BallSpec",
    "LotteryRule",
    "RegisteredRule",
    "LOTTERY_RULES",
    "get_lottery_rule",
    "list_lottery_rules",
    "validate_numbers",
]
