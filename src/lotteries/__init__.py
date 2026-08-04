# -*- coding: utf-8 -*-
"""双色球玩法注册表。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule, validate_numbers
from src.lotteries.ssq import SSQ_RULE, SSQRule

RegisteredRule = SSQRule
LOTTERY_RULES: dict[str, RegisteredRule] = {SSQ_RULE.code: SSQ_RULE}


def get_lottery_rule(code: str) -> RegisteredRule:
    normalized = code.lower().strip()
    if normalized not in LOTTERY_RULES:
        raise ValueError(f"未知彩票玩法：{code}")
    return LOTTERY_RULES[normalized]


def list_lottery_rules() -> list[RegisteredRule]:
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
