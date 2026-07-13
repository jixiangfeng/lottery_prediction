# -*- coding: utf-8 -*-
"""排列三玩法规则。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule

PL3_RULE = LotteryRule(
    code="pl3",
    display_name="排列三",
    category="digit",
    source_name="sporttery.cn",
    draw_count=3,
    default_pick_count=3,
    ball_specs=(BallSpec("百位", 0, 9), BallSpec("十位", 0, 9), BallSpec("个位", 0, 9)),
    allow_repeated=True,
    prize_mode="pl3_direct_group",
    description="体彩三位数字型玩法，支持位置分布、和值、跨度、组三组六分析。",
)
