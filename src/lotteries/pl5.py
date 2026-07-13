# -*- coding: utf-8 -*-
"""排列五玩法规则。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule

PL5_RULE = LotteryRule(
    code="pl5",
    display_name="排列五",
    category="digit",
    source_name="sporttery.cn",
    draw_count=5,
    default_pick_count=5,
    ball_specs=(
        BallSpec("万位", 0, 9),
        BallSpec("千位", 0, 9),
        BallSpec("百位", 0, 9),
        BallSpec("十位", 0, 9),
        BallSpec("个位", 0, 9),
    ),
    allow_repeated=True,
    prize_mode="pl5_direct",
    description="体彩五位数字型玩法，适合按位置热冷、和值、跨度和形态做统计。",
)
