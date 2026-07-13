# -*- coding: utf-8 -*-
"""快乐8玩法规则。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule

KL8_RULE = LotteryRule(
    code="kl8",
    display_name="快乐8",
    category="keno",
    source_name="cwl.gov.cn",
    draw_count=20,
    default_pick_count=10,
    ball_specs=tuple(BallSpec(f"红球_{idx}", 1, 80) for idx in range(1, 21)),
    allow_repeated=False,
    prize_mode="kl8_select10",
    description="1-80 中开奖 20 个号码；当前推荐主玩法为选十。",
)
