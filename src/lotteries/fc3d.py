# -*- coding: utf-8 -*-
"""福彩3D玩法规则。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule

FC3D_RULE = LotteryRule(
    code="fc3d",
    display_name="福彩3D",
    category="digit",
    source_name="cwl.gov.cn",
    draw_count=3,
    default_pick_count=3,
    ball_specs=(BallSpec("百位", 0, 9), BallSpec("十位", 0, 9), BallSpec("个位", 0, 9)),
    allow_repeated=True,
    prize_mode="fc3d_direct_group",
    description="三位数字型玩法，支持直选、组三、组六等统计口径。",
)
