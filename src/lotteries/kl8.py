# -*- coding: utf-8 -*-
"""中国福利彩票快乐8玩法规则。"""

from __future__ import annotations

from src.lotteries.base import BallSpec, LotteryRule

KL8_SUPPORTED_PICK_COUNTS = (4, 5)

KL8_RULE = LotteryRule(
    code="kl8",
    display_name="快乐8",
    category="multi_label",
    source_name="cwl.gov.cn",
    draw_count=20,
    default_pick_count=5,
    ball_specs=(BallSpec("号码", 1, 80),),
    allow_repeated=False,
    prize_mode="kl8_pick5",
    description="每期从1至80开出20个唯一号码；本项目支持选4安全测试入口和选5开发挑战器。",
)
