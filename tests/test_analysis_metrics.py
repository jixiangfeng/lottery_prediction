# -*- coding:utf-8 -*-
"""
analysis_metrics 单元测试：覆盖核心指标的基本行为。
"""
from __future__ import annotations

import numpy as np

from src.analysis.analysis_metrics import (
    cal_hot_cold,
    cal_ball_parity,
    cal_ball_group,
    analysis_consecutive_number,
    cal_repeat_rate,
    cal_not_repeat_rate,
)


def _make_draws(rows: int = 5) -> np.ndarray:
    # 简单构造：期号递减，号码为 1..20 的滑动窗口
    data = []
    for i in range(rows):
        issue = 2024000 - i
        row = [issue] + list(range(1 + i, 21 + i))
        data.append(row)
    return np.array(data, dtype=int)


def test_hot_cold_basic():
    draws = _make_draws(10)
    hot, cold = cal_hot_cold(draws, 0, 10)
    assert len(hot) <= 10 and len(cold) <= 10
    assert all(1 <= n <= 80 for n in hot + cold)


def test_parity_group_basic():
    draws = _make_draws(10)
    odd, even = cal_ball_parity(draws, 10)
    assert 0.0 <= odd <= 1.0 and 0.0 <= even <= 1.0
    group = cal_ball_group(draws, 10)
    assert len(group) == 8
    assert abs(sum(group) - 1.0) < 1e-6 or sum(group) == 0.0


def test_consecutive_basic():
    draws = _make_draws(5)
    rate = analysis_consecutive_number(draws, 5)
    assert isinstance(rate, list)
    assert len(rate) >= 21


def test_repeat_and_not_repeat_basic():
    draws = _make_draws(6)
    # 以第一行与第二行比较
    repeat_rate = cal_repeat_rate(draws, limit=1, cal_nums=10, j_shiftint=1)
    assert len(repeat_rate) == 11
    assert abs(sum(repeat_rate) - 1.0) < 1e-6 or sum(repeat_rate) == 0.0

    not_repeat = cal_not_repeat_rate(draws, limit=1, j_shiftint=1)
    assert 0.0 <= not_repeat <= 1.0
