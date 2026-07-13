# -*- coding:utf-8 -*-
"""
共享的开奖匹配常量表：
- CASH_SELECT_LIST: 每种选号数量对应的命中组合大小序列
- CASH_PRICE_LIST: 奖金映射表，按命中数索引
供 kl8_cash.py 与 kl8_cash_plus.py 复用，避免重复维护。
"""
from __future__ import annotations
from typing import List

# 生成 0..10 的递减序列列表
CASH_SELECT_LIST: List[List[int]] = []
for i in range(0, 11):
    _t = [element for element in range(i, -1, -1)]
    CASH_SELECT_LIST.append(_t)

# 与原脚本一致的奖金表
CASH_PRICE_LIST = [
    [5000000, 8000, 800, 80, 5, 3, 0, 0, 0, 0, 2],
    [300000, 2000, 200, 20, 5, 3, 0, 0, 0, 2],
    [50000, 800, 88, 10, 3, 0, 0, 0, 2],
    [10000, 288, 28, 4, 0, 0, 0, 2],
    [3000, 30, 10, 3, 0, 0, 0],
    [1000, 21, 3, 0, 0, 0],
    [100, 5, 3, 0, 0],
    [53, 3, 0, 0],
    [19, 0, 0],
    [4.6, 0],
]
