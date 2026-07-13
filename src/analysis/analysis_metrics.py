# -*- coding:utf-8 -*-
"""
通用分析指标函数：供 kl8_analysis 与 kl8_analysis_plus 复用
- 冷热号
- 奇偶比
- 分组概率
- 连号统计

注意：该模块不依赖全局 args/limit_line/ori_numpy，调用方需显式传参。
"""
from __future__ import annotations
from typing import List, Tuple


def cal_hot_cold(draws, begin: int, end: int) -> Tuple[List[int], List[int]]:
    """计算前10热号与后10冷号。
    draws: numpy.ndarray 或 list[list[int]]，每行第0列为期号，其余列为号码
    begin/end: 统计区间（闭区间左、开区间右，等价于 range(begin, end)）
    """
    balls = [0] * 81
    total_balls = 0
    n = len(draws)
    for i in range(begin, end):
        if i >= n:
            break
        row = draws[i]
        for j in range(1, min(len(row), 21)):
            total_balls += 1
            balls[row[j]] += 1
    if total_balls == 0:
        return [], []
    scored = [(i, round(balls[i] / total_balls, 5)) for i in range(1, 81)]
    scored.sort(key=lambda x: x[1], reverse=True)
    ordered = [item[0] for item in scored]
    return ordered[:10], ordered[-10:]


def cal_ball_parity(draws, limit: int) -> Tuple[float, float]:
    """计算奇偶比 (odd_ratio, even_ratio)。"""
    odd = 0
    even = 0
    for i in range(min(limit, len(draws))):
        row = draws[i]
        for j in range(1, min(len(row), 21)):
            if row[j] % 2 == 0:
                even += 1
            else:
                odd += 1
    total = odd + even
    if total == 0:
        return 0.0, 0.0
    return odd / total, even / total


def cal_ball_group(draws, limit: int) -> List[float]:
    """将 1..80 号分为 8 组，各组出现概率。"""
    group = [0] * 8
    for i in range(min(limit, len(draws))):
        row = draws[i]
        for j in range(1, min(len(row), 21)):
            group_index = (row[j] - 1) // 10
            group[group_index] += 1
    total = sum(group)
    if total == 0:
        return [0.0] * 8
    return [g / total for g in group]


def analysis_consecutive_number(draws, limit: int) -> List[float]:
    """统计连号长度的出现概率（长度下标为索引）。
    返回长度为号码数（通常为21列含期号，故返回长度 >= 21，以实际行列长度为准）。
    """
    try:
        from .shared_utils import find_consecutive_number  # 复用共享实现
    except Exception:
        from pathlib import Path
        import sys
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.analysis.shared_utils import find_consecutive_number

    total_draws = 0
    if len(draws) == 0:
        return []
    length = len(draws[0])
    consecutive_rate_list = [0] * length
    consecutive_rate = [0.0] * length
    for i in range(min(limit, len(draws))):
        row = list(draws[i][1:length])
        row.sort()
        groups = find_consecutive_number(row)
        if groups:
            total_draws += 1
            for item in groups:
                if len(item) < len(consecutive_rate_list):
                    consecutive_rate_list[len(item)] += 1
    if total_draws > 0:
        for i in range(len(consecutive_rate)):
            consecutive_rate[i] = consecutive_rate_list[i] / total_draws
    return consecutive_rate


def cal_not_repeat_rate(
    draws,
    limit: int,
    j_shiftint: int = 1,
    result_list=None,
) -> float:
    """计算当前与上期（或第 j_shiftint 期）不重复元素相邻(±1)的概率。

    参数说明：
    - draws: 历史开奖二维数组（每行第0列为期号，1..20为号码）
    - limit: 参与统计的行数上限
    - j_shiftint: 与第 i 行比较的偏移量（默认对比 i+1 行）
    - result_list: 若提供，则以其为左操作数集合（同样的行结构），否则使用 draws 本身
    """
    total_march = 0
    march_num = 0
    base = draws
    target = draws
    if result_list is not None:
        base = result_list
        j_shiftint = j_shiftint or 1

    max_i = min(limit, len(base))
    for i in range(max_i):
        tgt_idx = i + j_shiftint
        if tgt_idx >= len(target):
            break
        left_nums = set(base[i][1:])
        right_nums = set(target[tgt_idx][1:])
        overlap = left_nums & right_nums
        for item in base[i][1:]:
            total_march += 1
            if item not in overlap and ((item + 1) in right_nums or (item - 1) in right_nums):
                march_num += 1
    if total_march == 0:
        return 0.0
    return march_num / total_march


def cal_repeat_rate(
    draws,
    limit: int,
    cal_nums: int,
    j_shiftint: int = 1,
    result_list=None,
) -> List[float]:
    """计算往期重复数量的分布概率（0..cal_nums）。

    - draws: 历史开奖二维数组（每行第0列为期号，1..20为号码）
    - limit: 参与统计的左侧样本行上限
    - cal_nums: 目标选择数量（用于标准化，当左侧行长度超过 cal_nums+1 时按 20→cal_nums 等比缩放）
    - j_shiftint: 与第 i 行比较的右侧偏移（默认 i+1）
    - result_list: 左侧集合来源（默认使用 draws）
    返回长度 cal_nums+1 的列表，索引为重复个数。
    """
    march_cal = [0] * (cal_nums + 1)
    march_rate = [0.0] * (cal_nums + 1)
    total_march = 0

    left = draws if result_list is None else result_list
    right = draws

    max_i = min(limit, len(left))
    for i in range(max_i):
        tgt_idx = i + j_shiftint
        if tgt_idx >= len(right):
            break
        left_set = set(left[i][1:])
        right_set = set(right[tgt_idx][1:])
        total_march += 1
        march_num = len(left_set & right_set)
        # 当左侧样本包含的号码数多于 cal_nums 时，做比例缩放（20 → cal_nums）
        if len(left[i]) > (cal_nums + 1):
            # 四舍五入到最近的整数
            scale = cal_nums / 20.0
            march_num = int(round(march_num * scale, 0))
        if march_num <= cal_nums:
            march_cal[march_num] += 1

    if total_march > 0:
        for i in range(cal_nums + 1):
            march_rate[i] = march_cal[i] / total_march
    return march_rate
