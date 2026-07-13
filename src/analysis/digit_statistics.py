# -*- coding: utf-8 -*-
"""数字型彩票通用统计模块。

适用于福彩3D、排列三、排列五这类“按位置开奖、每位 0-9、允许重复”的玩法。
当前模块只做历史统计和形态分析，不承诺预测未来开奖结果。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.lotteries.base import LotteryRule, validate_numbers


@dataclass(frozen=True)
class DigitStatisticsResult:
    """数字彩历史统计结果。"""

    code: str
    display_name: str
    draw_count: int
    total_issues: int
    position_frequency: dict[str, Counter[int]]
    current_omission: dict[str, dict[int, int]]
    sum_distribution: Counter[int]
    span_distribution: Counter[int]
    shape_distribution: Counter[str]
    parity_distribution: Counter[str]
    big_small_distribution: Counter[str]
    latest_issue: str
    latest_numbers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "displayName": self.display_name,
            "drawCount": self.draw_count,
            "totalIssues": self.total_issues,
            "positionFrequency": {
                position: dict(sorted(counter.items())) for position, counter in self.position_frequency.items()
            },
            "currentOmission": self.current_omission,
            "sumDistribution": dict(sorted(self.sum_distribution.items())),
            "spanDistribution": dict(sorted(self.span_distribution.items())),
            "shapeDistribution": dict(self.shape_distribution),
            "parityDistribution": dict(self.parity_distribution),
            "bigSmallDistribution": dict(self.big_small_distribution),
            "latestIssue": self.latest_issue,
            "latestNumbers": self.latest_numbers,
        }


def _sorted_history(df: pd.DataFrame) -> pd.DataFrame:
    if "期数" not in df.columns:
        raise ValueError("数字彩历史数据必须包含【期数】列")
    output = df.copy()
    output["期数"] = output["期数"].astype(str)
    return output.sort_values("期数", ascending=False).reset_index(drop=True)


def _number_rows(df: pd.DataFrame, rule: LotteryRule) -> list[list[int]]:
    missing = [column for column in rule.number_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{rule.display_name}历史数据缺少字段：{', '.join(missing)}")
    rows: list[list[int]] = []
    for _, row in df.iterrows():
        numbers = [int(row[column]) for column in rule.number_columns]
        rows.append(validate_numbers(rule, numbers))
    return rows


def classify_digit_shape(numbers: Sequence[int]) -> str:
    """识别数字彩号码形态。"""

    counter = Counter(int(number) for number in numbers)
    counts = sorted(counter.values(), reverse=True)
    if len(numbers) == 3:
        if counts == [3]:
            return "豹子"
        if counts == [2, 1]:
            return "组三"
        return "组六"
    if len(numbers) == 5:
        mapping = {
            (5,): "五同",
            (4, 1): "四一",
            (3, 2): "三二",
            (3, 1, 1): "三一一",
            (2, 2, 1): "二二一",
            (2, 1, 1, 1): "二一一一",
            (1, 1, 1, 1, 1): "全不同",
        }
        return mapping.get(tuple(counts), "未知")
    return "-".join(str(count) for count in counts)


def _parity_label(numbers: Sequence[int]) -> str:
    odd = sum(1 for number in numbers if int(number) % 2 == 1)
    even = len(numbers) - odd
    return f"奇{odd}偶{even}"


def _big_small_label(numbers: Sequence[int]) -> str:
    big = sum(1 for number in numbers if int(number) >= 5)
    small = len(numbers) - big
    return f"大{big}小{small}"


def current_digit_omission(df: pd.DataFrame, rule: LotteryRule) -> dict[str, dict[int, int]]:
    """计算每个位置 0-9 的当前遗漏。"""

    sorted_df = _sorted_history(df)
    _number_rows(sorted_df, rule)
    omission: dict[str, dict[int, int]] = {}
    for column, spec in zip(rule.number_columns, rule.ball_specs):
        values = [int(value) for value in sorted_df[column].tolist()]
        position_omission: dict[int, int] = {}
        for digit in range(spec.min_number, spec.max_number + 1):
            miss = 0
            for value in values:
                if value == digit:
                    break
                miss += 1
            position_omission[digit] = miss
        omission[column] = position_omission
    return omission


def analyze_digit_history(df: pd.DataFrame, rule: LotteryRule) -> DigitStatisticsResult:
    """统计数字型彩票历史开奖形态。"""

    if rule.category != "digit":
        raise ValueError(f"数字彩统计模块不适用于玩法：{rule.display_name}")
    sorted_df = _sorted_history(df)
    rows = _number_rows(sorted_df, rule)
    position_frequency: dict[str, Counter[int]] = {column: Counter() for column in rule.number_columns}
    sum_distribution: Counter[int] = Counter()
    span_distribution: Counter[int] = Counter()
    shape_distribution: Counter[str] = Counter()
    parity_distribution: Counter[str] = Counter()
    big_small_distribution: Counter[str] = Counter()

    for numbers in rows:
        for column, number in zip(rule.number_columns, numbers):
            position_frequency[column][number] += 1
        sum_distribution[sum(numbers)] += 1
        span_distribution[max(numbers) - min(numbers)] += 1
        shape_distribution[classify_digit_shape(numbers)] += 1
        parity_distribution[_parity_label(numbers)] += 1
        big_small_distribution[_big_small_label(numbers)] += 1

    latest_numbers = rows[0] if rows else []
    latest_issue = str(sorted_df.iloc[0]["期数"]) if not sorted_df.empty else ""
    return DigitStatisticsResult(
        code=rule.code,
        display_name=rule.display_name,
        draw_count=rule.draw_count,
        total_issues=len(rows),
        position_frequency=position_frequency,
        current_omission=current_digit_omission(sorted_df, rule) if rows else {},
        sum_distribution=sum_distribution,
        span_distribution=span_distribution,
        shape_distribution=shape_distribution,
        parity_distribution=parity_distribution,
        big_small_distribution=big_small_distribution,
        latest_issue=latest_issue,
        latest_numbers=latest_numbers,
    )
