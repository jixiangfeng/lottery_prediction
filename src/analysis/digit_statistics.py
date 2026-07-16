# -*- coding: utf-8 -*-
"""数字型彩票通用统计模块。

适用于福彩3D、排列三、排列五这类“按位置开奖、每位 0-9、允许重复”的玩法。
当前模块只做历史统计和形态分析，不承诺预测未来开奖结果。
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.analysis.digit_data import sort_digit_dataframe_by_issue
from src.lotteries.base import LotteryRule, validate_numbers

DEFAULT_FREQUENCY_WINDOWS = (20, 50, 100, 300)
DEFAULT_BAYESIAN_PRIOR_STRENGTH = 2.0


@dataclass(frozen=True)
class DigitStatisticsResult:
    """数字彩历史统计结果。"""

    code: str
    display_name: str
    draw_count: int
    total_issues: int
    position_frequency: dict[str, Counter[int]]
    position_frequency_windows: dict[int, dict[str, Counter[int]]]
    position_probabilities: dict[int, dict[str, dict[int, float]]]
    pair_frequency_windows: dict[int, dict[str, Counter[tuple[int, int]]]]
    pair_probabilities: dict[int, dict[str, dict[tuple[int, int], float]]]
    shape_probabilities: dict[int, dict[str, float]]
    sum_probabilities: dict[int, dict[int, float]]
    span_probabilities: dict[int, dict[int, float]]
    prefix3_shape_probabilities: dict[int, dict[str, float]]
    prefix3_sum_probabilities: dict[int, dict[int, float]]
    prefix3_span_probabilities: dict[int, dict[int, float]]
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
            "positionFrequencyWindows": {
                str(window): {
                    position: dict(sorted(counter.items())) for position, counter in positions.items()
                }
                for window, positions in self.position_frequency_windows.items()
            },
            "positionProbabilities": {
                str(window): {
                    position: {str(digit): probability for digit, probability in sorted(probabilities.items())}
                    for position, probabilities in positions.items()
                }
                for window, positions in self.position_probabilities.items()
            },
            "pairFrequencyWindows": {
                str(window): {
                    pair: {f"{left},{right}": count for (left, right), count in sorted(counter.items())}
                    for pair, counter in pairs.items()
                }
                for window, pairs in self.pair_frequency_windows.items()
            },
            "pairProbabilities": {
                str(window): {
                    pair: {
                        f"{left},{right}": probability
                        for (left, right), probability in sorted(probabilities.items())
                    }
                    for pair, probabilities in pairs.items()
                }
                for window, pairs in self.pair_probabilities.items()
            },
            "shapeProbabilities": {
                str(window): dict(sorted(probabilities.items()))
                for window, probabilities in self.shape_probabilities.items()
            },
            "sumProbabilities": {
                str(window): {str(value): probability for value, probability in sorted(probabilities.items())}
                for window, probabilities in self.sum_probabilities.items()
            },
            "spanProbabilities": {
                str(window): {str(value): probability for value, probability in sorted(probabilities.items())}
                for window, probabilities in self.span_probabilities.items()
            },
            "prefix3ShapeProbabilities": {
                str(window): dict(sorted(probabilities.items()))
                for window, probabilities in self.prefix3_shape_probabilities.items()
            },
            "prefix3SumProbabilities": {
                str(window): {str(value): probability for value, probability in sorted(probabilities.items())}
                for window, probabilities in self.prefix3_sum_probabilities.items()
            },
            "prefix3SpanProbabilities": {
                str(window): {str(value): probability for value, probability in sorted(probabilities.items())}
                for window, probabilities in self.prefix3_span_probabilities.items()
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
    return sort_digit_dataframe_by_issue(df, ascending=False)


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


def _multi_window_position_statistics(
    sorted_df: pd.DataFrame,
    rule: LotteryRule,
    *,
    windows: Sequence[int],
    prior_strength: float,
) -> tuple[dict[int, dict[str, Counter[int]]], dict[int, dict[str, dict[int, float]]]]:
    """计算多窗口位置频率与贝叶斯平滑概率。"""

    frequency_windows: dict[int, dict[str, Counter[int]]] = {}
    probability_windows: dict[int, dict[str, dict[int, float]]] = {}
    for window in windows:
        window_size = max(1, int(window))
        recent = sorted_df.head(window_size)
        frequency_windows[window_size] = {}
        probability_windows[window_size] = {}
        for column, spec in zip(rule.number_columns, rule.ball_specs):
            counter = Counter(int(value) for value in recent[column].tolist())
            frequency_windows[window_size][column] = counter
            digit_count = spec.max_number - spec.min_number + 1
            prior_per_digit = prior_strength / digit_count
            denominator = len(recent) + prior_strength
            probability_windows[window_size][column] = {
                digit: (counter.get(digit, 0) + prior_per_digit) / denominator
                for digit in range(spec.min_number, spec.max_number + 1)
            }
    return frequency_windows, probability_windows


def _smoothed_probabilities(
    values: Sequence[Any],
    domain: Sequence[Any],
    prior_strength: float,
) -> dict[Any, float]:
    """对离散分布应用对称 Dirichlet 先验平滑。"""

    normalized_domain = tuple(domain)
    if not normalized_domain:
        return {}
    counter = Counter(values)
    prior_per_value = prior_strength / len(normalized_domain)
    denominator = len(values) + prior_strength
    if denominator <= 0:
        uniform = 1.0 / len(normalized_domain)
        return {value: uniform for value in normalized_domain}
    return {
        value: (counter.get(value, 0) + prior_per_value) / denominator
        for value in normalized_domain
    }


def _shape_domain(draw_count: int) -> tuple[str, ...]:
    if draw_count == 3:
        return ("豹子", "组三", "组六")
    if draw_count == 5:
        return ("五同", "四一", "三二", "三一一", "二二一", "二一一一", "全不同")
    return ()


def _multi_window_joint_statistics(
    sorted_df: pd.DataFrame,
    rule: LotteryRule,
    *,
    windows: Sequence[int],
    prior_strength: float,
) -> tuple[
    dict[int, dict[str, Counter[tuple[int, int]]]],
    dict[int, dict[str, dict[tuple[int, int], float]]],
    dict[int, dict[str, float]],
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
    dict[int, dict[str, float]],
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
]:
    """计算位置对与结构特征的多窗口平滑分布。"""

    pair_frequencies: dict[int, dict[str, Counter[tuple[int, int]]]] = {}
    pair_probabilities: dict[int, dict[str, dict[tuple[int, int], float]]] = {}
    shape_probabilities: dict[int, dict[str, float]] = {}
    sum_probabilities: dict[int, dict[int, float]] = {}
    span_probabilities: dict[int, dict[int, float]] = {}
    prefix3_shape_probabilities: dict[int, dict[str, float]] = {}
    prefix3_sum_probabilities: dict[int, dict[int, float]] = {}
    prefix3_span_probabilities: dict[int, dict[int, float]] = {}
    pair_domain = tuple(itertools.product(range(10), repeat=2))

    for window in windows:
        window_size = max(1, int(window))
        recent = sorted_df.head(window_size)
        rows = _number_rows(recent, rule)
        pair_frequencies[window_size] = {}
        pair_probabilities[window_size] = {}
        for left_index in range(rule.draw_count):
            for right_index in range(left_index + 1, rule.draw_count):
                key = f"{left_index}-{right_index}"
                values = [(row[left_index], row[right_index]) for row in rows]
                pair_frequencies[window_size][key] = Counter(values)
                pair_probabilities[window_size][key] = _smoothed_probabilities(
                    values,
                    pair_domain,
                    prior_strength,
                )

        shapes = [classify_digit_shape(row) for row in rows]
        sums = [sum(row) for row in rows]
        spans = [max(row) - min(row) for row in rows]
        shape_probabilities[window_size] = _smoothed_probabilities(
            shapes,
            _shape_domain(rule.draw_count),
            prior_strength,
        )
        sum_probabilities[window_size] = _smoothed_probabilities(
            sums,
            tuple(range(rule.draw_count * 9 + 1)),
            prior_strength,
        )
        span_probabilities[window_size] = _smoothed_probabilities(spans, tuple(range(10)), prior_strength)

        prefix_rows = [row[:3] for row in rows]
        prefix3_shape_probabilities[window_size] = _smoothed_probabilities(
            [classify_digit_shape(row) for row in prefix_rows],
            _shape_domain(3),
            prior_strength,
        )
        prefix3_sum_probabilities[window_size] = _smoothed_probabilities(
            [sum(row) for row in prefix_rows],
            tuple(range(28)),
            prior_strength,
        )
        prefix3_span_probabilities[window_size] = _smoothed_probabilities(
            [max(row) - min(row) for row in prefix_rows],
            tuple(range(10)),
            prior_strength,
        )

    return (
        pair_frequencies,
        pair_probabilities,
        shape_probabilities,
        sum_probabilities,
        span_probabilities,
        prefix3_shape_probabilities,
        prefix3_sum_probabilities,
        prefix3_span_probabilities,
    )


def analyze_digit_history(
    df: pd.DataFrame,
    rule: LotteryRule,
    *,
    frequency_windows: Sequence[int] = DEFAULT_FREQUENCY_WINDOWS,
    bayesian_prior_strength: float = DEFAULT_BAYESIAN_PRIOR_STRENGTH,
) -> DigitStatisticsResult:
    """统计数字型彩票历史开奖形态。

    ``frequency_windows`` 与 ``bayesian_prior_strength`` 可用于配置近期窗口和
    贝叶斯平滑强度；旧调用无需传参即可保持兼容。
    """

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
    position_frequency_windows, position_probabilities = _multi_window_position_statistics(
        sorted_df,
        rule,
        windows=frequency_windows,
        prior_strength=max(0.0, float(bayesian_prior_strength)),
    )
    (
        pair_frequency_windows,
        pair_probabilities,
        shape_probabilities,
        sum_probabilities,
        span_probabilities,
        prefix3_shape_probabilities,
        prefix3_sum_probabilities,
        prefix3_span_probabilities,
    ) = _multi_window_joint_statistics(
        sorted_df,
        rule,
        windows=frequency_windows,
        prior_strength=max(0.0, float(bayesian_prior_strength)),
    )
    return DigitStatisticsResult(
        code=rule.code,
        display_name=rule.display_name,
        draw_count=rule.draw_count,
        total_issues=len(rows),
        position_frequency=position_frequency,
        position_frequency_windows=position_frequency_windows,
        position_probabilities=position_probabilities,
        pair_frequency_windows=pair_frequency_windows,
        pair_probabilities=pair_probabilities,
        shape_probabilities=shape_probabilities,
        sum_probabilities=sum_probabilities,
        span_probabilities=span_probabilities,
        prefix3_shape_probabilities=prefix3_shape_probabilities,
        prefix3_sum_probabilities=prefix3_sum_probabilities,
        prefix3_span_probabilities=prefix3_span_probabilities,
        current_omission=current_digit_omission(sorted_df, rule) if rows else {},
        sum_distribution=sum_distribution,
        span_distribution=span_distribution,
        shape_distribution=shape_distribution,
        parity_distribution=parity_distribution,
        big_small_distribution=big_small_distribution,
        latest_issue=latest_issue,
        latest_numbers=latest_numbers,
    )
