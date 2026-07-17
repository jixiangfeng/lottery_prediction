# -*- coding: utf-8 -*-
"""三位彩固定评分算法 v4 的无泄漏特征工程。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_statistics import (
    classify_digit_shape,
    digit_consecutive_count,
    digit_mirror_count,
)
from src.lotteries.base import LotteryRule

Window = int | str
PAIR_SPECS = ((0, 1), (0, 2), (1, 2))
FEATURE_NAMES = (
    "position_frequency",
    "position_omission",
    "pair_frequency",
    "shape_distribution",
    "sum_distribution",
    "span_distribution",
    "parity_bigsmall",
    "recent_trend",
    "position_trend",
    "pair_trend",
    "sum_trend",
    "span_trend",
    "shape_trend",
    "trend_30_300",
    "trend_50_all",
    "trend_ratio_30_300",
    "position_trend_30_300",
    "pair_trend_30_300",
    "sum_trend_30_300",
    "span_trend_30_300",
    "shape_trend_30_300",
    "position_trend_50_all",
    "pair_trend_50_all",
    "sum_trend_50_all",
    "span_trend_50_all",
    "shape_trend_50_all",
    "position_trend_ratio_30_300",
    "pair_trend_ratio_30_300",
    "sum_trend_ratio_30_300",
    "span_trend_ratio_30_300",
    "shape_trend_ratio_30_300",
    "latest_distance",
    "repeat_latest",
    "omission_rebound",
    "constraint_penalty",
)


@dataclass(frozen=True)
class LearnedFeatureConfig:
    """v4 特征配置；窗口只读取目标期之前的数据。"""

    windows: tuple[Window, ...] = (10, 20, 30, 50, 100, 300, "all")
    alpha: float = 2.0
    half_life: float | None = None
    omission_cap: int = 50
    window_weights: tuple[tuple[str, float], ...] | Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("至少需要一个历史窗口")
        if any(value != "all" and int(value) <= 0 for value in self.windows):
            raise ValueError("历史窗口必须为正整数或 all")
        if self.alpha <= 0:
            raise ValueError("贝叶斯平滑 alpha 必须大于零")
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError("半衰期必须大于零或为空")
        if self.omission_cap <= 0:
            raise ValueError("遗漏上限必须大于零")
        labels = tuple(str(value) for value in self.windows)
        if len(set(labels)) != len(labels):
            raise ValueError("历史窗口不得重复")
        supplied = dict(self.window_weights or ((label, 1.0) for label in labels))
        if set(supplied) != set(labels):
            raise ValueError("窗口权重必须与 windows 一一对应")
        canonical = tuple((label, float(supplied[label])) for label in labels)
        if any(not math.isfinite(value) or value <= 0 for _, value in canonical):
            raise ValueError("窗口权重必须为有限正数")
        object.__setattr__(self, "window_weights", canonical)

    def window_weight_map(self) -> dict[str, float]:
        """返回按 canonical 窗口顺序构造的独立权重。"""

        return dict(self.window_weights or ())


@dataclass(frozen=True)
class LearnedHistoryState:
    """目标期之前的冻结历史状态。"""

    rule_code: str
    config: LearnedFeatureConfig
    target_issue: str | None
    history_issues: tuple[str, ...]
    history_end_issue: str | None
    numbers: tuple[tuple[int, int, int], ...]

    @property
    def latest_numbers(self) -> tuple[int, int, int] | None:
        return self.numbers[-1] if self.numbers else None


def decay_weight(age: int, half_life: float | None) -> float:
    """返回年龄权重；``age=1`` 代表最近一期。"""

    if age <= 0:
        raise ValueError("age 必须从 1 开始")
    if half_life is None:
        return 1.0
    if half_life <= 0:
        raise ValueError("half_life 必须大于零")
    return math.exp(-float(age) / float(half_life))


def build_history_state(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: LearnedFeatureConfig | None = None,
    *,
    target_issue: str | None = None,
) -> LearnedHistoryState:
    """构建只包含 ``target_issue`` 之前数据的历史状态。

    示例：``build_history_state(df, rule, target_issue="2026008")`` 不会读取
    2026008 及之后的任何行。
    """

    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("learned_ranker_v4 首版只支持 fc3d/pl3，pl5 请保持旧路径")
    effective = config or LearnedFeatureConfig()
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    if target_issue is not None:
        if not str(target_issue).isdigit():
            raise ValueError("目标期号必须为纯数字")
        chronological = chronological[
            chronological["期数"].astype(int) < int(target_issue)
        ]
    issues = tuple(chronological["期数"].astype(str).tolist())
    number_rows: list[tuple[int, int, int]] = []
    for _, row in chronological.iterrows():
        values = [int(row[column]) for column in rule.number_columns]
        number_rows.append((values[0], values[1], values[2]))
    rows = tuple(number_rows)
    return LearnedHistoryState(
        rule_code=rule.code,
        config=effective,
        target_issue=str(target_issue) if target_issue is not None else None,
        history_issues=issues,
        history_end_issue=issues[-1] if issues else None,
        numbers=rows,
    )


def enumerate_three_digit_candidates() -> tuple[str, ...]:
    """按字典序完整枚举 000-999。"""

    return tuple(f"{value:03d}" for value in range(1000))


def _window_rows(
    state: LearnedHistoryState, window: Window
) -> tuple[tuple[int, int, int], ...]:
    if window == "all":
        return state.numbers
    return state.numbers[-int(window) :]


def _weights(size: int, half_life: float | None) -> np.ndarray:
    return np.asarray(
        [decay_weight(size - index, half_life) for index in range(size)], dtype=float
    )


def _rate(
    values: Sequence[object],
    target: object,
    domain_size: int,
    weights: np.ndarray,
    alpha: float,
) -> float:
    weighted_count = sum(
        float(weight) for value, weight in zip(values, weights) if value == target
    )
    return (weighted_count + alpha / domain_size) / (float(weights.sum()) + alpha)


def _patterns(numbers: tuple[int, int, int]) -> dict[str, object]:
    odd = sum(value % 2 for value in numbers)
    big = sum(value >= 5 for value in numbers)
    prime = sum(value in {2, 3, 5, 7} for value in numbers)
    return {
        "sum": sum(numbers),
        "span": max(numbers) - min(numbers),
        "shape": classify_digit_shape(numbers),
        "parity": odd,
        "big_small": big,
        "prime": prime,
        "sum_tail": sum(numbers) % 10,
        "consecutive": int(digit_consecutive_count(numbers) > 0),
        "mirror": int(digit_mirror_count(numbers) > 0),
    }


def _omission_statistics(
    numbers: tuple[tuple[int, int, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = np.zeros((3, 10), dtype=float)
    maximum = np.ones((3, 10), dtype=float)
    means = np.zeros((3, 10), dtype=float)
    stds = np.ones((3, 10), dtype=float)
    for position in range(3):
        for digit in range(10):
            hits = [
                index for index, row in enumerate(numbers) if row[position] == digit
            ]
            current[position, digit] = (
                len(numbers) - 1 - hits[-1] if hits else len(numbers)
            )
            boundaries = [-1, *hits, len(numbers)]
            gaps = np.asarray(
                [
                    max(0, right - left - 1)
                    for left, right in zip(boundaries, boundaries[1:])
                ],
                dtype=float,
            )
            if gaps.size:
                maximum[position, digit] = max(1.0, float(gaps.max()))
                means[position, digit] = float(gaps.mean())
                stds[position, digit] = max(1.0, float(gaps.std()))
    return current, maximum, means, stds


def _window_feature_row(
    candidate: tuple[int, int, int],
    rows: tuple[tuple[int, int, int], ...],
    config: LearnedFeatureConfig,
) -> dict[str, float]:
    if not rows:
        return {
            "position": math.log(0.1),
            "pair": math.log(0.01),
            "shape": math.log(1 / 3),
            "sum": math.log(1 / 28),
            "span": math.log(0.1),
            "parity_bigsmall": math.log(1 / 16),
        }
    weights = _weights(len(rows), config.half_life)
    candidate_pattern = _patterns(candidate)
    patterns = [_patterns(row) for row in rows]
    position_rates = [
        _rate([row[index] for row in rows], candidate[index], 10, weights, config.alpha)
        for index in range(3)
    ]
    pair_rates = [
        _rate(
            [(row[left], row[right]) for row in rows],
            (candidate[left], candidate[right]),
            100,
            weights,
            config.alpha,
        )
        for left, right in PAIR_SPECS
    ]
    parity = _rate(
        [item["parity"] for item in patterns],
        candidate_pattern["parity"],
        4,
        weights,
        config.alpha,
    )
    big_small = _rate(
        [item["big_small"] for item in patterns],
        candidate_pattern["big_small"],
        4,
        weights,
        config.alpha,
    )
    prime = _rate(
        [item["prime"] for item in patterns],
        candidate_pattern["prime"],
        4,
        weights,
        config.alpha,
    )
    sum_tail = _rate(
        [item["sum_tail"] for item in patterns],
        candidate_pattern["sum_tail"],
        10,
        weights,
        config.alpha,
    )
    consecutive = _rate(
        [item["consecutive"] for item in patterns],
        candidate_pattern["consecutive"],
        2,
        weights,
        config.alpha,
    )
    mirror = _rate(
        [item["mirror"] for item in patterns],
        candidate_pattern["mirror"],
        2,
        weights,
        config.alpha,
    )
    return {
        "position": float(np.mean(np.log(position_rates))),
        "pair": float(np.mean(np.log(pair_rates))),
        "shape": math.log(
            _rate(
                [item["shape"] for item in patterns],
                candidate_pattern["shape"],
                3,
                weights,
                config.alpha,
            )
        ),
        "sum": math.log(
            _rate(
                [item["sum"] for item in patterns],
                candidate_pattern["sum"],
                28,
                weights,
                config.alpha,
            )
        ),
        "span": math.log(
            _rate(
                [item["span"] for item in patterns],
                candidate_pattern["span"],
                10,
                weights,
                config.alpha,
            )
        ),
        "parity_bigsmall": float(
            np.mean(np.log([parity, big_small, prime, sum_tail, consecutive, mirror]))
        ),
    }


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median))) * 1.4826
    if scale <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _weighted_window_mean(
    per_window: Mapping[Window, Mapping[str, float]],
    key: str,
    config: LearnedFeatureConfig,
) -> float:
    weights = config.window_weight_map()
    numerator = math.fsum(
        float(values[key]) * weights[str(window)]
        for window, values in per_window.items()
    )
    denominator = math.fsum(weights[str(window)] for window in per_window)
    return numerator / denominator


def _trend_value(
    per_window: Mapping[Window, Mapping[str, float]],
    key: str,
    left: Window,
    right: Window,
) -> float:
    if left not in per_window or right not in per_window:
        return 0.0
    return float(per_window[left][key] - per_window[right][key])


def build_candidate_features(
    state: LearnedHistoryState,
    rule: LotteryRule,
    candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """为候选生成多窗口、平滑、衰减、遗漏和上期关系特征矩阵。"""

    if rule.code != state.rule_code or rule.draw_count != 3:
        raise ValueError("历史状态与三位彩规则不匹配")
    texts = (
        tuple(candidates)
        if candidates is not None
        else enumerate_three_digit_candidates()
    )
    if len(set(texts)) != len(texts):
        raise ValueError("候选号码不得重复")
    parsed: list[tuple[int, int, int]] = []
    for text in texts:
        if len(text) != 3 or not text.isdigit():
            raise ValueError(f"三位彩候选必须是 000-999：{text}")
        parsed.append((int(text[0]), int(text[1]), int(text[2])))
    current, maximum, means, stds = _omission_statistics(state.numbers)
    window_rows = {
        window: _window_rows(state, window) for window in state.config.windows
    }
    window_omissions = {
        window: _omission_statistics(values) for window, values in window_rows.items()
    }
    rows: list[dict[str, object]] = []
    for text, candidate in zip(texts, parsed):
        pattern = _patterns(candidate)
        per_window = {
            window: _window_feature_row(candidate, values, state.config)
            for window, values in window_rows.items()
        }
        position_omissions = np.asarray(
            [current[index, digit] for index, digit in enumerate(candidate)]
        )
        omission_by_window = {
            window: float(
                np.mean(
                    [
                        values[0][index, digit] / values[1][index, digit]
                        for index, digit in enumerate(candidate)
                    ]
                )
            )
            for window, values in window_omissions.items()
        }
        omission_zscores = np.asarray(
            [
                (current[index, digit] - means[index, digit]) / stds[index, digit]
                for index, digit in enumerate(candidate)
            ]
        )
        rebound = np.log1p(
            np.minimum(position_omissions, state.config.omission_cap)
        ) / math.log1p(state.config.omission_cap)
        ordered_windows = list(state.config.windows)
        short = per_window[ordered_windows[0]]
        long = per_window[ordered_windows[-1]]
        component_keys = ("position", "pair", "sum", "span", "shape")
        component_trends = {
            key: float(short[key] - long[key]) for key in component_keys
        }
        trend_30_300_by_key = {
            key: _trend_value(per_window, key, 30, 300) for key in component_keys
        }
        trend_50_all_by_key = {
            key: _trend_value(per_window, key, 50, "all") for key in component_keys
        }
        ratio_by_key: dict[str, float] = {}
        for key in component_keys:
            if 30 in per_window and 300 in per_window:
                # 窗口特征本身是 log 概率；两者相减即 log(rate30/rate300)。
                ratio_by_key[key] = float(per_window[30][key]) - float(
                    per_window[300][key]
                )
            else:
                ratio_by_key[key] = 0.0
        latest = state.latest_numbers
        latest_distance = (
            0.0
            if latest is None
            else sum(abs(left - right) for left, right in zip(candidate, latest)) / 27.0
        )
        repeated_digits = (
            0 if latest is None else len(set(candidate) & set(latest)) / 3.0
        )
        same_position = (
            0
            if latest is None
            else sum(left == right for left, right in zip(candidate, latest)) / 3.0
        )
        constraint = (
            max(0.0, (abs(float(sum(candidate)) - 13.5) - 8.0) / 13.5)
            + float(len(set(candidate)) == 1)
            + 0.25 * float(max(candidate) - min(candidate) in {0, 9})
        )
        item: dict[str, object] = {
            "candidate": text,
            "digit_0": candidate[0],
            "digit_1": candidate[1],
            "digit_2": candidate[2],
            **pattern,
            "position_frequency": float(
                _weighted_window_mean(per_window, "position", state.config)
            ),
            "position_omission": float(
                np.mean(
                    [np.clip(value, 0.0, 1.0) for value in omission_by_window.values()]
                )
            ),
            "pair_frequency": float(
                _weighted_window_mean(per_window, "pair", state.config)
            ),
            "shape_distribution": float(
                _weighted_window_mean(per_window, "shape", state.config)
            ),
            "sum_distribution": float(
                _weighted_window_mean(per_window, "sum", state.config)
            ),
            "span_distribution": float(
                _weighted_window_mean(per_window, "span", state.config)
            ),
            "parity_bigsmall": float(
                _weighted_window_mean(per_window, "parity_bigsmall", state.config)
            ),
            "recent_trend": float(np.mean([short[key] - long[key] for key in short])),
            "position_trend": component_trends["position"],
            "pair_trend": component_trends["pair"],
            "sum_trend": component_trends["sum"],
            "span_trend": component_trends["span"],
            "shape_trend": component_trends["shape"],
            "trend_30_300": float(np.mean(list(trend_30_300_by_key.values()))),
            "trend_50_all": float(np.mean(list(trend_50_all_by_key.values()))),
            "trend_ratio_30_300": float(np.mean(list(ratio_by_key.values()))),
            **{
                f"{key}_trend_30_300": value
                for key, value in trend_30_300_by_key.items()
            },
            **{
                f"{key}_trend_50_all": value
                for key, value in trend_50_all_by_key.items()
            },
            **{
                f"{key}_trend_ratio_30_300": value
                for key, value in ratio_by_key.items()
            },
            "latest_distance": -latest_distance,
            "repeat_latest": float(np.mean([repeated_digits, same_position])),
            "omission_rebound": float(
                np.mean(rebound) + 0.05 * np.mean(np.clip(omission_zscores, -3, 3))
            ),
            "constraint_penalty": float(constraint),
        }
        for window, values in per_window.items():
            label = str(window)
            for key, value in values.items():
                item[f"{key}_{label}"] = value
            item[f"omission_{label}"] = omission_by_window[window]
        rows.append(item)
    frame = pd.DataFrame(rows)
    for name in FEATURE_NAMES:
        values = frame[name].to_numpy(dtype=float)
        if name == "constraint_penalty":
            frame[name] = np.maximum(values, 0.0)
        elif name in {
            "position_omission",
            "omission_rebound",
            "repeat_latest",
            "latest_distance",
        }:
            frame[name] = np.clip(values, -1.0, 1.5)
        else:
            frame[name] = _robust_z(values)
    return frame
