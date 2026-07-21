# -*- coding: utf-8 -*-
"""三位彩固定评分算法 v4 的无泄漏特征工程。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

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
DEFAULT_WINDOW_WEIGHTS: Mapping[str, float] = {
    "10": 3.0,
    "20": 2.5,
    "30": 2.0,
    "50": 1.5,
    "100": 1.0,
    "150": 0.75,
    "300": 0.5,
    "all": 0.25,
}
PAIR_SPECS = ((0, 1), (0, 2), (1, 2))
SHAPE_NAMES = ("组六", "组三", "豹子")
SHAPE_PRIORS = {"组六": 0.72, "组三": 0.27, "豹子": 0.01}
FEATURE_NAMES = (
    "position_frequency",
    "position_omission",
    "pair_frequency",
    "sum_distribution",
    "span_distribution",
    "recent_trend",
    "position_trend",
    "pair_trend",
    "shape_transition",
    "shape_recent_deviation",
    "constraint_penalty",
)
BEHAVIORAL_FEATURE_NAMES = (
    "exact_repeat_pressure",
    "group_repeat_pressure",
    "last_position_overlap",
    "last_digit_overlap",
    "shape_transition_deviation",
    "shape_run_length",
)


@dataclass(frozen=True)
class LearnedFeatureConfig:
    """v4 特征配置；窗口只读取目标期之前的数据。"""

    windows: tuple[Window, ...] = (20, 50, 150)
    alpha: float = 2.0
    half_life: float | None = 50.0
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
        supplied = dict(
            self.window_weights
            or {label: DEFAULT_WINDOW_WEIGHTS.get(label, 1.0) for label in labels}
        )
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
    finite_windows = [int(window) for window in effective.windows if window != "all"]
    if "all" not in effective.windows and finite_windows:
        chronological = chronological.tail(max(finite_windows))
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


def iter_rolling_history_states(
    history: pd.DataFrame,
    rule: LotteryRule,
    target_indices: Sequence[int],
    config: LearnedFeatureConfig | None = None,
) -> Iterator[LearnedHistoryState]:
    """按升序目标索引增量追加历史，并产出无泄漏只读快照。"""

    effective = config or LearnedFeatureConfig()
    indices = tuple(int(index) for index in target_indices)
    if any(index <= 0 for index in indices) or any(
        right <= left for left, right in zip(indices, indices[1:])
    ):
        raise ValueError("目标索引必须为严格递增正整数")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    if indices and indices[-1] >= len(chronological):
        raise ValueError("目标索引超过历史范围")
    finite_windows = [int(window) for window in effective.windows if window != "all"]
    maximum_history = (
        len(chronological) if "all" in effective.windows else max(finite_windows)
    )
    issue_buffer: deque[str] = deque(maxlen=maximum_history)
    number_buffer: deque[tuple[int, int, int]] = deque(maxlen=maximum_history)
    cursor = 0
    for target_index in indices:
        while cursor < target_index:
            row = chronological.iloc[cursor]
            issue_buffer.append(str(row["期数"]))
            values = tuple(int(row[column]) for column in rule.number_columns)
            number_buffer.append((values[0], values[1], values[2]))
            cursor += 1
        target_issue = str(chronological.iloc[target_index]["期数"])
        issues = tuple(issue_buffer)
        yield LearnedHistoryState(
            rule_code=rule.code,
            config=effective,
            target_issue=target_issue,
            history_issues=issues,
            history_end_issue=issues[-1] if issues else None,
            numbers=tuple(number_buffer),
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


def _shape_deviation_maps(
    numbers: tuple[tuple[int, int, int], ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """一次性计算三种形态的转移和近期先验偏离。"""
    zero = {name: 0.0 for name in SHAPE_NAMES}
    if not numbers:
        return zero.copy(), zero.copy()
    shapes = [classify_digit_shape(row) for row in numbers]
    smoothing = 20.0
    transitions = list(zip(shapes[:-1], shapes[1:]))
    previous_shape = shapes[-1]
    transition_values = [right for left, right in transitions if left == previous_shape]
    recent = shapes[-30:]
    transition = {
        name: math.log(
            max(
                (transition_values.count(name) + smoothing * SHAPE_PRIORS[name])
                / (len(transition_values) + smoothing),
                1e-12,
            )
            / SHAPE_PRIORS[name]
        )
        for name in SHAPE_NAMES
    }
    recent_map = {
        name: math.log(
            max(
                (recent.count(name) + smoothing * SHAPE_PRIORS[name])
                / (len(recent) + smoothing),
                1e-12,
            )
            / SHAPE_PRIORS[name]
        )
        for name in SHAPE_NAMES
    }
    return transition, recent_map


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


def _smoothed_rates(
    values: np.ndarray,
    domain_size: int,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    counts = np.bincount(values, weights=weights, minlength=domain_size).astype(float)
    return (counts + alpha / domain_size) / (float(weights.sum()) + alpha)


def _window_feature_arrays(
    candidates: np.ndarray,
    rows: tuple[tuple[int, int, int], ...],
    config: LearnedFeatureConfig,
) -> dict[str, np.ndarray]:
    """一次性计算一个窗口内全部候选的核心特征。"""

    candidate_count = len(candidates)
    if not rows:
        return {
            "position": np.full(candidate_count, math.log(0.1)),
            "pair": np.full(candidate_count, math.log(0.01)),
            "sum": np.full(candidate_count, math.log(1 / 28)),
            "span": np.full(candidate_count, math.log(0.1)),
        }
    history = np.asarray(rows, dtype=int)
    weights = _weights(len(rows), config.half_life)
    position_logs = []
    for position in range(3):
        rates = _smoothed_rates(history[:, position], 10, weights, config.alpha)
        position_logs.append(np.log(rates[candidates[:, position]]))
    pair_logs = []
    for left, right in PAIR_SPECS:
        history_codes = history[:, left] * 10 + history[:, right]
        candidate_codes = candidates[:, left] * 10 + candidates[:, right]
        rates = _smoothed_rates(history_codes, 100, weights, config.alpha)
        pair_logs.append(np.log(rates[candidate_codes]))
    history_sums = history.sum(axis=1)
    candidate_sums = candidates.sum(axis=1)
    history_spans = history.max(axis=1) - history.min(axis=1)
    candidate_spans = candidates.max(axis=1) - candidates.min(axis=1)
    return {
        "position": np.mean(np.vstack(position_logs), axis=0),
        "pair": np.mean(np.vstack(pair_logs), axis=0),
        "sum": np.log(
            _smoothed_rates(history_sums, 28, weights, config.alpha)[candidate_sums]
        ),
        "span": np.log(
            _smoothed_rates(history_spans, 10, weights, config.alpha)[candidate_spans]
        ),
    }


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median))) * 1.4826
    if scale <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _weighted_window_arrays(
    per_window: Mapping[Window, Mapping[str, np.ndarray]],
    key: str,
    config: LearnedFeatureConfig,
) -> np.ndarray:
    window_weights = config.window_weight_map()
    ordered = tuple(per_window)
    weights = np.asarray([window_weights[str(window)] for window in ordered])
    matrix = np.vstack([per_window[window][key] for window in ordered])
    return np.average(matrix, axis=0, weights=weights)


def _behavioral_context_arrays(
    candidates: np.ndarray,
    state: LearnedHistoryState,
) -> dict[str, np.ndarray]:
    candidate_count = len(candidates)
    zeros = np.zeros(candidate_count, dtype=float)
    if not state.numbers:
        return {name: zeros.copy() for name in BEHAVIORAL_FEATURE_NAMES}

    history = np.asarray(state.numbers, dtype=int)
    ages = np.arange(len(history), 0, -1, dtype=float)
    pressure_weights = np.exp(-ages / 50.0)
    history_exact_codes = history[:, 0] * 100 + history[:, 1] * 10 + history[:, 2]
    candidate_exact_codes = (
        candidates[:, 0] * 100 + candidates[:, 1] * 10 + candidates[:, 2]
    )
    exact_totals = np.bincount(
        history_exact_codes, weights=pressure_weights, minlength=1000
    )
    exact_pressure = np.log1p(exact_totals[candidate_exact_codes])

    def group_key(row: np.ndarray | tuple[int, int, int]) -> str:
        return "".join(str(int(value)) for value in sorted(row))

    group_totals: dict[str, float] = {}
    for row, weight in zip(history, pressure_weights):
        key = group_key(row)
        group_totals[key] = group_totals.get(key, 0.0) + float(weight)
    group_pressure = np.asarray(
        [math.log1p(group_totals.get(group_key(row), 0.0)) for row in candidates],
    )

    latest = history[-1]
    position_overlap = np.mean(candidates == latest, axis=1)
    latest_counts = np.bincount(latest, minlength=10)
    candidate_counts = np.stack([np.bincount(row, minlength=10) for row in candidates])
    digit_overlap = np.minimum(candidate_counts, latest_counts).sum(axis=1) / 3.0

    shape_transition_map, _ = _shape_deviation_maps(state.numbers)
    candidate_shapes = tuple(
        classify_digit_shape(tuple(int(value) for value in row)) for row in candidates
    )
    transition_deviation = np.clip(
        np.asarray([shape_transition_map[name] for name in candidate_shapes]),
        -3.0,
        3.0,
    )
    latest_shape = classify_digit_shape(tuple(int(value) for value in latest))
    run_length = 0
    for row in reversed(history):
        if classify_digit_shape(tuple(int(value) for value in row)) != latest_shape:
            break
        run_length += 1
    run_strength = min(run_length / 20.0, 1.0)
    shape_run = np.asarray(
        [run_strength if name == latest_shape else 0.0 for name in candidate_shapes],
        dtype=float,
    )
    return {
        "exact_repeat_pressure": exact_pressure,
        "group_repeat_pressure": group_pressure,
        "last_position_overlap": position_overlap,
        "last_digit_overlap": digit_overlap,
        "shape_transition_deviation": transition_deviation,
        "shape_run_length": shape_run,
    }


def build_candidate_features(
    state: LearnedHistoryState,
    rule: LotteryRule,
    candidates: Sequence[str] | None = None,
    *,
    include_behavioral_context: bool = False,
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
    candidate_digits = np.asarray(parsed, dtype=int)
    window_rows = {
        window: _window_rows(state, window) for window in state.config.windows
    }
    per_window = {
        window: _window_feature_arrays(candidate_digits, values, state.config)
        for window, values in window_rows.items()
    }
    window_omissions = {
        window: _omission_statistics(values) for window, values in window_rows.items()
    }
    shape_transition_map, shape_recent_map = _shape_deviation_maps(state.numbers)
    omission_arrays = []
    positions = np.arange(3)[:, None]
    for current, maximum, _, _ in window_omissions.values():
        selected = current[positions, candidate_digits.T]
        selected_maximum = maximum[positions, candidate_digits.T]
        omission_arrays.append(np.mean(selected / selected_maximum, axis=0))
    ordered_windows = tuple(state.config.windows)
    short = per_window[ordered_windows[0]]
    long = per_window[ordered_windows[-1]]
    component_keys = ("position", "pair", "sum", "span")
    shape_names = tuple(classify_digit_shape(candidate) for candidate in parsed)
    candidate_sums = candidate_digits.sum(axis=1).astype(float)
    candidate_spans = candidate_digits.max(axis=1) - candidate_digits.min(axis=1)
    is_triple = np.all(candidate_digits == candidate_digits[:, :1], axis=1).astype(
        float
    )
    constraint = (
        np.maximum(0.0, (np.abs(candidate_sums - 13.5) - 8.0) / 13.5)
        + is_triple
        + 0.25 * np.isin(candidate_spans, (0, 9)).astype(float)
    )
    frame = pd.DataFrame(
        {
            "candidate": texts,
            "position_frequency": _weighted_window_arrays(
                per_window, "position", state.config
            ),
            "position_omission": np.mean(
                np.clip(np.vstack(omission_arrays), 0.0, 1.0), axis=0
            ),
            "pair_frequency": _weighted_window_arrays(per_window, "pair", state.config),
            "sum_distribution": _weighted_window_arrays(
                per_window, "sum", state.config
            ),
            "span_distribution": _weighted_window_arrays(
                per_window, "span", state.config
            ),
            "recent_trend": np.mean(
                np.vstack([short[key] - long[key] for key in component_keys]), axis=0
            ),
            "position_trend": short["position"] - long["position"],
            "pair_trend": short["pair"] - long["pair"],
            "shape_transition": np.asarray(
                [shape_transition_map[name] for name in shape_names]
            ),
            "shape_recent_deviation": np.asarray(
                [shape_recent_map[name] for name in shape_names]
            ),
            "constraint_penalty": constraint,
        }
    )
    for name in FEATURE_NAMES:
        values = frame[name].to_numpy(dtype=float)
        if name == "constraint_penalty":
            frame[name] = np.maximum(values, 0.0)
        elif name in {
            "position_omission",
            "shape_recent_deviation",
        }:
            frame[name] = np.clip(values, -1.0, 1.5)
        else:
            frame[name] = _robust_z(values)
    if include_behavioral_context:
        for name, values in _behavioral_context_arrays(candidate_digits, state).items():
            frame[name] = values
    return frame
