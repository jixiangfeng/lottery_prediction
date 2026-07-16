# -*- coding: utf-8 -*-
"""基于多窗口位置分布的数字彩蒙特卡洛候选模拟。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.analysis.digit_candidates import (
    DigitCandidateConfig,
    _build_score_context,
    _effective_config,
    _passes_cached_filters,
    _structure_constraint_penalty,
)
from src.analysis.digit_statistics import DigitStatisticsResult, classify_digit_shape
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitMonteCarloResult:
    """蒙特卡洛模拟结果，分数是模拟频率而非开奖概率。"""

    simulations: int
    accepted: int
    scores: dict[str, float]
    pair_conditioned: bool = True
    structure_conditioned: bool = True


def _position_distribution(
    stats: DigitStatisticsResult,
    position: str,
    config: DigitCandidateConfig,
) -> np.ndarray:
    probabilities = np.zeros(10, dtype=float)
    total_weight = 0.0
    for window, weight in zip(
        config.frequency_windows, config.frequency_window_weights
    ):
        window_values = stats.position_probabilities.get(int(window), {}).get(position)
        if window_values is None or weight <= 0:
            continue
        for digit, probability in window_values.items():
            probabilities[int(digit)] += float(weight) * float(probability)
        total_weight += float(weight)
    if total_weight:
        probabilities /= total_weight
    if probabilities.sum() <= 0:
        probabilities[:] = 0.1
    return probabilities / probabilities.sum()


def _pair_conditional_table(
    stats: DigitStatisticsResult,
    left_index: int,
    right_index: int,
    config: DigitCandidateConfig,
) -> np.ndarray:
    """聚合多窗口位置对概率，转为 P(右位|左位) 条件概率表。"""

    joint = np.zeros((10, 10), dtype=float)
    total_weight = 0.0
    pair_key = f"{left_index}-{right_index}"
    for window, weight in zip(
        config.frequency_windows, config.frequency_window_weights
    ):
        positive_weight = max(0.0, float(weight))
        values = stats.pair_probabilities.get(int(window), {}).get(pair_key)
        if values is None or positive_weight == 0:
            continue
        for (left, right), probability in values.items():
            joint[int(left), int(right)] += positive_weight * float(probability)
        total_weight += positive_weight
    if total_weight:
        joint /= total_weight
    row_sums = joint.sum(axis=1, keepdims=True)
    return np.divide(
        joint,
        row_sums,
        out=np.full((10, 10), 0.1, dtype=float),
        where=row_sums > 0,
    )


def _weighted_shape_probabilities(
    stats: DigitStatisticsResult, config: DigitCandidateConfig
) -> dict[str, float]:
    output: dict[str, float] = {}
    total_weight = 0.0
    for window, weight in zip(
        config.frequency_windows, config.frequency_window_weights
    ):
        positive_weight = max(0.0, float(weight))
        values = stats.shape_probabilities.get(int(window))
        if values is None or positive_weight == 0:
            continue
        for shape, probability in values.items():
            output[shape] = output.get(shape, 0.0) + positive_weight * float(
                probability
            )
        total_weight += positive_weight
    if total_weight:
        return {shape: value / total_weight for shape, value in output.items()}
    return output


def simulate_digit_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitCandidateConfig | None = None,
    *,
    simulations: int = 20_000,
    seed: int = 20260716,
    pair_strength: float = 0.75,
    structure_strength: float = 0.35,
) -> DigitMonteCarloResult:
    """按多窗口边际、位置对条件概率和形态分布模拟候选。"""

    if simulations <= 0 or pair_strength < 0 or structure_strength < 0:
        raise ValueError("simulations 必须为正整数，联合分布强度不得为负数")
    effective = _effective_config(rule, config or DigitCandidateConfig())
    rng = np.random.default_rng(seed)
    sampled = np.zeros((simulations, rule.draw_count), dtype=int)
    for right_index, position in enumerate(rule.number_columns):
        probabilities = np.tile(
            _position_distribution(stats, position, effective), (simulations, 1)
        )
        for left_index in range(right_index):
            conditional = _pair_conditional_table(
                stats, left_index, right_index, effective
            )
            probabilities *= conditional[sampled[:, left_index], :] ** pair_strength
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        cumulative = probabilities.cumsum(axis=1)
        sampled[:, right_index] = (rng.random(simulations)[:, None] > cumulative).sum(
            axis=1
        )
    latest_text = "".join(str(number) for number in stats.latest_numbers)
    context = _build_score_context(stats, effective)
    shape_probabilities = _weighted_shape_probabilities(stats, effective)
    maximum_shape_probability = max(shape_probabilities.values(), default=1.0)
    counts: Counter[str] = Counter()
    for row in sampled:
        numbers = [int(value) for value in row]
        text = "".join(str(value) for value in numbers)
        if effective.exclude_latest and text == latest_text:
            continue
        if not _passes_cached_filters(
            sum(numbers),
            max(numbers) - min(numbers),
            classify_digit_shape(numbers),
            effective,
        ):
            continue
        if np.isinf(_structure_constraint_penalty(numbers, context, effective)):
            continue
        shape = classify_digit_shape(numbers)
        shape_acceptance = (
            shape_probabilities.get(shape, 0.0) / maximum_shape_probability
            if maximum_shape_probability > 0
            else 1.0
        ) ** structure_strength
        if rng.random() > shape_acceptance:
            continue
        counts[text] += 1
    accepted = sum(counts.values())
    scores = {text: count / accepted for text, count in counts.items() if accepted > 0}
    return DigitMonteCarloResult(
        simulations=simulations,
        accepted=accepted,
        scores=scores,
        pair_conditioned=pair_strength > 0,
        structure_conditioned=structure_strength > 0,
    )


__all__ = ["DigitMonteCarloResult", "simulate_digit_candidates"]
