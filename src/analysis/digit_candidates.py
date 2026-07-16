# -*- coding: utf-8 -*-
"""数字彩候选生成器。

基于位置频率与当前遗漏生成福彩3D、排列三、排列五候选，并用和值、跨度、形态做过滤。
它是统计辅助工具，不保证命中。
"""

from __future__ import annotations

import itertools
import heapq
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Sequence

from src.analysis.digit_statistics import (
    DigitStatisticsResult,
    classify_digit_shape,
    digit_consecutive_count,
    digit_latest_distance,
    digit_mirror_count,
    digit_prime_composite_label,
    digit_repeat_latest_count,
    digit_sum_tail,
)
from src.lotteries.base import LotteryRule

ENSEMBLE_MODEL_NAMES = (
    "position",
    "pair",
    "shape",
    "sum",
    "span",
    "parity",
    "bigSmall",
    "primeComposite",
    "consecutive",
    "mirror",
    "sumTail",
    "latestDistance",
    "repeatLatest",
    "omission",
    "monteCarlo",
    "mlRanker",
)


@dataclass(frozen=True)
class DigitCandidateConfig:
    """数字彩候选生成配置。

    ``frequency_weight``、``random_weight`` 与 ``top_digits_per_position``
    仅为兼容旧构造 API 保留，当前全空间复合评分不再使用这些配置。
    """

    count: int = 10
    sum_min: int | None = None
    sum_max: int | None = None
    span_min: int | None = None
    span_max: int | None = None
    allowed_shapes: tuple[str, ...] | None = None
    top_digits_per_position: int = 6
    frequency_weight: float = 0.7
    omission_weight: float = 0.3
    random_weight: float = 0.15
    exclude_latest: bool = True
    frequency_windows: tuple[int, ...] = (30, 50, 100, 300)
    frequency_window_weights: tuple[float, ...] = (0.35, 0.3, 0.2, 0.15)
    omission_cap: int = 30
    diversity_weight: float = 0.3
    high_score_pool_factor: int = 1000
    marginal_weight: float = 1.0
    pair_weight: float = 1.0
    shape_weight: float = 0.2
    sum_weight: float = 0.15
    span_weight: float = 0.1
    score_floor: float = 2.0
    ranking_mode: str = "composite"
    ensemble_model_weights: tuple[float, ...] = (
        0.18,
        0.12,
        0.07,
        0.06,
        0.04,
        0.04,
        0.04,
        0.05,
        0.04,
        0.04,
        0.04,
        0.06,
        0.05,
        0.05,
        0.06,
        0.06,
    )
    ensemble_score_floor: float = 0.08
    constraint_mode: str = "soft"
    constraint_probability_floor: float = 0.02
    constraint_penalty_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("候选数量必须为正整数")
        if len(self.frequency_windows) != len(self.frequency_window_weights):
            raise ValueError("频率窗口与窗口权重数量必须一致")
        if not self.frequency_windows or any(
            int(window) <= 0 for window in self.frequency_windows
        ):
            raise ValueError("频率窗口必须为正整数")
        weights = (
            self.frequency_weight,
            self.omission_weight,
            self.random_weight,
            self.diversity_weight,
            self.marginal_weight,
            self.pair_weight,
            self.shape_weight,
            self.sum_weight,
            self.span_weight,
        )
        if any(float(weight) < 0 for weight in weights):
            raise ValueError("评分权重不得为负数")
        if self.score_floor < 0:
            raise ValueError("score_floor 不得为负数")
        if self.ranking_mode not in {"composite", "ensemble"}:
            raise ValueError("ranking_mode 只支持 composite 或 ensemble")
        if len(self.ensemble_model_weights) != len(ENSEMBLE_MODEL_NAMES):
            raise ValueError("集成模型权重数量必须与子模型数量一致")
        if any(float(weight) < 0 for weight in self.ensemble_model_weights):
            raise ValueError("集成模型权重不得为负数")
        if sum(self.ensemble_model_weights) <= 0:
            raise ValueError("集成模型权重之和必须大于零")
        if not 0.0 <= self.ensemble_score_floor <= 1.0:
            raise ValueError("ensemble_score_floor 必须在 0 到 1 之间")
        if self.constraint_mode not in {"off", "soft", "hard"}:
            raise ValueError("constraint_mode 只支持 off、soft 或 hard")
        if not 0.0 <= self.constraint_probability_floor <= 1.0:
            raise ValueError("constraint_probability_floor 必须在 0 到 1 之间")
        if self.constraint_penalty_weight < 0:
            raise ValueError("constraint_penalty_weight 不得为负数")
        if sum(self.frequency_window_weights) <= 0:
            raise ValueError("频率窗口权重之和必须大于零")


@dataclass(frozen=True)
class DigitExternalModelScores:
    """蒙特卡洛和机器学习模型对完整候选空间的外部分数。"""

    monte_carlo: dict[str, float]
    ml_ranker: dict[str, float]

    def values_for(self, text: str) -> tuple[float, float]:
        return (
            float(self.monte_carlo.get(text, 0.0)),
            float(self.ml_ranker.get(text, 0.0)),
        )


def with_all_history_window(
    config: DigitCandidateConfig,
    total_issues: int,
    *,
    all_history_weight: float = 0.1,
) -> DigitCandidateConfig:
    """在 30/50/100/300 窗口之外加入当前可用全历史窗口。"""

    window = max(1, int(total_issues))
    if window in config.frequency_windows:
        return config
    return replace(
        config,
        frequency_windows=(*config.frequency_windows, window),
        frequency_window_weights=(
            *config.frequency_window_weights,
            max(0.0, float(all_history_weight)),
        ),
    )


@dataclass(frozen=True)
class DigitCandidate:
    """单个数字彩候选及其启发式复合模型诊断。"""

    numbers: list[int]
    text: str
    sum_value: int
    span: int
    shape: str
    score: float
    joint_probability: float = 0.0
    ensemble_score: float = 0.0
    model_rank_percentiles: tuple[float, ...] = ()
    constraint_penalty: float = 0.0

    @property
    def model_weight(self) -> float:
        """返回复合对数评分对应的未归一化模型权重，不是开奖概率。"""

        return math.exp(self.score)

    @property
    def composite_model_weight(self) -> float:
        """返回过滤空间内归一化模型质量，不是实际开奖概率。"""

        return self.joint_probability

    @property
    def top_decile_votes(self) -> int:
        """返回进入各子模型前 10% 的票数。"""

        return sum(value >= 0.9 for value in self.model_rank_percentiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "numbers": self.numbers,
            "sum": self.sum_value,
            "span": self.span,
            "shape": self.shape,
            "score": self.score,
            "modelWeight": self.model_weight,
            "compositeModelWeight": self.composite_model_weight,
            "jointProbability": self.joint_probability,
            "jointProbabilityDeprecated": True,
            "ensembleScore": self.ensemble_score,
            "modelRankPercentiles": {
                name: value
                for name, value in zip(
                    ENSEMBLE_MODEL_NAMES, self.model_rank_percentiles
                )
            },
            "topDecileVotes": self.top_decile_votes,
            "constraintPenalty": self.constraint_penalty,
        }


@dataclass(frozen=True)
class DigitGroupCandidate:
    """三位彩无序组选候选及过滤空间归一化模型质量。"""

    group_key: str
    numbers: list[int]
    shape: str
    probability_mass: float
    score: float
    permutations: int
    ensemble_score: float = 0.0
    model_rank_percentiles: tuple[float, ...] = ()
    ranking_model: str = "composite_aggregation"

    @property
    def composite_model_weight(self) -> float:
        """返回组选过滤空间归一化模型质量，不是实际开奖概率。"""

        return self.probability_mass

    def to_dict(self) -> dict[str, Any]:
        return {
            "groupKey": self.group_key,
            "numbers": self.numbers,
            "shape": self.shape,
            "compositeModelWeight": self.composite_model_weight,
            "probabilityMass": self.probability_mass,
            "probabilityMassDeprecated": True,
            "score": self.score,
            "permutations": self.permutations,
            "ensembleScore": self.ensemble_score,
            "modelRankPercentiles": {
                name: value
                for name, value in zip(
                    ENSEMBLE_MODEL_NAMES, self.model_rank_percentiles
                )
            },
            "rankingModel": self.ranking_model,
        }


@dataclass(frozen=True)
class DigitCandidateResult:
    """数字彩候选生成结果。"""

    rule_code: str
    display_name: str
    candidates: list[DigitCandidate]
    config: DigitCandidateConfig
    model_candidates: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "config": {
                "count": self.config.count,
                "sumMin": self.config.sum_min,
                "sumMax": self.config.sum_max,
                "spanMin": self.config.span_min,
                "spanMax": self.config.span_max,
                "allowedShapes": (
                    list(self.config.allowed_shapes)
                    if self.config.allowed_shapes
                    else None
                ),
                "excludeLatest": self.config.exclude_latest,
                "frequencyWeight": self.config.frequency_weight,
                "omissionWeight": self.config.omission_weight,
                "frequencyWindows": list(self.config.frequency_windows),
                "frequencyWindowWeights": list(self.config.frequency_window_weights),
                "omissionCap": self.config.omission_cap,
                "diversityWeight": self.config.diversity_weight,
                "marginalWeight": self.config.marginal_weight,
                "pairWeight": self.config.pair_weight,
                "shapeWeight": self.config.shape_weight,
                "sumWeight": self.config.sum_weight,
                "spanWeight": self.config.span_weight,
                "scoreFloor": self.config.score_floor,
                "rankingMode": self.config.ranking_mode,
                "ensembleModelWeights": {
                    name: weight
                    for name, weight in zip(
                        ENSEMBLE_MODEL_NAMES, self.config.ensemble_model_weights
                    )
                },
                "ensembleScoreFloor": self.config.ensemble_score_floor,
                "constraintMode": self.config.constraint_mode,
                "constraintProbabilityFloor": self.config.constraint_probability_floor,
                "constraintPenaltyWeight": self.config.constraint_penalty_weight,
                "deprecatedCompatibilityFields": [
                    "frequencyWeight",
                    "randomWeight",
                    "topDigitsPerPosition",
                ],
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "modelCandidates": self.model_candidates,
        }


@dataclass(frozen=True)
class DigitBettingCandidateResult:
    """兼容旧候选字段的直选/组选投注计划。"""

    rule_code: str
    display_name: str
    direct_candidates: list[DigitCandidate]
    group_candidates: list[DigitGroupCandidate]
    config: DigitCandidateConfig
    model_candidates: dict[str, list[str]] = field(default_factory=dict)

    @property
    def candidates(self) -> list[DigitCandidate]:
        """兼容旧调用：``candidates`` 始终代表直选候选。"""

        return self.direct_candidates

    def to_dict(self) -> dict[str, Any]:
        legacy = DigitCandidateResult(
            self.rule_code,
            self.display_name,
            self.direct_candidates,
            self.config,
            self.model_candidates,
        ).to_dict()
        legacy["directCandidates"] = [
            candidate.to_dict() for candidate in self.direct_candidates
        ]
        legacy["groupCandidates"] = [
            candidate.to_dict() for candidate in self.group_candidates
        ]
        return legacy


def _weighted_log_probability(
    probability_windows: dict[int, dict[Any, float]],
    value: Any,
    config: DigitCandidateConfig,
) -> float:
    """按配置窗口聚合离散特征的启发式对数分量。"""

    weighted = 0.0
    total_weight = 0.0
    for window, weight in zip(
        config.frequency_windows, config.frequency_window_weights
    ):
        probabilities = probability_windows.get(int(window))
        positive_weight = max(0.0, float(weight))
        if probabilities is None or positive_weight == 0:
            continue
        weighted += positive_weight * math.log(
            max(probabilities.get(value, 0.0), 1e-300)
        )
        total_weight += positive_weight
    if total_weight:
        return weighted / total_weight
    return math.log(1e-300)


def _position_log_probability(
    stats: DigitStatisticsResult,
    position: str,
    digit: int,
    config: DigitCandidateConfig,
) -> float:
    windows = {
        window: positions[position]
        for window, positions in stats.position_probabilities.items()
        if position in positions
    }
    return _weighted_log_probability(windows, digit, config)


def _pair_log_probability(
    stats: DigitStatisticsResult,
    pair_key: str,
    pair: tuple[int, int],
    config: DigitCandidateConfig,
) -> float:
    windows = {
        window: pairs[pair_key]
        for window, pairs in stats.pair_probabilities.items()
        if pair_key in pairs
    }
    return _weighted_log_probability(windows, pair, config)


def _structure_log_probability(
    probability_windows: dict[int, dict[Any, float]],
    value: Any,
    config: DigitCandidateConfig,
) -> float:
    return _weighted_log_probability(probability_windows, value, config)


def _weighted_log_map(
    probability_windows: dict[int, dict[Any, float]],
    config: DigitCandidateConfig,
) -> dict[Any, float]:
    values = {
        value
        for window in config.frequency_windows
        for value in probability_windows.get(int(window), {})
    }
    return {
        value: _weighted_log_probability(probability_windows, value, config)
        for value in values
    }


def _weighted_omission_score(
    stats: DigitStatisticsResult,
    position: str,
    digit: int,
    config: DigitCandidateConfig,
) -> float:
    """聚合短中长窗口遗漏，窗口内未出现时以该窗口样本长度截断。"""

    total = 0.0
    total_weight = 0.0
    omission_cap = max(1, int(config.omission_cap))
    for window, weight in zip(
        config.frequency_windows, config.frequency_window_weights
    ):
        positive_weight = max(0.0, float(weight))
        values = stats.omission_windows.get(int(window), {}).get(position)
        if values is None or positive_weight == 0:
            continue
        miss = min(int(values.get(int(digit), 0)), omission_cap)
        total += positive_weight * math.log1p(miss) / math.log1p(omission_cap)
        total_weight += positive_weight
    if total_weight:
        return total / total_weight
    miss = min(
        int(stats.current_omission.get(position, {}).get(int(digit), 0)),
        omission_cap,
    )
    return math.log1p(miss) / math.log1p(omission_cap)


def _build_score_context(
    stats: DigitStatisticsResult, config: DigitCandidateConfig
) -> dict[str, Any]:
    positions = list(stats.position_frequency)
    marginal = {}
    for position in positions:
        windows = {
            window: probabilities[position]
            for window, probabilities in stats.position_probabilities.items()
            if position in probabilities
        }
        marginal[position] = _weighted_log_map(windows, config)
    pair = {}
    for pair_key in stats.pair_probabilities.get(int(config.frequency_windows[0]), {}):
        windows = {
            window: probabilities[pair_key]
            for window, probabilities in stats.pair_probabilities.items()
            if pair_key in probabilities
        }
        pair[pair_key] = _weighted_log_map(windows, config)
    omission = {
        position: {
            digit: _weighted_omission_score(stats, position, digit, config)
            for digit in stats.current_omission.get(position, {})
        }
        for position in positions
    }
    return {
        "positions": positions,
        "marginal": marginal,
        "pair": pair,
        "omission": omission,
        "prefixShape": _weighted_log_map(stats.prefix3_shape_probabilities, config),
        "prefixSum": _weighted_log_map(stats.prefix3_sum_probabilities, config),
        "prefixSpan": _weighted_log_map(stats.prefix3_span_probabilities, config),
        "shape": _weighted_log_map(stats.shape_probabilities, config),
        "sum": _weighted_log_map(stats.sum_probabilities, config),
        "span": _weighted_log_map(stats.span_probabilities, config),
        "parity": _weighted_log_map(stats.parity_probabilities, config),
        "bigSmall": _weighted_log_map(stats.big_small_probabilities, config),
        "primeComposite": _weighted_log_map(
            stats.prime_composite_probabilities, config
        ),
        "consecutive": _weighted_log_map(stats.consecutive_probabilities, config),
        "mirror": _weighted_log_map(stats.mirror_probabilities, config),
        "sumTail": _weighted_log_map(stats.sum_tail_probabilities, config),
        "latestDistance": _weighted_log_map(
            stats.latest_distance_probabilities, config
        ),
        "repeatLatest": _weighted_log_map(stats.repeat_latest_probabilities, config),
        "latestNumbers": stats.latest_numbers,
    }


def _structure_constraint_penalty(
    numbers: Sequence[int],
    context: dict[str, Any],
    config: DigitCandidateConfig,
) -> float:
    """返回历史结构罕见度惩罚；硬约束不通过时返回无穷大。"""

    if config.constraint_mode == "off" or config.constraint_probability_floor <= 0:
        return 0.0
    values = (
        context["parity"][_parity_label(numbers)],
        context["bigSmall"][_big_small_label(numbers)],
        context["primeComposite"][digit_prime_composite_label(numbers)],
    )
    threshold = math.log(max(config.constraint_probability_floor, 1e-300))
    deficits = [max(0.0, threshold - value) for value in values]
    if config.constraint_mode == "hard" and any(deficits):
        return math.inf
    return config.constraint_penalty_weight * sum(deficits)


def _parity_label(numbers: Sequence[int]) -> str:
    odd = sum(int(number) % 2 for number in numbers)
    return f"奇{odd}偶{len(numbers) - odd}"


def _big_small_label(numbers: Sequence[int]) -> str:
    big = sum(int(number) >= 5 for number in numbers)
    return f"大{big}小{len(numbers) - big}"


def _model_component_scores(
    numbers: Sequence[int], context: dict[str, Any]
) -> tuple[float, ...]:
    """计算 14 个统计子模型及 2 个外部票占位分，后续仅通过排名分位融合。"""

    positions = context["positions"]
    marginal_logs = [
        context["marginal"][positions[index]][int(number)]
        for index, number in enumerate(numbers)
    ]
    pair_logs = [
        context["pair"][f"{left}-{right}"][(int(numbers[left]), int(numbers[right]))]
        for left in range(len(numbers))
        for right in range(left + 1, len(numbers))
    ]
    omission_scores = [
        context["omission"].get(positions[index], {}).get(int(number), 0.0)
        for index, number in enumerate(numbers)
    ]
    shape_map = context["prefixShape"] if len(numbers) == 3 else context["shape"]
    sum_map = context["prefixSum"] if len(numbers) == 3 else context["sum"]
    span_map = context["prefixSpan"] if len(numbers) == 3 else context["span"]
    return (
        sum(marginal_logs) / len(marginal_logs),
        sum(pair_logs) / len(pair_logs),
        shape_map[classify_digit_shape(numbers)],
        sum_map[sum(numbers)],
        span_map[max(numbers) - min(numbers)],
        context["parity"][_parity_label(numbers)],
        context["bigSmall"][_big_small_label(numbers)],
        context["primeComposite"][digit_prime_composite_label(numbers)],
        context["consecutive"][digit_consecutive_count(numbers)],
        context["mirror"][digit_mirror_count(numbers)],
        context["sumTail"][digit_sum_tail(numbers)],
        context["latestDistance"][
            digit_latest_distance(numbers, context["latestNumbers"])
        ],
        context["repeatLatest"][
            digit_repeat_latest_count(numbers, context["latestNumbers"])
        ],
        sum(omission_scores) / len(omission_scores),
        0.0,
        0.0,
    )


def _rank_percentiles(values: Sequence[float]) -> list[float]:
    """把不同量纲的子模型分数转换为带并列中位排名的 0-1 分位。"""

    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    start = 0
    denominator = len(values) - 1
    while start < len(order):
        end = start
        value = values[order[start]]
        while end + 1 < len(order) and abs(values[order[end + 1]] - value) <= 1e-12:
            end += 1
        percentile = ((start + end) / 2) / denominator
        for offset in range(start, end + 1):
            output[order[offset]] = percentile
        start = end + 1
    return output


def _score_prefix_with_context(
    numbers: Sequence[int],
    config: DigitCandidateConfig,
    context: dict[str, Any],
) -> float:
    positions = context["positions"]
    marginal_logs = [
        context["marginal"][positions[index]][int(numbers[index])] for index in range(3)
    ]
    pair_logs = [
        context["pair"][f"{left}-{right}"][(int(numbers[left]), int(numbers[right]))]
        for left in range(3)
        for right in range(left + 1, 3)
    ]
    omission_logs = [
        context["omission"].get(positions[index], {}).get(int(numbers[index]), 0.0)
        for index in range(3)
    ]
    score = config.marginal_weight * sum(marginal_logs) / len(marginal_logs)
    score += config.pair_weight * sum(pair_logs) / len(pair_logs)
    score += config.shape_weight * context["prefixShape"][classify_digit_shape(numbers)]
    score += config.sum_weight * context["prefixSum"][sum(numbers)]
    score += config.span_weight * context["prefixSpan"][max(numbers) - min(numbers)]
    score += config.omission_weight * sum(omission_logs) / len(omission_logs)
    return score


def _score_numbers_with_context(
    numbers: Sequence[int],
    config: DigitCandidateConfig,
    context: dict[str, Any],
) -> float:
    if len(numbers) == 3:
        return _score_prefix_with_context(numbers, config, context)
    prefix_score = _score_prefix_with_context(numbers[:3], config, context)
    positions = context["positions"]
    suffix_indexes = tuple(range(3, len(numbers)))
    suffix_marginals = [
        context["marginal"][positions[index]][int(numbers[index])]
        for index in suffix_indexes
    ]
    suffix_pairs = [
        context["pair"][f"{left}-{right}"][(int(numbers[left]), int(numbers[right]))]
        for left in range(len(numbers))
        for right in range(max(left + 1, 3), len(numbers))
    ]
    suffix_omission = [
        context["omission"].get(positions[index], {}).get(int(numbers[index]), 0.0)
        for index in suffix_indexes
    ]
    score = prefix_score
    if suffix_marginals:
        score += config.marginal_weight * sum(suffix_marginals) / len(suffix_marginals)
    if suffix_pairs:
        score += config.pair_weight * sum(suffix_pairs) / len(suffix_pairs)
    score += config.shape_weight * context["shape"][classify_digit_shape(numbers)]
    score += config.sum_weight * context["sum"][sum(numbers)]
    score += config.span_weight * context["span"][max(numbers) - min(numbers)]
    if suffix_omission:
        score += config.omission_weight * sum(suffix_omission) / len(suffix_omission)
    return score


def _omission_score(
    stats: DigitStatisticsResult,
    indexes: Sequence[int],
    numbers: Sequence[int],
    config: DigitCandidateConfig,
) -> float:
    positions = list(stats.position_frequency)
    scores = []
    for index, digit in zip(indexes, numbers):
        scores.append(
            _weighted_omission_score(stats, positions[index], int(digit), config)
        )
    return sum(scores) / len(scores) if scores else 0.0


def score_digit_prefix(
    stats: DigitStatisticsResult,
    numbers: Sequence[int],
    config: DigitCandidateConfig | None = None,
) -> float:
    """使用共享三位前缀模型计算启发式复合对数评分。

    排列三与排列五的前三位都调用本函数；排列五后两位只作为附加分。
    该评分由重叠特征加权得到，不是规范联合概率。
    """

    if len(numbers) != 3:
        raise ValueError("三位前缀评分必须提供恰好 3 个数字")
    config = config or DigitCandidateConfig()
    return _score_prefix_with_context(
        numbers, config, _build_score_context(stats, config)
    )


def score_digit_numbers(
    stats: DigitStatisticsResult,
    numbers: Sequence[int],
    config: DigitCandidateConfig | None = None,
) -> float:
    """计算完整数字彩候选的启发式复合对数评分，不是联合概率。"""

    config = config or DigitCandidateConfig()
    if len(numbers) != stats.draw_count:
        raise ValueError(f"候选位数应为 {stats.draw_count}")
    return _score_numbers_with_context(
        numbers, config, _build_score_context(stats, config)
    )


def _digit_score(
    stats: DigitStatisticsResult,
    position: str,
    digit: int,
    config: DigitCandidateConfig,
) -> float:
    """兼容位置排序辅助函数；主候选使用复合模型分。"""

    return _position_log_probability(stats, position, digit, config)


def _ranked_digits(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    position: str,
    index: int,
    config: DigitCandidateConfig,
) -> list[tuple[int, float]]:
    spec = rule.ball_specs[index]
    scored = [
        (digit, _digit_score(stats, position, digit, config))
        for digit in range(spec.min_number, spec.max_number + 1)
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _effective_config(
    rule: LotteryRule, config: DigitCandidateConfig
) -> DigitCandidateConfig:
    """补齐更适合实战候选的默认过滤条件。"""

    is_default_fc3d_profile = (
        rule.code == "fc3d"
        and config.marginal_weight == DigitCandidateConfig.marginal_weight
        and config.pair_weight == DigitCandidateConfig.pair_weight
        and config.shape_weight == DigitCandidateConfig.shape_weight
        and config.sum_weight == DigitCandidateConfig.sum_weight
        and config.span_weight == DigitCandidateConfig.span_weight
        and config.omission_weight == DigitCandidateConfig.omission_weight
    )
    if rule.draw_count == 3:
        return DigitCandidateConfig(
            count=config.count,
            sum_min=6 if config.sum_min is None else config.sum_min,
            sum_max=21 if config.sum_max is None else config.sum_max,
            span_min=1 if config.span_min is None else config.span_min,
            span_max=9 if config.span_max is None else config.span_max,
            allowed_shapes=(
                ("组三", "组六")
                if config.allowed_shapes is None
                else config.allowed_shapes
            ),
            top_digits_per_position=config.top_digits_per_position,
            frequency_weight=config.frequency_weight,
            omission_weight=0.03 if is_default_fc3d_profile else config.omission_weight,
            random_weight=config.random_weight,
            exclude_latest=config.exclude_latest,
            frequency_windows=config.frequency_windows,
            frequency_window_weights=config.frequency_window_weights,
            omission_cap=config.omission_cap,
            diversity_weight=config.diversity_weight,
            high_score_pool_factor=config.high_score_pool_factor,
            marginal_weight=config.marginal_weight,
            pair_weight=0.0 if is_default_fc3d_profile else config.pair_weight,
            shape_weight=0.0 if is_default_fc3d_profile else config.shape_weight,
            sum_weight=0.0 if is_default_fc3d_profile else config.sum_weight,
            span_weight=0.0 if is_default_fc3d_profile else config.span_weight,
            score_floor=config.score_floor,
            ranking_mode=config.ranking_mode,
            ensemble_model_weights=config.ensemble_model_weights,
            ensemble_score_floor=config.ensemble_score_floor,
            constraint_mode=config.constraint_mode,
            constraint_probability_floor=config.constraint_probability_floor,
            constraint_penalty_weight=config.constraint_penalty_weight,
        )
    if rule.draw_count == 5:
        return DigitCandidateConfig(
            count=config.count,
            sum_min=10 if config.sum_min is None else config.sum_min,
            sum_max=35 if config.sum_max is None else config.sum_max,
            span_min=3 if config.span_min is None else config.span_min,
            span_max=9 if config.span_max is None else config.span_max,
            allowed_shapes=(
                ("全不同", "二一一一", "二二一", "三一一", "三二")
                if config.allowed_shapes is None
                else config.allowed_shapes
            ),
            top_digits_per_position=config.top_digits_per_position,
            frequency_weight=config.frequency_weight,
            omission_weight=config.omission_weight,
            random_weight=config.random_weight,
            exclude_latest=config.exclude_latest,
            frequency_windows=config.frequency_windows,
            frequency_window_weights=config.frequency_window_weights,
            omission_cap=config.omission_cap,
            diversity_weight=config.diversity_weight,
            high_score_pool_factor=config.high_score_pool_factor,
            marginal_weight=config.marginal_weight,
            pair_weight=config.pair_weight,
            shape_weight=config.shape_weight,
            sum_weight=config.sum_weight,
            span_weight=config.span_weight,
            score_floor=config.score_floor,
            ranking_mode=config.ranking_mode,
            ensemble_model_weights=config.ensemble_model_weights,
            ensemble_score_floor=config.ensemble_score_floor,
            constraint_mode=config.constraint_mode,
            constraint_probability_floor=config.constraint_probability_floor,
            constraint_penalty_weight=config.constraint_penalty_weight,
        )
    return config


def _passes_filters(numbers: Sequence[int], config: DigitCandidateConfig) -> bool:
    sum_value = sum(numbers)
    span = max(numbers) - min(numbers)
    shape = classify_digit_shape(numbers)
    if config.sum_min is not None and sum_value < config.sum_min:
        return False
    if config.sum_max is not None and sum_value > config.sum_max:
        return False
    if config.span_min is not None and span < config.span_min:
        return False
    if config.span_max is not None and span > config.span_max:
        return False
    if config.allowed_shapes is not None and shape not in config.allowed_shapes:
        return False
    return True


def _candidate_text(numbers: Sequence[int]) -> str:
    return "".join(str(int(number)) for number in numbers)


def _make_candidate(numbers: list[int], score: float) -> DigitCandidate:
    return DigitCandidate(
        numbers=numbers,
        text=_candidate_text(numbers),
        sum_value=sum(numbers),
        span=max(numbers) - min(numbers),
        shape=classify_digit_shape(numbers),
        score=round(score, 6),
        joint_probability=math.exp(score),
    )


@lru_cache(maxsize=8)
def _digit_universe(
    draw_count: int,
) -> tuple[tuple[tuple[int, ...], str, int, int, str], ...]:
    """缓存 0-9 数字彩号码空间的静态形态信息。"""

    rows = []
    for values in itertools.product(range(10), repeat=draw_count):
        text = "".join(str(number) for number in values)
        rows.append(
            (
                values,
                text,
                sum(values),
                max(values) - min(values),
                classify_digit_shape(values),
            )
        )
    return tuple(rows)


def _passes_cached_filters(
    sum_value: int, span: int, shape: str, config: DigitCandidateConfig
) -> bool:
    if config.sum_min is not None and sum_value < config.sum_min:
        return False
    if config.sum_max is not None and sum_value > config.sum_max:
        return False
    if config.span_min is not None and span < config.span_min:
        return False
    if config.span_max is not None and span > config.span_max:
        return False
    return config.allowed_shapes is None or shape in config.allowed_shapes


def _shape_budget(
    rule: LotteryRule, config: DigitCandidateConfig, count: int
) -> dict[str, int]:
    """返回统计与随机策略共享的严格形态预算。"""

    allowed = set(config.allowed_shapes or ())
    if rule.draw_count == 3:
        has_group6 = not allowed or "组六" in allowed
        has_group3 = not allowed or "组三" in allowed
        if has_group6 and not has_group3:
            return {"mainstream": count, "defensive": 0}
        if has_group3 and not has_group6:
            return {"mainstream": 0, "defensive": count}
        defensive = int(count * 0.2)
        return {"mainstream": count - defensive, "defensive": defensive}
    if rule.draw_count == 5:
        mainstream_shapes = {"全不同", "二一一一", "二二一"}
        defensive_shapes = {"三一一", "三二"}
        has_mainstream = not allowed or bool(allowed.intersection(mainstream_shapes))
        has_defensive = not allowed or bool(allowed.intersection(defensive_shapes))
        if has_mainstream and not has_defensive:
            return {"mainstream": count, "defensive": 0}
        if has_defensive and not has_mainstream:
            return {"mainstream": 0, "defensive": count}
        defensive = min(3, int(count * 0.15))
        return {"mainstream": count - defensive, "defensive": defensive}
    return {"mainstream": count, "defensive": 0}


def _is_defensive_shape(rule: LotteryRule, shape: str) -> bool:
    if rule.draw_count == 3:
        return shape == "组三"
    if rule.draw_count == 5:
        return shape in {"三一一", "三二"}
    return False


def _selection_score(candidate: DigitCandidate, config: DigitCandidateConfig) -> float:
    if config.ranking_mode == "ensemble":
        return candidate.ensemble_score
    return candidate.score


def _diversity_select(
    ranked_candidates: Sequence[DigitCandidate],
    count: int,
    config: DigitCandidateConfig,
    *,
    existing: Sequence[DigitCandidate] = (),
) -> list[DigitCandidate]:
    """从高分池中确定性选择位置覆盖更分散的候选。"""

    if count <= 0:
        return []
    if not ranked_candidates:
        return []
    best_score = _selection_score(ranked_candidates[0], config)
    score_only_floor = _selection_score(
        ranked_candidates[min(count, len(ranked_candidates)) - 1], config
    )
    configured_floor = best_score - (
        config.ensemble_score_floor
        if config.ranking_mode == "ensemble"
        else config.score_floor
    )
    quality_pool = [
        candidate
        for candidate in ranked_candidates
        if _selection_score(candidate, config) >= configured_floor - 1e-12
    ]
    if len(quality_pool) < count:
        quality_pool = list(ranked_candidates[:count])
    else:
        quality_pool = [
            candidate
            for candidate in quality_pool
            if _selection_score(candidate, config) >= score_only_floor - 1e-12
        ]
    pool_size = max(200, count * max(1, config.high_score_pool_factor))
    remaining = list(quality_pool[:pool_size])
    selected = list(existing)
    output: list[DigitCandidate] = []
    while remaining and len(output) < count:
        best_index = 0
        best_key: tuple[float, float, str] | None = None
        for index, candidate in enumerate(remaining):
            repeated_positions = 0.0
            unseen_positions = 0
            nearest_similarity = 0.0
            if selected:
                for position, digit in enumerate(candidate.numbers):
                    position_repeats = sum(
                        item.numbers[position] == digit for item in selected
                    )
                    repeated_positions += position_repeats
                    if position_repeats == 0:
                        unseen_positions += 1
                nearest_similarity = max(
                    sum(
                        left == right
                        for left, right in zip(candidate.numbers, item.numbers)
                    )
                    / len(candidate.numbers)
                    for item in selected
                )
            raw_diversity = (
                2.0 * unseen_positions - repeated_positions - nearest_similarity
            )
            diversity = max(-1.0, min(1.0, raw_diversity))
            ranking_score = _selection_score(candidate, config)
            same_score = round(ranking_score, 3)
            diversity_scale = 0.1 if config.ranking_mode == "ensemble" else 1.0
            adjusted = (
                same_score + config.diversity_weight * diversity_scale * diversity
            )
            key = (
                adjusted,
                ranking_score,
                "".join(chr(255 - ord(char)) for char in candidate.text),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        output.append(chosen)
    return output


def _enumerate_scored_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitCandidateConfig,
    latest_text_for_exclusion: str | None,
    external_scores: DigitExternalModelScores | None = None,
) -> list[DigitCandidate]:
    """枚举过滤空间并按启发式复合模型分排序。"""

    rows: list[
        tuple[list[int], str, int, int, str, float, tuple[float, ...], float]
    ] = []
    context = _build_score_context(stats, config)
    for values, text, sum_value, span, shape in _digit_universe(rule.draw_count):
        numbers = list(values)
        if not _passes_cached_filters(sum_value, span, shape, config):
            continue
        if config.exclude_latest and text == latest_text_for_exclusion:
            continue
        constraint_penalty = _structure_constraint_penalty(numbers, context, config)
        if math.isinf(constraint_penalty):
            continue
        score = (
            _score_numbers_with_context(numbers, config, context) - constraint_penalty
        )
        components = ()
        if config.ranking_mode == "ensemble":
            components = _model_component_scores(numbers, context)
            if external_scores is not None:
                components = (*components[:-2], *external_scores.values_for(text))
        rows.append(
            (
                numbers,
                text,
                sum_value,
                span,
                shape,
                score,
                components,
                constraint_penalty,
            )
        )
    if not rows:
        return []
    maximum = max(row[5] for row in rows)
    total_weight = sum(math.exp(row[5] - maximum) for row in rows)
    model_percentiles: list[list[float]] = []
    if config.ranking_mode == "ensemble":
        model_percentiles = [
            _rank_percentiles([row[6][model_index] for row in rows])
            for model_index in range(len(ENSEMBLE_MODEL_NAMES))
        ]
    ensemble_weight_total = sum(config.ensemble_model_weights)
    candidates = [
        DigitCandidate(
            numbers,
            text,
            sum_value,
            span,
            shape,
            round(score, 6),
            math.exp(score - maximum) / total_weight if total_weight else 0.0,
            (
                round(
                    max(
                        0.0,
                        sum(
                            weight * model_percentiles[model_index][row_index]
                            for model_index, weight in enumerate(
                                config.ensemble_model_weights
                            )
                        )
                        / ensemble_weight_total
                        - constraint_penalty,
                    ),
                    6,
                )
                if model_percentiles
                else 0.0
            ),
            (
                tuple(
                    model_percentiles[model_index][row_index]
                    for model_index in range(len(model_percentiles))
                )
                if model_percentiles
                else ()
            ),
            round(constraint_penalty, 6),
        )
        for row_index, (
            numbers,
            text,
            sum_value,
            span,
            shape,
            score,
            _,
            constraint_penalty,
        ) in enumerate(rows)
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            -_selection_score(candidate, config),
            -candidate.score,
            candidate.text,
        ),
    )


_SCORED_POOL_CACHE: OrderedDict[
    tuple[int, str, DigitCandidateConfig],
    tuple[DigitStatisticsResult, list[DigitCandidate]],
] = OrderedDict()


def _scored_candidate_pool(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitCandidateConfig,
    external_scores: DigitExternalModelScores | None = None,
) -> list[DigitCandidate]:
    """缓存同一期统计、同一配置的完整评分池，供多随机基线复用。"""

    if external_scores is not None:
        latest_text = (
            _candidate_text(stats.latest_numbers) if stats.latest_numbers else None
        )
        return _enumerate_scored_candidates(
            stats,
            rule,
            config,
            latest_text,
            external_scores,
        )
    key = (id(stats), rule.code, config)
    cached = _SCORED_POOL_CACHE.get(key)
    if cached is not None and cached[0] is stats:
        _SCORED_POOL_CACHE.move_to_end(key)
        return cached[1]
    latest_text = (
        _candidate_text(stats.latest_numbers) if stats.latest_numbers else None
    )
    ranked = _enumerate_scored_candidates(stats, rule, config, latest_text)
    _SCORED_POOL_CACHE[key] = (stats, ranked)
    _SCORED_POOL_CACHE.move_to_end(key)
    while len(_SCORED_POOL_CACHE) > 8:
        _SCORED_POOL_CACHE.popitem(last=False)
    return ranked


def _order_three_digit_candidates_by_shape_probability(
    candidates: Sequence[DigitCandidate], config: DigitCandidateConfig
) -> list[DigitCandidate]:
    """按形态概率重排三位数字彩候选。

    组三理论概率只有 27%，用户查看前 5 注时不能被高分组三挤占。
    因此前缀按“4 注组六/其他 + 1 注组三”的节奏输出，豹子仍由过滤器默认排除。
    """

    if config.allowed_shapes is not None and "组六" not in config.allowed_shapes:
        return list(candidates)
    group3 = [candidate for candidate in candidates if candidate.shape == "组三"]
    others = [candidate for candidate in candidates if candidate.shape != "组三"]
    ordered: list[DigitCandidate] = []

    def take(source: list[DigitCandidate], amount: int) -> None:
        while source and amount > 0 and len(ordered) < config.count:
            ordered.append(source.pop(0))
            amount -= 1

    first_block = True
    while len(ordered) < config.count and (others or group3):
        take(others, 4 if first_block else 3)
        take(group3, 1)
        first_block = False
        if not group3:
            take(others, config.count - len(ordered))
        if not others:
            take(group3, config.count - len(ordered))
    return ordered[: config.count]


def _order_pl5_candidates_by_shape_probability(
    candidates: Sequence[DigitCandidate], config: DigitCandidateConfig
) -> list[DigitCandidate]:
    """让排列五防守形态均匀插入，避免挤占主流形态。"""

    defensive_shapes = {"三一一", "三二"}
    defensive = [
        candidate for candidate in candidates if candidate.shape in defensive_shapes
    ]
    mainstream = [
        candidate for candidate in candidates if candidate.shape not in defensive_shapes
    ]
    if not mainstream:
        return list(candidates)[: config.count]
    ordered: list[DigitCandidate] = []
    while len(ordered) < config.count and (mainstream or defensive):
        ordered.extend(mainstream[:4])
        del mainstream[:4]
        if defensive and len(ordered) < config.count:
            ordered.append(defensive.pop(0))
        if not defensive:
            ordered.extend(mainstream[: config.count - len(ordered)])
            break
        if not mainstream:
            ordered.extend(defensive[: config.count - len(ordered)])
            break
    return ordered[: config.count]


def _build_model_candidate_lists(
    ranked: Sequence[DigitCandidate],
    rule: LotteryRule,
    config: DigitCandidateConfig,
    external_scores: DigitExternalModelScores | None,
) -> dict[str, list[str]]:
    """在相同过滤空间和形态预算下保存每个子模型的 Top 候选。"""

    if config.ranking_mode != "ensemble" or not ranked:
        return {}
    budget = _shape_budget(rule, config, config.count)
    output: dict[str, list[str]] = {}
    for model_index, model_name in enumerate(ENSEMBLE_MODEL_NAMES):
        if model_name == "monteCarlo" and not (
            external_scores is not None and external_scores.monte_carlo
        ):
            continue
        if model_name == "mlRanker" and not (
            external_scores is not None and external_scores.ml_ranker
        ):
            continue

        def ranking_key(candidate: DigitCandidate) -> tuple[float, float, str]:
            return (
                -candidate.model_rank_percentiles[model_index],
                -candidate.score,
                candidate.text,
            )

        mainstream = heapq.nsmallest(
            budget["mainstream"],
            (
                candidate
                for candidate in ranked
                if not _is_defensive_shape(rule, candidate.shape)
            ),
            key=ranking_key,
        )
        defensive = heapq.nsmallest(
            budget["defensive"],
            (
                candidate
                for candidate in ranked
                if _is_defensive_shape(rule, candidate.shape)
            ),
            key=ranking_key,
        )
        selected = [*mainstream, *defensive]
        output[model_name] = [candidate.text for candidate in selected]
    return output


def generate_digit_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    *,
    config: DigitCandidateConfig | None = None,
    seed: int | None = None,
    external_scores: DigitExternalModelScores | None = None,
) -> DigitCandidateResult:
    """生成数字彩候选号码。

    ``seed`` 为兼容旧调用保留；当前实现采用全空间确定性评分，同一份统计输入
    不再因随机种子变化而改变结果。
    """

    if rule.category != "digit":
        raise ValueError(f"数字彩候选生成不适用于玩法：{rule.display_name}")
    config = _effective_config(rule, config or DigitCandidateConfig())
    _ = seed
    ranked = _scored_candidate_pool(stats, rule, config, external_scores)
    model_candidates = _build_model_candidate_lists(
        ranked, rule, config, external_scores
    )
    budget = _shape_budget(rule, config, config.count)
    mainstream = [
        candidate
        for candidate in ranked
        if not _is_defensive_shape(rule, candidate.shape)
    ]
    defensive = [
        candidate for candidate in ranked if _is_defensive_shape(rule, candidate.shape)
    ]
    selected_mainstream = _diversity_select(mainstream, budget["mainstream"], config)
    selected_defensive = _diversity_select(
        defensive,
        budget["defensive"],
        config,
        existing=selected_mainstream,
    )
    candidates = [*selected_mainstream, *selected_defensive]

    if rule.draw_count == 3:
        candidates = _order_three_digit_candidates_by_shape_probability(
            candidates, config
        )
    elif rule.draw_count == 5:
        candidates = _order_pl5_candidates_by_shape_probability(candidates, config)
    if len(candidates) != config.count:
        raise ValueError(
            f"候选不足：请求 {config.count} 注，过滤后只能生成 {len(candidates)} 注"
        )
    return DigitCandidateResult(
        rule_code=rule.code,
        display_name=rule.display_name,
        candidates=candidates,
        config=config,
        model_candidates=model_candidates,
    )


def generate_digit_betting_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    *,
    config: DigitCandidateConfig | None = None,
    group_count: int | None = None,
    external_scores: DigitExternalModelScores | None = None,
) -> DigitBettingCandidateResult:
    """生成直选与组选分离的数字彩投注候选。

    三位彩组选按所有有序排列的复合模型权重聚合，并在组选过滤空间归一化；
    该质量不是实际开奖概率。排列五不提供组选。
    """

    direct = generate_digit_candidates(
        stats,
        rule,
        config=config,
        external_scores=external_scores,
    )
    if rule.draw_count != 3:
        return DigitBettingCandidateResult(
            rule.code,
            rule.display_name,
            direct.candidates,
            [],
            direct.config,
            direct.model_candidates,
        )
    requested_group_count = (
        direct.config.count if group_count is None else int(group_count)
    )
    if requested_group_count < 0:
        raise ValueError("group_count 不得为负数")
    if requested_group_count == 0:
        return DigitBettingCandidateResult(
            rule.code,
            rule.display_name,
            direct.candidates,
            [],
            direct.config,
            direct.model_candidates,
        )

    ordered = _scored_candidate_pool(stats, rule, direct.config, external_scores)
    ordered = [
        candidate for candidate in ordered if candidate.shape in {"组六", "组三"}
    ]
    if not ordered:
        groups: list[DigitGroupCandidate] = []
    else:
        raw_mass: dict[str, float] = {}
        ensemble_components: dict[str, list[float]] = {}
        group_shape: dict[str, str] = {}
        permutation_counts: Counter[str] = Counter()
        for candidate in ordered:
            key = "".join(sorted(candidate.text))
            raw_mass[key] = raw_mass.get(key, 0.0) + candidate.composite_model_weight
            if candidate.model_rank_percentiles:
                totals = ensemble_components.setdefault(
                    key, [0.0] * len(ENSEMBLE_MODEL_NAMES)
                )
                for index, value in enumerate(candidate.model_rank_percentiles):
                    totals[index] += value
            group_shape[key] = candidate.shape
            permutation_counts[key] += 1
        total_mass = sum(raw_mass.values())
        component_means = {
            key: tuple(
                value / permutation_counts[key]
                for value in ensemble_components.get(key, [])
            )
            for key in raw_mass
        }
        shape_model_percentiles: dict[str, tuple[float, ...]] = {}
        if direct.config.ranking_mode == "ensemble":
            for shape in ("组六", "组三"):
                shape_keys = [key for key in raw_mass if group_shape[key] == shape]
                ranked_components = [
                    _rank_percentiles(
                        [component_means[key][index] for key in shape_keys]
                    )
                    for index in range(len(ENSEMBLE_MODEL_NAMES))
                ]
                for key_index, key in enumerate(shape_keys):
                    shape_model_percentiles[key] = tuple(
                        ranked_components[index][key_index]
                        for index in range(len(ENSEMBLE_MODEL_NAMES))
                    )
        weight_total = sum(direct.config.ensemble_model_weights)
        groups = [
            DigitGroupCandidate(
                group_key=key,
                numbers=[int(value) for value in key],
                shape=group_shape[key],
                probability_mass=mass / total_mass if total_mass else 0.0,
                score=round(math.log(max(mass, 1e-300)), 6),
                permutations=permutation_counts[key],
                ensemble_score=round(
                    (
                        sum(
                            weight * value
                            for weight, value in zip(
                                direct.config.ensemble_model_weights,
                                shape_model_percentiles.get(key, ()),
                            )
                        )
                        / weight_total
                        if shape_model_percentiles.get(key)
                        else 0.0
                    ),
                    6,
                ),
                model_rank_percentiles=shape_model_percentiles.get(key, ()),
                ranking_model=(
                    "shape_specific_ensemble"
                    if direct.config.ranking_mode == "ensemble"
                    else "composite_aggregation"
                ),
            )
            for key, mass in raw_mass.items()
        ]

        def group_ranking(candidate: DigitGroupCandidate) -> float:
            if direct.config.ranking_mode == "ensemble":
                return candidate.ensemble_score
            return candidate.probability_mass

        groups.sort(
            key=lambda candidate: (-group_ranking(candidate), candidate.group_key)
        )
        budget = _shape_budget(rule, direct.config, requested_group_count)
        group6 = [candidate for candidate in groups if candidate.shape == "组六"]
        group3 = [candidate for candidate in groups if candidate.shape == "组三"]
        if len(group6) < budget["mainstream"] or len(group3) < budget["defensive"]:
            raise ValueError(
                f"组选候选不足：请求 {requested_group_count} 注，过滤后无法满足形态预算"
            )
        selected_group6 = group6[: budget["mainstream"]]
        selected_group3 = group3[: budget["defensive"]]
        groups = sorted(
            [*selected_group6, *selected_group3],
            key=lambda candidate: (-group_ranking(candidate), candidate.group_key),
        )[:requested_group_count]
    if len(groups) != requested_group_count:
        raise ValueError(
            f"组选候选不足：请求 {requested_group_count} 注，过滤后只能生成 {len(groups)} 注"
        )
    return DigitBettingCandidateResult(
        rule.code,
        rule.display_name,
        direct.candidates,
        groups,
        direct.config,
        direct.model_candidates,
    )


def rank_digit_numbers(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    numbers: Sequence[int],
    config: DigitCandidateConfig | None = None,
    external_scores: DigitExternalModelScores | None = None,
) -> tuple[int, float]:
    """兼容返回目标号码在过滤空间中的复合模型排名与评分。

    该函数只计数，不构造 10 万个候选对象，供嵌套前推的密集指标使用。
    """

    rank, score, _ = rank_digit_numbers_with_eligible_count(
        stats, rule, numbers, config, external_scores
    )
    return rank, score


def rank_digit_numbers_with_eligible_count(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    numbers: Sequence[int],
    config: DigitCandidateConfig | None = None,
    external_scores: DigitExternalModelScores | None = None,
) -> tuple[int, float, int]:
    """返回复合模型排名、评分与过滤空间大小，供归一化排名诊断。

    排列五完整过滤空间较大；这里复用候选池缓存，避免严格前推中对同一期
    先生成候选、再为目标排名重复枚举评分。
    """

    effective = _effective_config(rule, config or DigitCandidateConfig())
    target_numbers = [int(number) for number in numbers]
    target_score = score_digit_numbers(stats, target_numbers, effective)
    target_text = _candidate_text(target_numbers)
    latest_text = (
        _candidate_text(stats.latest_numbers) if stats.latest_numbers else None
    )
    target_in_space = _passes_filters(target_numbers, effective) and not (
        effective.exclude_latest and target_text == latest_text
    )
    ranked = _scored_candidate_pool(stats, rule, effective, external_scores)
    eligible = len(ranked)
    if target_in_space:
        for rank, candidate in enumerate(ranked, 1):
            if candidate.text == target_text:
                return rank, _selection_score(candidate, effective), eligible
    return eligible + 1, target_score, eligible


def generate_uniform_digit_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    *,
    config: DigitCandidateConfig | None = None,
    seed: int | str = 0,
) -> DigitCandidateResult:
    """在相同过滤与形态配额下均匀随机生成基线候选。"""

    if rule.category != "digit":
        raise ValueError(f"数字彩候选生成不适用于玩法：{rule.display_name}")
    config = _effective_config(rule, config or DigitCandidateConfig())
    universe = _scored_candidate_pool(stats, rule, config)
    rng = random.Random(str(seed))
    if len(universe) < config.count:
        raise ValueError(
            f"候选不足：请求 {config.count} 注，过滤后只能生成 {len(universe)} 注"
        )
    if rule.draw_count in {3, 5}:
        budget = _shape_budget(rule, config, config.count)
        defensive = [
            candidate
            for candidate in universe
            if _is_defensive_shape(rule, candidate.shape)
        ]
        mainstream = [
            candidate
            for candidate in universe
            if not _is_defensive_shape(rule, candidate.shape)
        ]
        if (
            len(mainstream) < budget["mainstream"]
            or len(defensive) < budget["defensive"]
        ):
            raise ValueError(
                f"候选不足：请求 {config.count} 注，过滤后无法满足形态预算"
            )
        candidates = rng.sample(mainstream, budget["mainstream"]) + rng.sample(
            defensive, budget["defensive"]
        )
    else:
        candidates = rng.sample(universe, config.count)
    if rule.draw_count == 3:
        candidates = _order_three_digit_candidates_by_shape_probability(
            candidates, config
        )
    elif rule.draw_count == 5:
        candidates = _order_pl5_candidates_by_shape_probability(candidates, config)
    if len(candidates) != config.count:
        raise ValueError(
            f"候选不足：请求 {config.count} 注，过滤后只能生成 {len(candidates)} 注"
        )
    return DigitCandidateResult(
        rule_code=rule.code,
        display_name=rule.display_name,
        candidates=candidates,
        config=config,
    )


def generate_uniform_digit_betting_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    *,
    config: DigitCandidateConfig | None = None,
    group_count: int | None = None,
    seed: int | str = 0,
) -> DigitBettingCandidateResult:
    """生成与统计策略同口径的随机直选/组选基线。"""

    direct = generate_uniform_digit_candidates(stats, rule, config=config, seed=seed)
    if rule.draw_count != 3:
        return DigitBettingCandidateResult(
            rule.code, rule.display_name, direct.candidates, [], direct.config
        )
    requested = direct.config.count if group_count is None else int(group_count)
    if requested < 0:
        raise ValueError("group_count 不得为负数")
    if requested == 0:
        return DigitBettingCandidateResult(
            rule.code, rule.display_name, direct.candidates, [], direct.config
        )

    permutation_counts: Counter[str] = Counter()
    group_shapes: dict[str, str] = {}
    for candidate in _scored_candidate_pool(stats, rule, direct.config):
        if candidate.shape not in {"组六", "组三"}:
            continue
        key = "".join(sorted(candidate.text))
        permutation_counts[key] += 1
        group_shapes[key] = candidate.shape

    budget = _shape_budget(rule, direct.config, requested)
    group6 = sorted(key for key, shape in group_shapes.items() if shape == "组六")
    group3 = sorted(key for key, shape in group_shapes.items() if shape == "组三")
    if len(group6) < budget["mainstream"] or len(group3) < budget["defensive"]:
        raise ValueError(f"组选候选不足：请求 {requested} 注，过滤后无法满足形态预算")
    rng = random.Random(f"{seed}:group")
    selected_keys = rng.sample(group6, budget["mainstream"]) + rng.sample(
        group3, budget["defensive"]
    )
    total_permutations = sum(permutation_counts.values())
    groups = [
        DigitGroupCandidate(
            group_key=key,
            numbers=[int(value) for value in key],
            shape=group_shapes[key],
            probability_mass=(
                permutation_counts[key] / total_permutations
                if total_permutations
                else 0.0
            ),
            score=(
                math.log(permutation_counts[key] / total_permutations)
                if total_permutations
                else -math.inf
            ),
            permutations=permutation_counts[key],
        )
        for key in selected_keys
    ]
    groups.sort(key=lambda candidate: (candidate.shape == "组三", candidate.group_key))
    if len(groups) != requested:
        raise ValueError(
            f"组选候选不足：请求 {requested} 注，过滤后只能生成 {len(groups)} 注"
        )
    return DigitBettingCandidateResult(
        rule.code, rule.display_name, direct.candidates, groups, direct.config
    )
