# -*- coding: utf-8 -*-
"""数字彩预测相对精确随机基线的统计可行性闸门。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

DEFAULT_MIN_PERIODS = 500
DEFAULT_SIGNIFICANCE_LEVEL = 0.01
DEFAULT_MIN_RELATIVE_LIFT = 0.25
DEFAULT_BLOCK_COUNT = 3
_WILSON_Z_99 = NormalDist().inv_cdf(0.995)


@dataclass(frozen=True)
class ViabilityBlock:
    """一个非重叠时间块的命中与随机基线对比。"""

    index: int
    start_period: int
    end_period: int
    periods: int
    hits: int
    actual_hit_rate: float
    random_hit_probability: float
    expected_random_hits: float
    meets_random_baseline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "startPeriod": self.start_period,
            "endPeriod": self.end_period,
            "periods": self.periods,
            "hits": self.hits,
            "actualHitRate": self.actual_hit_rate,
            "randomHitProbability": self.random_hit_probability,
            "expectedRandomHits": self.expected_random_hits,
            "meetsRandomBaseline": self.meets_random_baseline,
        }


@dataclass(frozen=True)
class ViabilityMetric:
    """直选或组选单项统计闸门结果。"""

    metric: str
    periods: int
    hits: int
    actual_hit_rate: float
    random_hit_probability: float
    expected_random_hits: float
    relative_lift: float | None
    p_value: float
    wilson_lower_bound_99: float
    blocks: list[ViabilityBlock]
    conditions: dict[str, bool]
    viable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "periods": self.periods,
            "hits": self.hits,
            "actualHitRate": self.actual_hit_rate,
            "randomHitProbability": self.random_hit_probability,
            "expectedRandomHits": self.expected_random_hits,
            "relativeLift": self.relative_lift,
            "pValue": self.p_value,
            "wilsonLowerBound99": self.wilson_lower_bound_99,
            "blocks": [block.to_dict() for block in self.blocks],
            "conditions": self.conditions,
            "viable": self.viable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PredictionViabilityReport:
    """同一策略的直选与组选统计可行性报告。"""

    viable: bool
    direct_gate: ViabilityMetric
    group_gate: ViabilityMetric | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "viable": self.viable,
            "directGate": self.direct_gate.to_dict(),
            "groupGate": self.group_gate.to_dict() if self.group_gate else None,
            "reason": self.reason,
        }


def poisson_binomial_right_tail(
    probabilities: Sequence[float], observed_hits: int
) -> float:
    """计算独立但概率可不同的伯努利和 ``P(X >= observed_hits)``。

    示例：``poisson_binomial_right_tail([0.1, 0.2, 0.3], 2)``。
    """

    values = [float(value) for value in probabilities]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("随机命中概率必须在 0 到 1 之间")
    if observed_hits <= 0:
        return 1.0
    if observed_hits > len(values):
        return 0.0
    distribution = [0.0] * (len(values) + 1)
    distribution[0] = 1.0
    for period_index, probability in enumerate(values, start=1):
        for hits in range(period_index, 0, -1):
            distribution[hits] = (
                distribution[hits] * (1.0 - probability)
                + distribution[hits - 1] * probability
            )
        distribution[0] *= 1.0 - probability
    return min(1.0, max(0.0, sum(distribution[observed_hits:])))


def calculate_group_random_probability(
    group_keys: Sequence[str], *, draw_count: int = 3
) -> float:
    """按组选号码覆盖的有序排列数计算单期随机命中概率。

    三位彩组六、组三、豹子分别覆盖 6、3、1 个有序开奖号。
    示例：``calculate_group_random_probability(["012", "001"])``。
    """

    unique_keys = {"".join(sorted(str(key))) for key in group_keys}
    covered = 0
    for key in unique_keys:
        if len(key) != draw_count or not key.isdigit():
            raise ValueError("组选号码必须由与开奖位数相同的数字组成")
        permutations = math.factorial(draw_count)
        for count in Counter(key).values():
            permutations //= math.factorial(count)
        covered += permutations
    return covered / (10**draw_count)


def _wilson_lower_bound(hits: int, periods: int, z_value: float) -> float:
    if periods <= 0:
        return 0.0
    rate = hits / periods
    z_squared = z_value**2
    denominator = 1.0 + z_squared / periods
    center = rate + z_squared / (2.0 * periods)
    margin = z_value * math.sqrt(
        rate * (1.0 - rate) / periods + z_squared / (4.0 * periods**2)
    )
    return max(0.0, (center - margin) / denominator)


def _build_blocks(
    hits: Sequence[bool], probabilities: Sequence[float], block_count: int
) -> list[ViabilityBlock]:
    periods = len(hits)
    if periods == 0:
        return []
    effective_blocks = min(block_count, periods)
    blocks: list[ViabilityBlock] = []
    for block_index in range(effective_blocks):
        start = block_index * periods // effective_blocks
        end = (block_index + 1) * periods // effective_blocks
        block_hits = sum(bool(value) for value in hits[start:end])
        block_probabilities = probabilities[start:end]
        expected_hits = sum(block_probabilities)
        block_periods = end - start
        random_probability = expected_hits / block_periods
        blocks.append(
            ViabilityBlock(
                index=block_index + 1,
                start_period=start + 1,
                end_period=end,
                periods=block_periods,
                hits=block_hits,
                actual_hit_rate=block_hits / block_periods,
                random_hit_probability=random_probability,
                expected_random_hits=expected_hits,
                meets_random_baseline=block_hits + 1e-12 >= expected_hits,
            )
        )
    return blocks


def evaluate_viability_metric(
    metric: str,
    hits: Sequence[bool],
    random_probabilities: Sequence[float],
    *,
    min_periods: int = DEFAULT_MIN_PERIODS,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    min_relative_lift: float = DEFAULT_MIN_RELATIVE_LIFT,
    block_count: int = DEFAULT_BLOCK_COUNT,
) -> ViabilityMetric:
    """根据精确随机基线评估一项命中指标。

    闸门同时要求样本量、单侧显著性、相对提升、99% Wilson 下界和
    非重叠时间块稳定性。示例：``evaluate_viability_metric("direct", hits, ps)``。
    """

    hit_values = [bool(value) for value in hits]
    probabilities = [float(value) for value in random_probabilities]
    if len(hit_values) != len(probabilities):
        raise ValueError("命中序列与随机概率序列长度必须一致")
    if min_periods <= 0 or block_count <= 0:
        raise ValueError("最小样本数与时间块数量必须为正整数")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("显著性水平必须在 0 到 1 之间")
    if min_relative_lift < 0:
        raise ValueError("最小相对提升不得为负数")
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("随机命中概率必须在 0 到 1 之间")

    periods = len(hit_values)
    observed_hits = sum(hit_values)
    expected_hits = sum(probabilities)
    actual_rate = observed_hits / periods if periods else 0.0
    random_probability = expected_hits / periods if periods else 0.0
    relative_lift = (
        actual_rate / random_probability - 1.0 if random_probability > 0 else None
    )
    p_value = poisson_binomial_right_tail(probabilities, observed_hits)
    wilson_lower = _wilson_lower_bound(observed_hits, periods, _WILSON_Z_99)
    blocks = _build_blocks(hit_values, probabilities, block_count)
    conditions = {
        "enoughPeriods": periods >= min_periods,
        "significant": p_value < significance_level,
        "relativeLift": (
            relative_lift is not None and relative_lift >= min_relative_lift
        ),
        "confidenceLowerBound": wilson_lower > random_probability,
        "stableAcrossBlocks": len(blocks) == block_count
        and all(block.meets_random_baseline for block in blocks),
    }
    viable = all(conditions.values())
    condition_names = {
        "enoughPeriods": f"样本少于{min_periods}期",
        "significant": f"单侧p值未小于{significance_level:.2f}",
        "relativeLift": f"相对随机提升未达到{min_relative_lift:.0%}",
        "confidenceLowerBound": "99% Wilson下界未高于随机基准",
        "stableAcrossBlocks": f"{block_count}个时间块未全部达到随机基准",
    }
    failures = [label for key, label in condition_names.items() if not conditions[key]]
    reason = "满足统计可行性闸门" if viable else "；".join(failures)
    return ViabilityMetric(
        metric=metric,
        periods=periods,
        hits=observed_hits,
        actual_hit_rate=actual_rate,
        random_hit_probability=random_probability,
        expected_random_hits=expected_hits,
        relative_lift=relative_lift,
        p_value=p_value,
        wilson_lower_bound_99=wilson_lower,
        blocks=blocks,
        conditions=conditions,
        viable=viable,
        reason=reason,
    )


def build_prediction_viability_report(
    direct_hits: Sequence[bool],
    direct_random_probabilities: Sequence[float],
    *,
    group_hits: Sequence[bool] | None = None,
    group_random_probabilities: Sequence[float] | None = None,
) -> PredictionViabilityReport:
    """构建一个策略的预测可行性报告。

    三位彩只有直选和组选均通过时才整体通过；排列五仅判断直选。
    """

    if (group_hits is None) != (group_random_probabilities is None):
        raise ValueError("组选命中与组选随机概率必须同时提供")
    direct_gate = evaluate_viability_metric(
        "direct", direct_hits, direct_random_probabilities
    )
    group_gate = (
        evaluate_viability_metric("group", group_hits, group_random_probabilities)
        if group_hits is not None and group_random_probabilities is not None
        else None
    )
    viable = direct_gate.viable and (group_gate is None or group_gate.viable)
    if viable:
        reason = "直选和组选均满足统计可行性闸门" if group_gate else direct_gate.reason
    else:
        failures = []
        if not direct_gate.viable:
            failures.append(f"直选：{direct_gate.reason}")
        if group_gate is not None and not group_gate.viable:
            failures.append(f"组选：{group_gate.reason}")
        reason = "未达到统计可行性门槛（" + "；".join(failures) + "）"
    return PredictionViabilityReport(
        viable=viable,
        direct_gate=direct_gate,
        group_gate=group_gate,
        reason=reason,
    )


__all__ = [
    "PredictionViabilityReport",
    "ViabilityBlock",
    "ViabilityMetric",
    "build_prediction_viability_report",
    "calculate_group_random_probability",
    "evaluate_viability_metric",
    "poisson_binomial_right_tail",
]
