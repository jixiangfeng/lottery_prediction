# -*- coding: utf-8 -*-
"""前500期开发集的滚动外层验证协议与跨彩种稳健性闸门。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RollingDevelopmentFold:
    """一个扩展选择区间和紧随其后的外层验证区间。"""

    number: int
    selection_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    development_end: int
    frozen_test_used: bool = False

    @property
    def all_target_indices(self) -> tuple[int, ...]:
        return self.selection_indices + self.validation_indices


def build_rolling_development_folds(
    *,
    development_end: int,
    min_train_size: int,
    initial_selection_end: int,
    validation_size: int,
    evaluation_stride: int = 1,
) -> tuple[RollingDevelopmentFold, ...]:
    """构造扩展窗口滚动折；目标索引永不达到 development_end。"""

    if min_train_size <= 0 or evaluation_stride <= 0 or validation_size <= 0:
        raise ValueError("训练期数、验证块和评估步长必须为正整数")
    if not min_train_size < initial_selection_end < development_end:
        raise ValueError(
            "必须满足 min_train_size < initial_selection_end < development_end"
        )
    if (development_end - initial_selection_end) % validation_size:
        raise ValueError("开发区间必须可被 validation_size 完整分块")

    folds = []
    selection_end = initial_selection_end
    number = 1
    while selection_end < development_end:
        validation_end = selection_end + validation_size
        if validation_end > development_end:
            raise ValueError("滚动验证折越过开发集终点")
        fold = RollingDevelopmentFold(
            number=number,
            selection_indices=tuple(
                range(min_train_size, selection_end, evaluation_stride)
            ),
            validation_indices=tuple(
                range(selection_end, validation_end, evaluation_stride)
            ),
            development_end=development_end,
        )
        if not fold.validation_indices:
            raise ValueError("每个滚动折至少需要一个验证目标期")
        if max(fold.all_target_indices) >= development_end:
            raise AssertionError("滚动开发协议不得触碰冻结测试段")
        folds.append(fold)
        selection_end = validation_end
        number += 1
    return tuple(folds)


def assess_joint_rolling_gate(
    metrics_by_lottery: Mapping[str, Sequence[Mapping[str, float | int]]],
    *,
    minimum_lift: float = 1.0,
) -> dict[str, object]:
    """按跨彩种最差滚动折判定；相等于随机基线也不算通过。"""

    if not metrics_by_lottery:
        raise ValueError("联合闸门至少需要一个彩种")
    records: list[tuple[float, str, int]] = []
    reasons = []
    for lottery in sorted(metrics_by_lottery):
        metrics = metrics_by_lottery[lottery]
        if not metrics:
            raise ValueError(f"{lottery} 缺少滚动折指标")
        for metric in metrics:
            fold = int(metric["fold"])
            lift = float(metric["lift"])
            records.append((lift, lottery, fold))
            if lift <= minimum_lift:
                reasons.append(
                    f"{lottery} 第{fold}折 lift={lift:.6f} "
                    f"未超过闸门 {minimum_lift:.6f}"
                )
    worst_lift, worst_lottery, worst_fold = min(records)
    qualified = not reasons
    return {
        "qualified": qualified,
        "worstLottery": worst_lottery,
        "worstFold": worst_fold,
        "worstFoldLift": worst_lift,
        "minimumLift": minimum_lift,
        "reasons": reasons,
        "frozenTestAllowed": qualified,
        "testSegmentUsedForSelection": False,
    }
