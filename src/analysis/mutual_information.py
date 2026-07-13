# -*- coding: utf-8 -*-
"""
互信息矩阵计算模块
----------------
基于历史开奖数据构建 0/1 指示矩阵，并计算号码两两之间的互信息，用于多样性惩罚或筛选。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

Number = int


def _extract_numbers(row: Sequence[int]) -> Sequence[int]:
    """去除期号，仅保留开奖号码。"""

    return [int(v) for v in row[1:] if int(v) > 0]


def _build_indicator_matrix(draws: ArrayLike, limit: int | None = None) -> np.ndarray:
    """构建开奖指示矩阵。"""

    matrix_input = np.asarray(draws, dtype=int)
    if matrix_input.ndim != 2 or matrix_input.shape[1] < 21:
        raise ValueError("开奖数据需为二维矩阵，且包含期号 + 20 个号码列。")
    effective = matrix_input if limit is None else matrix_input[:limit]
    indicators = np.zeros((effective.shape[0], 80), dtype=float)
    for idx, row in enumerate(effective):
        for number in _extract_numbers(row):
            indicators[idx, number - 1] = 1.0
    return indicators


def compute_mutual_information_matrix(draws: ArrayLike, limit: int | None = None, smoothing: float = 1e-6) -> np.ndarray:
    """计算 80x80 的互信息矩阵。"""

    indicators = _build_indicator_matrix(draws, limit)
    total = indicators.shape[0]
    if total == 0:
        return np.zeros((80, 80), dtype=float)

    counts = indicators.sum(axis=0)  # p(变量=1) 的计数
    joint_counts = indicators.T @ indicators  # 联合出现次数

    total_smooth = total + 4 * smoothing
    p1 = np.clip((counts + smoothing) / (total + 2 * smoothing), smoothing, 1.0)
    p0 = np.clip(1.0 - p1, smoothing, 1.0)

    p11 = np.clip((joint_counts + smoothing) / total_smooth, smoothing, 1.0)
    p10 = np.clip(p1[:, None] - p11, smoothing, 1.0)
    p01 = np.clip(p1[None, :] - p11, smoothing, 1.0)
    p00 = np.clip(1.0 - p11 - p10 - p01, smoothing, 1.0)

    mi = (
        p11 * np.log(p11 / np.clip(p1[:, None] * p1[None, :], smoothing, None))
        + p10 * np.log(p10 / np.clip(p1[:, None] * p0[None, :], smoothing, None))
        + p01 * np.log(p01 / np.clip(p0[:, None] * p1[None, :], smoothing, None))
        + p00 * np.log(p00 / np.clip(p0[:, None] * p0[None, :], smoothing, None))
    )

    np.fill_diagonal(mi, 0.0)
    mi = np.maximum(mi, 0.0)
    return mi


__all__ = ["compute_mutual_information_matrix"]
