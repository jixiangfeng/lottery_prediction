# -*- coding: utf-8 -*-
"""
Copula 采样辅助器
------------------
基于历史开奖的二元指示矩阵估计高斯 Copula 相关结构，再结合边缘概率生成多样化的候选组合。
实现目标：
1. 兼容 CPU/GPU 独立调整（核心计算使用 NumPy/SciPy，Python311 环境即可）；
2. 支持通过配置/命令行控制采样数量、相关矩阵收缩强度等参数；
3. 提供调试信息，便于在文档记录中追踪采样过程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

Number = int


def _extract_numbers(row: Sequence[int]) -> Iterable[int]:
    """去除期号，仅返回号码列。"""

    for value in row[1:]:
        number = int(value)
        if 1 <= number <= 80:
            yield number


def _build_indicator_matrix(draws: ArrayLike, limit: int | None = None) -> np.ndarray:
    """将历史开奖转换为 0/1 指示矩阵。"""

    array = np.asarray(draws, dtype=int)
    if array.ndim != 2 or array.shape[1] < 21:
        raise ValueError("开奖数据需为二维矩阵且至少包含期号 + 20 个号码。")

    effective = array if limit is None else array[:limit]
    rows = effective.shape[0]
    indicators = np.zeros((rows, 80), dtype=float)

    for idx in range(rows):
        for number in _extract_numbers(effective[idx]):
            indicators[idx, number - 1] = 1.0
    return indicators


@dataclass(frozen=True)
class CopulaSamplerConfig:
    """Copula 采样配置。"""

    min_draws: int = 180
    shrinkage: float = 0.12
    samples: int = 48
    topk_multiplier: float = 1.3
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.min_draws <= 0:
            raise ValueError("min_draws 必须为正整数。")
        if not (0.0 <= self.shrinkage < 1.0):
            raise ValueError("shrinkage 需位于 [0,1) 区间。")
        if self.samples <= 0:
            raise ValueError("samples 必须大于 0。")
        if self.topk_multiplier < 1.0:
            raise ValueError("topk_multiplier 至少为 1.0。")


@dataclass
class CopulaSamplerDiagnostics:
    """便于写入报告的调试信息。"""

    effective_draws: int
    shrinkage: float
    marginal_min: float
    marginal_max: float
    condition_number: float


class CopulaSampler:
    """高斯 Copula 采样器。"""

    def __init__(self, config: CopulaSamplerConfig):
        self._config = config
        self._marginals: np.ndarray | None = None
        self._cholesky: np.ndarray | None = None
        self._rng = np.random.default_rng(config.random_seed)
        self.diagnostics: CopulaSamplerDiagnostics | None = None

    @property
    def is_fitted(self) -> bool:
        return self._marginals is not None and self._cholesky is not None

    def fit(self, draws: ArrayLike, limit: int | None = None) -> None:
        """根据历史开奖拟合相关结构。"""

        indicators = _build_indicator_matrix(draws, limit)
        sample_count = indicators.shape[0]
        if sample_count < self._config.min_draws:
            raise ValueError(
                f"历史样本量不足（{sample_count} < {self._config.min_draws}），无法稳定估计 Copula。"
            )

        marginals = indicators.mean(axis=0)
        marginals = np.clip(marginals, 1e-4, 1.0 - 1e-4)

        centered = indicators - marginals
        covariance = np.dot(centered.T, centered) / max(sample_count - 1, 1)
        variances = np.clip(np.diag(covariance), 1e-6, None)
        std = np.sqrt(variances)
        correlation = covariance / np.outer(std, std)
        np.fill_diagonal(correlation, 1.0)

        if self._config.shrinkage > 0.0:
            shrink = np.clip(self._config.shrinkage, 0.0, 1.0)
            correlation = (1.0 - shrink) * correlation + shrink * np.eye(80, dtype=float)

        # 对称化并截断特征值，确保半正定
        eigenvalues, eigenvectors = np.linalg.eigh((correlation + correlation.T) / 2.0)
        eigenvalues = np.clip(eigenvalues, 1e-6, None)
        correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
        correlation = (correlation + correlation.T) / 2.0

        try:
            cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:  # 极端情况下再做微调
            jitter = 1e-4
            cholesky = np.linalg.cholesky(correlation + jitter * np.eye(80, dtype=float))

        self._marginals = marginals
        self._cholesky = cholesky
        condition = float(np.linalg.cond(correlation))
        self.diagnostics = CopulaSamplerDiagnostics(
            effective_draws=sample_count,
            shrinkage=float(self._config.shrinkage),
            marginal_min=float(marginals.min()),
            marginal_max=float(marginals.max()),
            condition_number=condition,
        )

    def sample(self, n_samples: int | None = None, topk: int = 20) -> List[List[Number]]:
        """采样候选组合。"""

        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 拟合 Copula 结构。")
        n = n_samples if n_samples is not None else self._config.samples
        n = max(n, 1)
        cholesky = self._cholesky
        marginals = self._marginals
        assert cholesky is not None and marginals is not None

        gaussian = self._rng.standard_normal((n, 80))
        correlated = gaussian @ cholesky.T

        log_bias = np.log(marginals)
        candidates: List[List[Number]] = []

        for idx, latent in enumerate(correlated):
            combined = latent + log_bias
            indices = np.argpartition(combined, -topk)[-topk:]
            sorted_indices = np.sort(indices)
            combo = [int(idx) + 1 for idx in sorted_indices]
            if len(combo) == topk:
                candidates.append(combo)
        return candidates


def generate_copula_candidates(
    draws: ArrayLike,
    limit: int,
    desired: int,
    config: CopulaSamplerConfig,
) -> Tuple[List[List[Number]], CopulaSamplerDiagnostics | None]:
    """便捷函数：拟合后直接返回候选组合及诊断信息。"""

    sampler = CopulaSampler(config)
    sampler.fit(draws, limit)
    multiplier = max(1.0, config.topk_multiplier)
    n_samples = int(np.ceil(desired * multiplier))
    combos = sampler.sample(n_samples, topk=20)
    return combos, sampler.diagnostics
