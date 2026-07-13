# -*- coding: utf-8 -*-
"""
特征增强工具箱
--------------
面向 `kl8_analysis*.py` 脚本提供基于历史开奖的高级特征计算：
1. 近期频率与动量；
2. 号码共现谱分析；
3. Dirichlet-Multinomial 分层平滑；
4. 图嵌入特征（由 Node2Vec 随脚本训练缓存）；
5. PCA 主成分辅助特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover - optional依赖
    PCA = None

try:
    from ..config import DIRICHLET_CONFIG, GRAPH_EMBED_CONFIG
except Exception:  # pragma: no cover - 脚本直跑时的路径回退
    from pathlib import Path as _Path
    import sys

    PROJECT_ROOT = _Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import DIRICHLET_CONFIG, GRAPH_EMBED_CONFIG  # type: ignore


Number = int
Score = float

_GRAPH_EMBED_CACHE: Optional[np.ndarray] = None
_GRAPH_EMBED_SOURCE: Optional[Path] = None


@dataclass(frozen=True)
class FeatureDebugInfo:
    """用于日志输出的结构化特征信息。"""

    recency_scores: Dict[Number, Score]
    momentum_scores: Dict[Number, Score]
    co_occurrence_scores: Dict[Number, Score]
    dirichlet_mean: Dict[Number, Score]
    dirichlet_variance: Dict[Number, Score]
    dirichlet_scores: Dict[Number, Score]
    graph_embedding_scores: Dict[Number, Score]
    combined_scores: List[Tuple[Number, Score]]


def clear_graph_embedding_cache() -> None:
    """测试辅助函数：主动清理图嵌入缓存。"""

    global _GRAPH_EMBED_CACHE, _GRAPH_EMBED_SOURCE
    _GRAPH_EMBED_CACHE = None
    _GRAPH_EMBED_SOURCE = None


def _iter_recent_draws(draws: np.ndarray, limit: int) -> Iterable[np.ndarray]:
    """返回按时间倒序截断后的开奖数据。"""

    if limit <= 0 or limit > len(draws):
        limit = len(draws)
    for row in draws[:limit]:
        yield row


def _normalise(values: np.ndarray) -> np.ndarray:
    """将数组缩放到 0-1 区间，避免除零异常。"""

    vmax = values.max()
    vmin = values.min()
    if vmax - vmin < 1e-9:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _extract_numbers(draw_row: Sequence[Number]) -> List[Number]:
    """去除行首的期号，仅返回 20 个开奖号码。"""

    return [int(n) for n in draw_row[1:] if int(n) > 0]


def _numbers_matrix(draws: np.ndarray, limit: int | None = None) -> np.ndarray:
    """将开奖记录转换为 0/1 指示矩阵。"""

    effective = draws if limit is None else draws[:limit]
    matrix = np.zeros((effective.shape[0], 80), dtype=float)
    for idx, row in enumerate(effective):
        for number in _extract_numbers(row):
            matrix[idx, number - 1] = 1.0
    return matrix


def _load_graph_embeddings() -> Optional[np.ndarray]:
    """加载预训练的号码图嵌入向量。"""

    global _GRAPH_EMBED_CACHE, _GRAPH_EMBED_SOURCE
    if not GRAPH_EMBED_CONFIG.get("enabled", True):
        return None

    cache_path = Path(GRAPH_EMBED_CONFIG.get("cache_file", "")).expanduser()
    if cache_path == _GRAPH_EMBED_SOURCE and _GRAPH_EMBED_CACHE is not None:
        return _GRAPH_EMBED_CACHE

    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=True)
        embeddings = payload.get("embeddings")
        if embeddings is None:
            return None
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.shape[0] != 80:
            return None
        _GRAPH_EMBED_CACHE = embeddings
        _GRAPH_EMBED_SOURCE = cache_path
        return embeddings
    except Exception:  # pragma: no cover - 容忍损坏缓存
        return None


def _compute_graph_embedding_scores(draws: np.ndarray, limit: int) -> np.ndarray:
    """基于图嵌入向量生成号码得分。"""

    embeddings = _load_graph_embeddings()
    scores = np.zeros(81, dtype=float)
    if embeddings is None:
        return scores

    metric = str(GRAPH_EMBED_CONFIG.get("metric", "norm")).lower()
    if metric == "cosine_mean":
        reference = embeddings.mean(axis=0)
        reference_norm = np.linalg.norm(reference) + 1e-9
        norms = np.linalg.norm(embeddings, axis=1)
        cosine = np.zeros(embeddings.shape[0], dtype=float)
        valid = norms > 1e-9
        cosine[valid] = (embeddings[valid] @ reference) / (norms[valid] * reference_norm)
        values = cosine
    else:
        values = np.linalg.norm(embeddings, axis=1)

    scores[1:] = _normalise(values)
    return scores


def compute_recency_and_momentum_scores(
    draws: np.ndarray,
    limit: int,
    recent_window: int = 40,
    reference_window: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算近期频率与动量得分。

    - ``recent_window``：近期窗口长度；
    - ``reference_window``：长期参考窗口长度。
    """

    max_number = 80
    recency_counts = np.zeros(max_number + 1, dtype=float)
    reference_counts = np.zeros(max_number + 1, dtype=float)

    limited_draws = list(_iter_recent_draws(draws, limit))
    recent_slice = limited_draws[:recent_window]
    reference_slice = limited_draws[:reference_window]

    for idx, row in enumerate(recent_slice):
        weight = 1.0 - (idx / max(len(recent_slice), 1))
        for value in _extract_numbers(row):
            recency_counts[value] += max(weight, 0.1)

    for row in reference_slice:
        for value in _extract_numbers(row):
            reference_counts[value] += 1.0

    recency_scores = _normalise(recency_counts)
    momentum_raw = recency_counts - reference_counts * (
        len(recent_slice) / max(len(reference_slice), 1)
    )
    momentum_scores = _normalise(momentum_raw)

    return recency_scores, momentum_scores


def compute_co_occurrence_scores(
    draws: np.ndarray,
    limit: int,
    decay: float = 0.97,
) -> np.ndarray:
    """构建共现矩阵并返回谱中心性得分。"""

    max_number = 80
    matrix = np.zeros((max_number + 1, max_number + 1), dtype=float)

    for idx, row in enumerate(_iter_recent_draws(draws, limit)):
        numbers = _extract_numbers(row)
        if len(numbers) < 2:
            continue
        weight = decay**idx
        for i in range(len(numbers)):
            ni = numbers[i]
            for j in range(i + 1, len(numbers)):
                nj = numbers[j]
                matrix[ni, nj] += weight
                matrix[nj, ni] += weight

    try:
        eig_values, eig_vectors = np.linalg.eig(matrix[1:, 1:])
        principal = np.abs(eig_vectors[:, np.argmax(eig_values.real)])
        scores = np.zeros(81, dtype=float)
        scores[1:] = _normalise(principal.real)
    except Exception:  # pragma: no cover - 防御性回退
        scores = np.zeros(81, dtype=float)

    return scores


def _compute_dirichlet_scores(
    draws: np.ndarray,
    limit: int,
    window_size: int,
    prior_strength: float,
    variance_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于 Dirichlet-Multinomial 后验计算得分、均值与方差。"""

    max_number = 80
    effective_limit = min(limit, draws.shape[0])
    local_counts = np.zeros(max_number + 1, dtype=float)
    global_counts = np.zeros(max_number + 1, dtype=float)

    for idx, row in enumerate(_iter_recent_draws(draws, effective_limit)):
        numbers = _extract_numbers(row)
        for value in numbers:
            global_counts[value] += 1.0
            if idx < window_size:
                local_counts[value] += 1.0

    global_total = global_counts.sum()
    if global_total <= 0:
        base_prior = np.ones(max_number + 1, dtype=float)
    else:
        freq = global_counts / global_total
        base_prior = np.maximum(freq * prior_strength * max_number, 1e-6)

    posterior_alpha = base_prior + local_counts
    alpha_sum = posterior_alpha.sum()
    if alpha_sum <= 0:
        zero = np.zeros(max_number + 1, dtype=float)
        return zero, zero, zero

    posterior_mean = posterior_alpha / alpha_sum
    variance = (posterior_alpha * (alpha_sum - posterior_alpha)) / (alpha_sum**2 * (alpha_sum + 1.0))
    adjusted = posterior_mean - variance_weight * np.sqrt(np.maximum(variance, 0.0))
    dirichlet_scores = _normalise(adjusted)

    return dirichlet_scores, posterior_mean, variance


def compute_enhanced_scores(
    draws: np.ndarray,
    limit: int,
    recent_window: int = 40,
    reference_window: int = 160,
    decay: float = 0.97,
    weights: Tuple[float, float, float] = (0.45, 0.25, 0.30),
    dirichlet_weight: float | None = None,
    pca_components: int = 1,
    use_pca: bool = True,
) -> Tuple[List[Tuple[Number, Score]], FeatureDebugInfo]:
    """汇总多源特征，返回排序结果与调试信息。"""

    if draws.size == 0:
        empty = FeatureDebugInfo({}, {}, {}, {}, {}, {}, {}, [])
        return [], empty

    recency_scores, momentum_scores = compute_recency_and_momentum_scores(
        draws,
        limit=limit,
        recent_window=recent_window,
        reference_window=reference_window,
    )
    co_occurrence_scores = compute_co_occurrence_scores(
        draws,
        limit=limit,
        decay=decay,
    )

    dirichlet_scores, dirichlet_mean, dirichlet_variance = _compute_dirichlet_scores(
        draws=draws,
        limit=limit,
        window_size=max(1, min(DIRICHLET_CONFIG["window_size"], limit)),
        prior_strength=max(DIRICHLET_CONFIG["prior_strength"], 1e-6),
        variance_weight=max(DIRICHLET_CONFIG["variance_weight"], 0.0),
    )

    graph_scores = _compute_graph_embedding_scores(draws, limit)

    pca_scores = np.zeros(81, dtype=float)
    if use_pca and PCA is not None:
        numbers_matrix = _numbers_matrix(draws, limit=min(limit, draws.shape[0]))
        try:
            pca = PCA(n_components=pca_components)
            pca.fit(numbers_matrix)
            if pca_components == 1:
                pc1 = pca.components_[0]
                pc1_norm = _normalise(pc1)
                for n in range(1, 81):
                    pca_scores[n] = float(pc1_norm[n - 1])
            else:
                pc_sum = np.sum(np.abs(pca.components_), axis=0)
                pc_sum_norm = _normalise(pc_sum)
                for n in range(1, 81):
                    pca_scores[n] = float(pc_sum_norm[n - 1])
        except Exception:  # pragma: no cover - 防御 fallback
            pca_scores = np.zeros(81, dtype=float)

    w_recency, w_momentum, w_co = weights
    w_pca = 0.18 if use_pca and PCA is not None else 0.0
    w_dirichlet = dirichlet_weight if dirichlet_weight is not None else 0.22
    w_graph = max(float(GRAPH_EMBED_CONFIG.get("weight", 0.0)), 0.0)

    combined = (
        w_recency * recency_scores
        + w_momentum * momentum_scores
        + w_co * co_occurrence_scores
        + w_dirichlet * dirichlet_scores
        + w_pca * pca_scores
        + w_graph * graph_scores
    )

    ranked = sorted(
        ((number, float(combined[number])) for number in range(1, 81)),
        key=lambda item: item[1],
        reverse=True,
    )

    debug_info = FeatureDebugInfo(
        recency_scores={i: float(recency_scores[i]) for i in range(1, 81)},
        momentum_scores={i: float(momentum_scores[i]) for i in range(1, 81)},
        co_occurrence_scores={i: float(co_occurrence_scores[i]) for i in range(1, 81)},
        dirichlet_mean={i: float(dirichlet_mean[i]) for i in range(1, 81)},
        dirichlet_variance={i: float(dirichlet_variance[i]) for i in range(1, 81)},
        dirichlet_scores={i: float(dirichlet_scores[i]) for i in range(1, 81)},
        graph_embedding_scores={i: float(graph_scores[i]) for i in range(1, 81)},
        combined_scores=ranked,
    )

    return ranked, debug_info
