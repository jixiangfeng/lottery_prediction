# -*- coding: utf-8 -*-
import numpy as np

from src.analysis import feature_enhancer as fe
from src.analysis.feature_enhancer import (
    FeatureDebugInfo,
    clear_graph_embedding_cache,
    compute_co_occurrence_scores,
    compute_enhanced_scores,
    compute_recency_and_momentum_scores,
)


def _build_sample_draws(rounds: int = 40) -> np.ndarray:
    draws = []
    for idx in range(rounds):
        issue = 2025000 - idx
        base = list(range(1, 21))
        rotated = base[idx % len(base):] + base[:idx % len(base)]
        draws.append([issue] + rotated)
    return np.asarray(draws, dtype=int)


def test_recency_and_momentum_scores_shape():
    draws = _build_sample_draws()
    recency, momentum = compute_recency_and_momentum_scores(draws, limit=30)
    assert recency.shape[0] == 81
    assert momentum.shape[0] == 81
    assert 0.0 <= recency.max() <= 1.0
    assert 0.0 <= momentum.max() <= 1.0


def test_co_occurrence_scores_normalised():
    draws = _build_sample_draws()
    scores = compute_co_occurrence_scores(draws, limit=20)
    assert scores.shape[0] == 81
    assert 0.0 <= scores.max() <= 1.0


def test_compute_enhanced_scores_returns_ranked_list():
    draws = _build_sample_draws()
    ranked, debug = compute_enhanced_scores(draws, limit=40, recent_window=25, reference_window=35)
    assert len(ranked) == 80
    assert isinstance(debug, FeatureDebugInfo)
    assert len(debug.dirichlet_scores) == 80
    sample_key = next(iter(debug.dirichlet_scores))
    assert 0.0 <= debug.dirichlet_scores[sample_key] <= 1.0
    assert len(debug.dirichlet_mean) == 80
    assert len(debug.dirichlet_variance) == 80
    assert len(debug.graph_embedding_scores) == 80
    # 检查排序单调性
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i + 1][1]


def test_compute_enhanced_scores_with_empty_input():
    draws = np.empty((0, 21), dtype=int)
    ranked, debug = compute_enhanced_scores(draws, limit=10)
    assert ranked == []
    assert isinstance(debug, FeatureDebugInfo)
    assert debug.combined_scores == []


def test_compute_enhanced_scores_with_graph_embeddings(tmp_path, monkeypatch):
    draws = _build_sample_draws()
    cache_path = tmp_path / "graph_embeddings.npz"
    rng = np.random.default_rng(2024)
    embeddings = rng.random((80, 16))
    np.savez(cache_path, embeddings=embeddings)

    original_path = fe.GRAPH_EMBED_CONFIG.get("cache_file")
    original_enabled = fe.GRAPH_EMBED_CONFIG.get("enabled", True)
    original_weight = fe.GRAPH_EMBED_CONFIG.get("weight", 0.0)
    monkeypatch.setitem(fe.GRAPH_EMBED_CONFIG, "cache_file", str(cache_path))
    monkeypatch.setitem(fe.GRAPH_EMBED_CONFIG, "enabled", True)
    monkeypatch.setitem(fe.GRAPH_EMBED_CONFIG, "weight", max(0.1, original_weight))
    clear_graph_embedding_cache()

    ranked, debug = compute_enhanced_scores(draws, limit=30)
    assert len(ranked) == 80
    assert isinstance(debug, FeatureDebugInfo)
    assert any(score > 0.0 for score in debug.graph_embedding_scores.values())

    # 恢复配置（由 monkeypatch 自动完成，但显式重置缓存）
    clear_graph_embedding_cache()
