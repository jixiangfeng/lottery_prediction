# -*- coding: utf-8 -*-
import numpy as np
import pytest

from src.analysis.copula_sampler import (
    CopulaSampler,
    CopulaSamplerConfig,
    generate_copula_candidates,
)


def _build_sample_draws(rounds: int = 200) -> np.ndarray:
    draws = []
    base_numbers = list(range(1, 81))
    for idx in range(rounds):
        issue = 2025000 - idx
        rng = np.random.default_rng(idx + 123)
        picks = rng.choice(base_numbers, size=20, replace=False)
        draws.append([issue] + sorted(picks.tolist()))
    return np.asarray(draws, dtype=int)


def test_copula_sampler_produces_combinations():
    draws = _build_sample_draws()
    config = CopulaSamplerConfig(min_draws=50, shrinkage=0.15, samples=10, topk_multiplier=1.1, random_seed=7)
    sampler = CopulaSampler(config)
    sampler.fit(draws, limit=120)
    combos = sampler.sample(6)
    assert len(combos) == 6
    for combo in combos:
        assert len(combo) == 20
        assert len(set(combo)) == 20
        assert all(1 <= num <= 80 for num in combo)

    # 确认随机种子保证确定性
    sampler_again = CopulaSampler(config)
    sampler_again.fit(draws, limit=120)
    combos_again = sampler_again.sample(6)
    assert combos == combos_again


def test_generate_copula_candidates_respects_desired_count():
    draws = _build_sample_draws(180)
    config = CopulaSamplerConfig(min_draws=60, samples=8, shrinkage=0.1, topk_multiplier=1.2, random_seed=42)
    combos, diagnostics = generate_copula_candidates(draws, limit=150, desired=5, config=config)
    assert len(combos) >= 5
    assert diagnostics is not None
    assert diagnostics.effective_draws >= 60


def test_copula_sampler_raises_on_small_sample():
    draws = _build_sample_draws(20)
    config = CopulaSamplerConfig(min_draws=50, samples=5)
    sampler = CopulaSampler(config)
    with pytest.raises(ValueError):
        sampler.fit(draws, limit=20)
