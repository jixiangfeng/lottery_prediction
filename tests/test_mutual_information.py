# -*- coding: utf-8 -*-
import numpy as np

from src.analysis.mutual_information import compute_mutual_information_matrix


def _build_draws(rounds: int = 60) -> np.ndarray:
    draws = []
    base = list(range(1, 81))
    rng = np.random.default_rng(10)
    for idx in range(rounds):
        issue = 2024000 - idx
        picks = rng.choice(base, size=20, replace=False)
        draws.append([issue] + sorted(picks.tolist()))
    return np.asarray(draws, dtype=int)


def test_mutual_information_matrix_properties():
    draws = _build_draws()
    matrix = compute_mutual_information_matrix(draws, limit=50)
    assert matrix.shape == (80, 80)
    assert np.all(matrix >= 0.0)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_mutual_information_matrix_empty_input():
    draws = np.empty((0, 21), dtype=int)
    matrix = compute_mutual_information_matrix(draws, limit=None)
    assert matrix.shape == (80, 80)
    assert np.all(matrix == 0.0)
