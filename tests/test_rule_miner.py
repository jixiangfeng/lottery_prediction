# -*- coding: utf-8 -*-
import numpy as np

from src.analysis.rule_miner import build_rule_filter
from src.config import ensure_runtime_directories


def _build_sample_draws():
    tails = [
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [4, 6, 8, 10, 12, 14, 16, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        [5, 7, 9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 31, 33, 35, 37, 39],
        [6, 9, 12, 15, 18, 21, 24, 27, 31, 34, 37, 40, 43, 46, 49, 52, 55],
    ]
    draws = []
    for idx, tail in enumerate(tails, start=1):
        draws.append([2024000 + idx] + [1, 2, 3] + tail)
    draws.append([2024005] + [1, 2, 21] + tails[-1])
    return np.asarray(draws, dtype=int)


def test_rule_filter_hard_rejects_violation():
    ensure_runtime_directories()
    draws = _build_sample_draws()
    rule_filter = build_rule_filter(
        draws=draws,
        lottery_code="kl8",
        limit=None,
        mode="hard",
        min_support=0.6,
        min_confidence=0.9,
        max_itemset_size=2,
    )
    assert rule_filter is not None

    # 缺少 3 时应被拒绝
    evaluation = rule_filter.evaluate([1, 2, 4, 5, 6])
    assert evaluation.accepted is False
    # 完整包含规则后应通过
    ok_eval = rule_filter.evaluate([1, 2, 3, 4, 5])
    assert ok_eval.accepted is True


def test_rule_filter_soft_returns_penalty():
    ensure_runtime_directories()
    draws = _build_sample_draws()
    rule_filter = build_rule_filter(
        draws=draws,
        lottery_code="kl8",
        limit=None,
        mode="soft",
        min_support=0.6,
        min_confidence=0.9,
        max_itemset_size=2,
        penalty_weight=0.5,
    )
    assert rule_filter is not None

    evaluation = rule_filter.evaluate([1, 2, 4, 5, 6])
    assert evaluation.accepted is True
    assert evaluation.penalty > 0

    ok_eval = rule_filter.evaluate([1, 2, 3, 4, 5])
    assert ok_eval.penalty == 0.0
