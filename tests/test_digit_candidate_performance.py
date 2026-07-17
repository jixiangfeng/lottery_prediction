# -*- coding: utf-8 -*-
"""排列五完整候选空间的性能与等价性回归测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import digit_candidates
from src.analysis.digit_candidates import (
    DigitCandidateConfig,
    DigitExternalModelScores,
    generate_digit_candidates,
    rank_digit_numbers_with_eligible_count,
)
from src.analysis.digit_statistics import analyze_digit_history
from src.lotteries import get_lottery_rule


def _pl5_stats():
    rule = get_lottery_rule("pl5")
    history = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "万位": index % 10,
                "千位": (index * 3 + 1) % 10,
                "百位": (index * 7 + 2) % 10,
                "十位": (index + 4) % 10,
                "个位": (index + 8) % 10,
            }
            for index in range(80)
        ]
    )
    return rule, analyze_digit_history(history, rule)


def test_pl5_vectorized_result_keeps_legacy_text_order_and_unique_candidates():
    rule, stats = _pl5_stats()

    result = generate_digit_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=12, ranking_mode="ensemble"),
    )

    texts = [candidate.text for candidate in result.candidates]
    assert texts == [
        "27668",
        "41228",
        "81242",
        "43064",
        "56660",
        "56953",
        "06443",
        "63282",
        "24258",
        "14228",
        "43046",
        "46043",
    ]
    assert len(texts) == len(set(texts))
    assert result.candidates[0].score == pytest.approx(-13.18001)
    assert result.candidates[0].ensemble_score == pytest.approx(0.713992)
    assert result.candidates[0].composite_model_weight == pytest.approx(
        0.002960381094037504
    )


def test_pl5_scored_pool_normalizes_over_the_complete_filtered_space():
    rule, stats = _pl5_stats()
    config = digit_candidates._effective_config(
        rule, DigitCandidateConfig(count=12, ranking_mode="ensemble")
    )

    pool = digit_candidates._scored_candidate_pool(stats, rule, config)

    assert pool.composite_weights.sum() == pytest.approx(1.0)
    assert len(pool) == len(set(pool.universe_indexes.tolist()))


def test_pl5_generation_only_materializes_candidates_needed_for_selection(monkeypatch):
    rule, stats = _pl5_stats()
    original = digit_candidates.DigitCandidate
    materialized = 0

    def counting_candidate(*args, **kwargs):
        nonlocal materialized
        materialized += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(digit_candidates, "DigitCandidate", counting_candidate)

    result = generate_digit_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=12, ranking_mode="ensemble"),
    )

    assert len(result.candidates) == 12
    assert materialized < 500


def test_same_issue_reuses_ensemble_pool_for_generation_and_target_rank(monkeypatch):
    rule, stats = _pl5_stats()
    config = DigitCandidateConfig(count=12, ranking_mode="ensemble")
    external = DigitExternalModelScores({"27668": 0.8}, {"27668": 0.7})
    original = digit_candidates._build_score_context
    calls = 0

    def counting_context(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(digit_candidates, "_build_score_context", counting_context)

    generate_digit_candidates(stats, rule, config=config, external_scores=external)
    rank_digit_numbers_with_eligible_count(
        stats,
        rule,
        [2, 7, 6, 6, 8],
        config,
        external,
    )

    assert calls == 1


def test_external_model_score_caches_are_isolated_by_score_object():
    rule, stats = _pl5_stats()
    config = DigitCandidateConfig(count=1, ranking_mode="ensemble")
    first = DigitExternalModelScores({"27668": 1.0}, {})
    second = DigitExternalModelScores({"27668": 0.0}, {})

    first_result = generate_digit_candidates(
        stats, rule, config=config, external_scores=first
    )
    second_result = generate_digit_candidates(
        stats, rule, config=config, external_scores=second
    )

    assert (
        first_result.candidates[0].model_rank_percentiles
        != second_result.candidates[0].model_rank_percentiles
    )


def test_external_model_scores_take_read_only_defensive_copies():
    monte_carlo = {"27668": 0.8}
    ml_ranker = {"27668": 0.7}

    external = DigitExternalModelScores(monte_carlo, ml_ranker)
    monte_carlo["27668"] = 0.1
    ml_ranker["99999"] = 1.0

    assert external.monte_carlo.get("27668") == pytest.approx(0.8)
    assert list(external.ml_ranker.items()) == [("27668", 0.7)]
    assert external.monte_carlo
    with pytest.raises(TypeError):
        external.monte_carlo["27668"] = 0.2  # type: ignore[index]
    with pytest.raises(TypeError):
        external.ml_ranker["99999"] = 1.0  # type: ignore[index]


def test_cached_pool_cannot_become_stale_after_source_score_mutation():
    rule, stats = _pl5_stats()
    config = DigitCandidateConfig(count=1, ranking_mode="ensemble")
    source = {"27668": 1.0}
    external = DigitExternalModelScores(source, {})

    first = digit_candidates._scored_candidate_pool(
        stats,
        rule,
        digit_candidates._effective_config(rule, config),
        external,
    )
    first_percentiles = first.model_percentiles.copy()
    source["27668"] = 0.0
    second = digit_candidates._scored_candidate_pool(
        stats,
        rule,
        digit_candidates._effective_config(rule, config),
        external,
    )

    assert second is first
    assert second.model_percentiles == pytest.approx(first_percentiles)
    assert external.monte_carlo["27668"] == pytest.approx(1.0)


def test_scored_pool_cache_keeps_at_most_two_recent_external_score_pools():
    rule, stats = _pl5_stats()
    config = digit_candidates._effective_config(
        rule, DigitCandidateConfig(count=1, ranking_mode="ensemble")
    )
    digit_candidates._SCORED_POOL_CACHE.clear()

    for index in range(5):
        external = DigitExternalModelScores({"27668": index / 10}, {})
        digit_candidates._scored_candidate_pool(stats, rule, config, external)

    assert len(digit_candidates._SCORED_POOL_CACHE) <= 2
