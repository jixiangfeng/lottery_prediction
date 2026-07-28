# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import date, timedelta

from src.analysis.ssq_ensemble_v1 import (
    EVALUATION_BLOCK_SIZE,
    EVALUATION_WARMUP_DRAWS,
    RESEARCH_RED_AUDIT_COUNT,
    FixedEnsembleState,
    beam_red_combinations,
    blue_top1,
    evaluate_ssq_ensemble,
    generate_research_tickets,
    validate_research_candidates,
    walk_forward_prediction_fingerprints,
)
from src.analysis.ssq_history import SSQDraw, build_ssq_source_url


def _synthetic_draws(count: int) -> list[SSQDraw]:
    start = date(2020, 1, 1)
    draws: list[SSQDraw] = []
    for index in range(count):
        offset = (index * 7) % 33
        red = tuple(sorted(((offset + step * 5) % 33) + 1 for step in range(6)))
        if len(set(red)) != 6:
            raise AssertionError("合成红球必须唯一")
        draws.append(
            SSQDraw(
                issue=str(2020000 + index + 1),
                draw_date=(start + timedelta(days=index * 3)).isoformat(),
                red=red,
                blue=(index * 5) % 16 + 1,
                source_url=build_ssq_source_url(1),
                raw_hash="a" * 64,
                raw={},
            )
        )
    return draws


def test_walk_forward_predictions_do_not_use_future_draws():
    prefix = _synthetic_draws(80)
    extended = [*prefix, *_synthetic_draws(20)]
    for index, draw in enumerate(extended[80:], start=80):
        extended[index] = SSQDraw(
            issue=str(2030000 + index),
            draw_date=draw.draw_date,
            red=tuple(reversed(draw.red)),
            blue=16,
            source_url=draw.source_url,
            raw_hash=draw.raw_hash,
            raw={},
        )

    prefix_fingerprints = walk_forward_prediction_fingerprints(prefix)
    extended_fingerprints = walk_forward_prediction_fingerprints(extended)

    assert extended_fingerprints[: len(prefix)] == prefix_fingerprints


def test_incomplete_tail_is_excluded_and_no_formal_activation_occurs():
    count = EVALUATION_WARMUP_DRAWS + EVALUATION_BLOCK_SIZE + 37
    report = evaluate_ssq_ensemble(_synthetic_draws(count))

    assert len(report["blocks"]) == 1
    assert report["evaluatedDraws"] == EVALUATION_BLOCK_SIZE
    assert report["excludedIncompleteTailDraws"] == 37
    assert report["recommendationEnabled"] is False
    assert report["formalCandidates"] == []
    assert report["researchOnly"] is True


def test_insufficient_complete_blocks_force_uniform_abstain():
    report = evaluate_ssq_ensemble(
        _synthetic_draws(EVALUATION_WARMUP_DRAWS + EVALUATION_BLOCK_SIZE - 1)
    )

    assert report["hardGatePassed"] is False
    assert report["decision"] == "uniform_abstain"
    assert report["formalCandidates"] == []
    assert report["recommendationEnabled"] is False


def test_research_top20_is_valid_unique_and_deterministic():
    state = FixedEnsembleState()
    red, blue, pairs = state.predict()

    first = generate_research_tickets(red, blue, pairs)
    second = generate_research_tickets(red, blue, pairs)

    assert first == second
    validate_research_candidates(first)
    keys = [(tuple(ticket["red"]), ticket["blue"]) for ticket in first]
    assert len(keys) == len(set(keys)) == 20
    assert all(ticket["predictionClaim"] is False for ticket in first)
    assert keys == sorted(keys)


def test_report_exposes_fixed_current_predraw_ranking_audit():
    report = evaluate_ssq_ensemble([])
    state = FixedEnsembleState()
    red, blue, pairs = state.predict()
    expected_red = beam_red_combinations(red, pairs)[:RESEARCH_RED_AUDIT_COUNT]
    audit = report["auditMetadata"]

    assert audit["orderedRed6CombinationCount"] == RESEARCH_RED_AUDIT_COUNT
    assert audit["researchBlueTop1"] == blue_top1(blue)
    assert audit["orderedRed6Combinations"] == [
        {
            "rank": rank,
            "red": [f"{ball:02d}" for ball in combination],
            "redScore": score,
        }
        for rank, (combination, score) in enumerate(expected_red, start=1)
    ]
    assert audit["finalNextProbabilities"] == {"red": red, "blue": blue}
    diversified = report["diversifiedPortfolioV2"]
    assert diversified["audit"]["redUnionCount"] == 33
    assert diversified["audit"]["maximumRedExposure"] == 2
    assert diversified["audit"]["distinctBlues"] == 5
