# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import cast

from scripts.ssq_challenger_e_v1_history import build_parser
from src.analysis.ssq_8red1blue_v1_history import CONTROL_COUNT
from src.analysis.ssq_challenger_e_v1_history import (
    COMPOUND_METRICS,
    evaluate_full_history,
    walk_forward_history_fingerprints,
)
from src.analysis.ssq_history import SSQDraw


def _draws(count: int) -> list[SSQDraw]:
    return [
        SSQDraw(
            issue=str(2024001 + index),
            draw_date=f"2024-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
            red=tuple(sorted(((index + offset * 5) % 33) + 1 for offset in range(6))),
            blue=index % 16 + 1,
            source_url="fixture",
            raw_hash=f"{index:064x}",
            raw={},
        )
        for index in range(count)
    ]


def test_history_contract_scores_overlaps_and_hashes() -> None:
    report = evaluate_full_history(list(reversed(_draws(121))))
    history = cast(dict[str, object], report["history"])
    proper = cast(dict[str, dict[str, object]], report["properScores"])
    compounds = cast(dict[str, dict[str, object]], report["compoundSummary"])
    per_issue = cast(list[dict[str, object]], report["perIssue"])

    assert history["totalPeriods"] == 121
    assert history["warmupPeriods"] == 120
    assert history["evaluatedPeriods"] == 1
    assert proper["E"]["observations"] == 1
    assert proper["incumbent"]["observations"] == 1
    assert proper["uniform"]["observations"] == 1
    assert compounds["E"]["observations"] == 1
    assert compounds["D8"]["observations"] == 1
    assert compounds["C32"]["observations"] == CONTROL_COUNT
    assert report["retrospective"] is True
    assert report["formalGate"] is False
    assert report["selection"] is None
    assert report["formalRecommendationStatus"] == "uniform_abstain"
    assert report["historicalPromotion"] is False
    assert len(per_issue) == 1
    item = per_issue[0]
    assert cast(dict[str, object], item["overlaps"])["EWithBTickets"] == 0
    assert cast(dict[str, object], item["overlaps"])["D8WithBTickets"] == 0
    e_scores = cast(
        dict[str, float], cast(dict[str, object], item["properScores"])["E"]
    )
    assert abs(e_scores["sumRedProbabilities"] - 6.0) <= 1e-10
    assert set(
        cast(
            dict[str, float],
            cast(dict[str, object], report["compoundDifferences"])["EMinusD8Mean"],
        )
    ) == set(COMPOUND_METRICS)
    assert isinstance(report["reportSha256"], str)


def test_history_prediction_prefix_is_invariant_to_future_suffix() -> None:
    prefix = _draws(6)
    extended = [
        *prefix,
        SSQDraw(
            issue="2030001",
            draw_date="2030-01-01",
            red=(1, 7, 13, 19, 25, 31),
            blue=16,
            source_url="fixture",
            raw_hash="f" * 64,
            raw={},
        ),
    ]
    assert walk_forward_history_fingerprints(extended)[: len(prefix)] == (
        walk_forward_history_fingerprints(prefix)
    )


def test_history_cli_exposes_paths_only() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert option_strings == {"--csv", "--output"}
