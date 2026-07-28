# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest

from scripts.ssq_challenger_e2_selection_v1 import build_parser as selection_parser
from scripts.ssq_challenger_e2_v1 import build_parser as current_parser
from src.analysis.ssq_challenger_e2_selection_v1 import (
    DIAGNOSTIC_DRAWS,
    SEARCH_DRAWS,
    VALIDATION_DRAWS,
    WARMUP_DRAWS,
    assess_eligibility,
    build_selection_report,
    fixed_split_boundaries,
    select_candidate,
    validate_selection_report,
)
from src.analysis.ssq_challenger_e2_v1 import (
    CANDIDATE_SPECS,
    FEATURE_NAMES,
    ChallengerE2State,
    build_current_report,
    protocol_sha256,
    walk_forward_prediction_fingerprints,
)
from src.analysis.ssq_diversified_portfolio_v2 import build_diversified_portfolio_v2
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
    validate_small_compound_8red1blue_v1,
)

ENSEMBLE_PATH = Path("reports/retrospective/ssq_ensemble_v1_through_2026085.json")
ENSEMBLE_FILE_SHA256 = (
    "8fac0f001c1ccff59ece8776f932f22d738ef7bad881f367f613821828d0f6a1"
)


def _draw(index: int, issue: str | None = None) -> SSQDraw:
    return SSQDraw(
        issue=issue or str(2024001 + index),
        draw_date=f"2024-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
        red=tuple(sorted(((index + offset * 5) % 33) + 1 for offset in range(6))),
        blue=index % 16 + 1,
        source_url="fixture",
        raw_hash=f"{index:064x}",
        raw={},
    )


def _metric(
    candidate_id: str,
    *,
    log_loss: float = 0.473,
    brier: float = 0.148,
    mean_delta: float = 0.02,
    blocks: tuple[float, ...] = (0.01, 0.02, 0.03, 0.0, 0.04),
) -> dict[str, object]:
    return {
        "candidateId": candidate_id,
        "properScores": {
            "candidate": {
                "redLogLossPerBall": log_loss,
                "redBrierPerBall": brier,
            },
            "uniform": {
                "redLogLossPerBall": 0.474,
                "redBrierPerBall": 0.149,
            },
        },
        "paired": {
            "meanRed8OverlapCandidateMinusD8": mean_delta,
            "red5RateCandidateMinusD8": 0.0,
            "red6CountCandidateMinusD8": 0,
            "blocks": [
                {"blockIndex": index, "meanRed8OverlapCandidateMinusD8": value}
                for index, value in enumerate(blocks)
            ],
        },
    }


def test_protocol_has_exactly_eight_fixed_candidates_and_no_overrides() -> None:
    assert [spec.candidate_id for spec in CANDIDATE_SPECS] == [
        "F0_L001",
        "F0_L010",
        "F1_L001",
        "F1_L010",
        "F2_L001",
        "F2_L010",
        "F3_L001",
        "F3_L010",
    ]
    assert len({spec.candidate_id for spec in CANDIDATE_SPECS}) == 8
    assert {(spec.l2) for spec in CANDIDATE_SPECS} == {0.01, 0.10}
    assert len(protocol_sha256()) == 64

    for parser in (current_parser(), selection_parser()):
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option not in {"-h", "--help"}
        }
        assert not any(
            token in option
            for option in options
            for token in ("feature", "l2", "retry", "candidate", "warmup")
        )


def test_feature_masks_and_trend_equation_are_frozen() -> None:
    expected = {
        "F0": FEATURE_NAMES[:2],
        "F1": FEATURE_NAMES[:4],
        "F2": FEATURE_NAMES[:5],
        "F3": FEATURE_NAMES,
    }
    for spec in CANDIDATE_SPECS:
        assert spec.feature_names == expected[spec.feature_set]
        assert len(ChallengerE2State(spec).beta) == len(spec.feature_names)

    state = ChallengerE2State(CANDIDATE_SPECS[-1])
    state.predict()
    state.score_then_update(
        SSQDraw(
            issue="2024001",
            draw_date="2024-01-01",
            red=(1, 2, 3, 4, 5, 6),
            blue=1,
            source_url="fixture",
            raw_hash="0" * 64,
            raw={},
        )
    )
    locked = state.predict()
    assert all(row[5] == pytest.approx(row[0] - row[1]) for row in locked.features)
    expected_gap = math.log1p(1) / math.log(61.0) - 0.5
    assert locked.features[0][2] == pytest.approx(expected_gap)


def test_sum_six_lock_serialization_and_no_leakage_prefix() -> None:
    state = ChallengerE2State(CANDIDATE_SPECS[0])
    locked = state.predict()
    assert sum(locked.probabilities) == pytest.approx(6.0, abs=1e-12)
    with pytest.raises(RuntimeError, match="先更新"):
        state.predict()

    payload = state.state_payload()
    assert ChallengerE2State.from_dict(payload).state_payload() == payload
    tampered = json.loads(json.dumps(payload))
    tampered["pendingPrediction"]["probabilities"][0] += 0.01
    with pytest.raises(ValueError, match="预测锁"):
        ChallengerE2State.from_dict(tampered)

    prefix = [_draw(index) for index in range(7)]
    extended = [*prefix, _draw(20, "2030001")]
    assert walk_forward_prediction_fingerprints(extended, CANDIDATE_SPECS[3])[:7] == (
        walk_forward_prediction_fingerprints(prefix, CANDIDATE_SPECS[3])
    )


def test_fixed_splits_are_exact_120_923_500_500() -> None:
    draws = [_draw(index) for index in range(2043)]
    splits = fixed_split_boundaries(draws)
    assert (WARMUP_DRAWS, SEARCH_DRAWS, VALIDATION_DRAWS, DIAGNOSTIC_DRAWS) == (
        120,
        923,
        500,
        500,
    )
    assert splits["warmup"]["startIndex"] == 0
    assert splits["warmup"]["endIndexExclusive"] == 120
    assert splits["search"]["startIndex"] == 120
    assert splits["search"]["endIndexExclusive"] == 1043
    assert splits["validation"]["startIndex"] == 1043
    assert splits["validation"]["endIndexExclusive"] == 1543
    assert splits["diagnostic"]["startIndex"] == 1543
    assert splits["diagnostic"]["endIndexExclusive"] == 2043
    with pytest.raises(ValueError, match="恰好2043"):
        fixed_split_boundaries(draws[:-1])


def test_eligibility_block_gates_and_secondary_never_select() -> None:
    eligible = assess_eligibility(_metric("F0_L001"))
    assert eligible["eligible"] is True
    assert eligible["checks"] == {
        "logLossStrictlyBetterThanUniform": True,
        "brierNoWorseThanUniform": True,
        "meanRed8OverlapCandidateMinusD8Positive": True,
        "atLeastFourNonnegativeBlocks": True,
        "noBlockBelowMinus0_05": True,
    }
    assert (
        assess_eligibility(_metric("F0_L001", blocks=(0.1, -0.01, -0.02, 0.03, 0.04)))[
            "eligible"
        ]
        is False
    )
    assert (
        assess_eligibility(_metric("F0_L001", blocks=(0.1, 0.1, 0.1, 0.1, -0.051)))[
            "eligible"
        ]
        is False
    )


def test_selection_is_deterministic_lexicographic_and_can_reject() -> None:
    metrics = [_metric(spec.candidate_id, log_loss=0.472) for spec in CANDIDATE_SPECS]
    metrics[2] = _metric("F1_L001", log_loss=0.470, brier=0.147, mean_delta=0.03)
    metrics[3] = _metric("F1_L010", log_loss=0.470, brier=0.147, mean_delta=0.03)
    selected = select_candidate(metrics)
    assert selected == {"selectionStatus": "selected", "selectedCandidateId": "F1_L001"}

    rejected = [
        _metric(spec.candidate_id, log_loss=0.475, mean_delta=-0.01)
        for spec in CANDIDATE_SPECS
    ]
    assert select_candidate(rejected) == {
        "selectionStatus": "rejected",
        "selectedCandidateId": None,
    }


def test_builder_is_legal_28_and_zero_overlap_with_b() -> None:
    red = [6.0 / 33.0] * 33
    blue = [1.0 / 16.0] * 16
    b_document = build_diversified_portfolio_v2(red, blue)
    document = build_small_compound_8red1blue_v1(red, blue, b_document)
    validate_small_compound_8red1blue_v1(
        document,
        red_probabilities=red,
        blue_probabilities=blue,
        diversified_portfolio=b_document,
    )
    assert len(cast(list[object], document["expandedTickets"])) == 28
    assert cast(dict[str, object], document["audit"])["overlapWithB"] == 0


def test_rejected_current_report_is_null_and_ensemble_bytes_unchanged(
    tmp_path: Path,
) -> None:
    before = ENSEMBLE_PATH.read_bytes()
    assert hashlib.sha256(before).hexdigest() == ENSEMBLE_FILE_SHA256
    selection = {
        "selectionStatus": "rejected",
        "selectedCandidateId": None,
        "protocolSha256": protocol_sha256(),
        "reportSha256": "fixture",
    }
    report = build_current_report(
        csv_path=tmp_path / "unused.csv",
        ensemble_report_path=ENSEMBLE_PATH,
        selection_report=selection,
        draws=None,
    )
    assert report["currentTargetGroup"] is None
    assert report["selectionStatus"] == "rejected"
    assert report["formalGate"] is False
    assert report["formalRecommendationStatus"] == "uniform_abstain"
    assert ENSEMBLE_PATH.read_bytes() == before


def test_selection_output_schema_requires_all_eight_and_selected_diagnostic_only() -> (
    None
):
    report: dict[str, object] = {
        "schemaVersion": "ssq_challenger_e2_selection_v1",
        "researchOnly": True,
        "retrospective": True,
        "formalGate": False,
        "formalRecommendationStatus": "uniform_abstain",
        "automaticPromotion": False,
        "selectionStatus": "selected",
        "selectedCandidateId": "F0_L001",
        "validation": {
            "candidates": [_metric(spec.candidate_id) for spec in CANDIDATE_SPECS]
        },
        "diagnostic": {"candidateId": "F0_L001", "promotionEvidence": False},
        "splits": {"warmup": {}, "search": {}, "validation": {}, "diagnostic": {}},
        "protocolSha256": protocol_sha256(),
        "dataSha256": "0" * 64,
        "inputSha256": "1" * 64,
        "auditFingerprints": {},
        "reportSha256": "2" * 64,
    }
    validate_selection_report(report)
    report["validation"] = {
        "candidates": cast(list[object], report["validation"]["candidates"])[:-1]
    }
    with pytest.raises(ValueError, match="8个"):
        validate_selection_report(report)


def test_real_2043_draw_selection_reproduces_rejected_report() -> None:
    """重放冻结全量选择，防止代码与审计报告静默漂移。"""

    expected = json.loads(
        Path("reports/retrospective/ssq_challenger_e2_selection_v1.json").read_text(
            encoding="utf-8"
        )
    )
    actual = build_selection_report(
        "data/ssq/official_history.csv",
        ENSEMBLE_PATH,
    )
    assert actual["selectionStatus"] == "rejected"
    assert actual["selectedCandidateId"] is None
    assert actual["reportSha256"] == expected["reportSha256"]
    candidates = cast(dict[str, object], actual["validation"])["candidates"]
    assert all(
        cast(dict[str, object], item["eligibility"])["eligible"] is False
        for item in cast(list[dict[str, object]], candidates)
    )
