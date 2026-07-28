# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest

from scripts.ssq_challenger_e_v1 import build_parser
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    UNIFORM_RED_PROBABILITY,
    CalibratedRedChallengerEState,
    build_current_report,
    train_state,
    walk_forward_prediction_fingerprints,
)
from src.analysis.ssq_diversified_portfolio_v2 import (
    build_diversified_portfolio_v2,
)
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)

# 当前滚动Ensemble已通过2026086；E2冻结工件另有独立哈希断言。
ENSEMBLE_FILE_SHA256 = (
    "a7d5c381cb2979db5f6123da5184d4923e9d5747c6503b327ddb6162851b3c04"
)
ENSEMBLE_REPORT_SHA256 = (
    "bbe1b751d5a990680bdd33342f5df1319c14e78b267b29cd6400f28d4f56f909"
)


def _draw(index: int, issue: str | None = None) -> SSQDraw:
    red = tuple(sorted(((index + offset * 5) % 33) + 1 for offset in range(6)))
    return SSQDraw(
        issue=issue or str(2024001 + index),
        draw_date=f"2024-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
        red=red,
        blue=index % 16 + 1,
        source_url="fixture",
        raw_hash=f"{index:064x}",
        raw={},
    )


def test_initial_prediction_is_uniform_and_calibrated_to_sum_six() -> None:
    locked = CalibratedRedChallengerEState().predict()

    assert locked.probabilities == pytest.approx([UNIFORM_RED_PROBABILITY] * 33)
    assert sum(locked.probabilities) == pytest.approx(6.0, abs=1e-12)
    assert all(
        row == pytest.approx((0.0, 0.0, -0.5, -6.0 / 33.0, 0.0))
        for row in locked.features
    )


def test_feature_equations_match_frozen_protocol_exactly() -> None:
    state = CalibratedRedChallengerEState()
    first_draw = SSQDraw(
        issue="2024001",
        draw_date="2024-01-01",
        red=(1, 2, 3, 4, 5, 6),
        blue=1,
        source_url="fixture",
        raw_hash="0" * 64,
        raw={},
    )
    state.predict()
    state.score_then_update(first_draw)
    locked = state.predict()

    selected_probability = 12.0 / 39.0
    unselected_probability = 6.0 / 39.0
    selected_pair_modifier = math.log((30.0 / 543.0) / (15.0 / 528.0))
    unselected_pair_modifier = math.log((15.0 / 543.0) / (15.0 / 528.0))
    expected_gap = math.log1p(1) / math.log(61.0) - 0.5

    assert locked.features[0] == pytest.approx(
        (
            (selected_probability - 6.0 / 33.0) / 0.05,
            (selected_probability - 6.0 / 33.0) / 0.05,
            expected_gap,
            1.0 - 6.0 / 33.0,
            selected_pair_modifier,
        )
    )
    assert locked.features[6] == pytest.approx(
        (
            (unselected_probability - 6.0 / 33.0) / 0.05,
            (unselected_probability - 6.0 / 33.0) / 0.05,
            expected_gap,
            -6.0 / 33.0,
            unselected_pair_modifier,
        )
    )


def test_prediction_lock_order_calibration_and_fingerprint_are_deterministic() -> None:
    state = CalibratedRedChallengerEState()
    first = state.predict()
    with pytest.raises(RuntimeError, match="未消费"):
        state.predict()
    assert sum(first.probabilities) == pytest.approx(6.0, abs=1e-12)

    restored = CalibratedRedChallengerEState.from_dict(state.state_payload())
    assert restored.pending_prediction == first
    with pytest.raises(RuntimeError, match="未消费"):
        restored.predict()
    restored.score_then_update(_draw(0))
    second = restored.predict()
    assert second.prediction_fingerprint != first.prediction_fingerprint
    assert sum(second.probabilities) == pytest.approx(6.0, abs=1e-12)


def test_serialization_roundtrip_rejects_tampered_pending_lock() -> None:
    state = CalibratedRedChallengerEState()
    state.predict()
    payload = state.state_payload()
    restored = CalibratedRedChallengerEState.from_dict(payload)
    assert restored.state_payload() == payload

    tampered = json.loads(json.dumps(payload))
    pending = cast(dict[str, object], tampered["pendingPrediction"])
    probabilities = cast(list[float], pending["probabilities"])
    probabilities[0] += 0.01
    with pytest.raises(ValueError, match="预测锁"):
        CalibratedRedChallengerEState.from_dict(tampered)


def test_ties_use_ball_ascending_and_legal_28_tickets_have_zero_b_overlap() -> None:
    red = [UNIFORM_RED_PROBABILITY] * 33
    blue = [1.0 / 16.0] * 16
    b_document = build_diversified_portfolio_v2(red, blue)
    candidate = build_small_compound_8red1blue_v1(red, blue, b_document)

    assert candidate["top12RedRanking"] == list(range(1, 13))
    assert candidate["blue"] == 1
    assert len(candidate["expandedTickets"]) == 28
    assert candidate["audit"]["overlapWithB"] == 0
    assert candidate["audit"]["combinedUniqueTicketCount"] == 63


def test_training_and_walk_forward_are_prefix_invariant() -> None:
    prefix = [_draw(index) for index in range(8)]
    extended = [*prefix, _draw(20, issue="2030001"), _draw(21, issue="2030002")]

    assert walk_forward_prediction_fingerprints(extended)[: len(prefix)] == (
        walk_forward_prediction_fingerprints(prefix)
    )
    trained = train_state(list(reversed(prefix)))
    assert trained.periods_seen == len(prefix)
    assert trained.pending_prediction is None


def test_cli_exposes_paths_only_and_no_tuning_options() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert option_strings == {"--csv", "--ensemble-report", "--output"}


def test_current_builder_keeps_existing_ensemble_bytes_and_report_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ensemble_path = Path("reports/research/ssq_ensemble_v1.json")
    before = ensemble_path.read_bytes()
    embedded = json.loads(before)["reportSha256"]
    assert hashlib.sha256(before).hexdigest() == ENSEMBLE_FILE_SHA256
    assert embedded == ENSEMBLE_REPORT_SHA256

    csv_path = tmp_path / "history.csv"
    csv_path.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.analysis.ssq_calibrated_red_challenger_e_v1.load_official_history_csv",
        lambda _: [_draw(0), _draw(1)],
    )
    report = build_current_report(csv_path, ensemble_path)

    assert ensemble_path.read_bytes() == before
    assert json.loads(ensemble_path.read_bytes())["reportSha256"] == embedded
    assert report["ensembleIntegrity"] == {
        "bytesUnchanged": True,
        "fileSha256Unchanged": True,
        "reportSha256Unchanged": True,
    }
