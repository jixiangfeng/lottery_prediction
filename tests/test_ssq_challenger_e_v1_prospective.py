# -*- coding: utf-8 -*-
"""Challenger E1 前瞻链的固定协议与结构回归测试。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from scripts.ssq_challenger_e_v1_prospective import _parser
from src.analysis import ssq_8red1blue_v1_prospective as d8_prospective
from src.analysis import ssq_challenger_e_v1_prospective as prospective
from src.analysis.ssq_history import SSQDraw


def _draw() -> SSQDraw:
    return SSQDraw(
        issue="2026086",
        draw_date="2026-07-28",
        red=(1, 2, 3, 4, 5, 6),
        blue=7,
        source_url="fixture",
        raw_hash="f" * 64,
        raw={},
    )


def test_protocol_is_fixed_future_only_and_never_auto_promotes() -> None:
    assert prospective.HORIZON == 500
    assert prospective.BLOCK_COUNT == 5
    assert prospective.BLOCK_SIZE == 100
    claims = cast(Mapping[str, object], prospective.PROTOCOL["claims"])
    assert claims["researchOnly"] is True
    assert claims["predictionClaim"] is False
    assert claims["autoPromotion"] is False
    assert claims["formalRecommendationStatus"] == "uniform_abstain"


def test_real_b_document_expands_to_exactly_35_unique_tickets() -> None:
    report = json.loads(
        Path("reports/research/ssq_ensemble_v1.json").read_text(encoding="utf-8")
    )
    assert len(prospective._b_tickets(report["diversifiedPortfolioV2"])) == 35


def test_proper_scores_are_per_ball_and_finite() -> None:
    probabilities = [6.0 / 33.0] * 33
    scores = prospective._proper_scores(probabilities, _draw())
    assert 0.0 < scores["redLogLossPerBall"] < 1.0
    assert 0.0 < scores["redBrierPerBall"] < 1.0


def test_compound_scoring_uses_locked_red8_and_blue() -> None:
    portfolio = {"red": [1, 2, 3, 4, 8, 9, 10, 11], "blue": 7}
    scores = prospective._compound_scores(portfolio, _draw())
    assert scores["red8Overlap"] == 4
    assert scores["atLeast3"] is True
    assert scores["atLeast4"] is True
    assert scores["atLeast5"] is False
    assert scores["blueHit"] is True
    assert scores["exact6PlusBlue"] is False


def test_gates_are_not_evaluated_before_exactly_500() -> None:
    assert prospective._gate_evaluation(0, [], {}, {}, {}) is None
    assert prospective._gate_evaluation(499, [], {}, {}, {}) is None


def test_cli_exposes_operations_and_paths_but_no_model_tuning() -> None:
    parser = _parser()
    help_text = parser.format_help()
    assert "register" in help_text
    assert "snapshot" in help_text
    assert "update" in help_text
    assert "status" in help_text
    forbidden = ["eta", "learning-rate", "l2", "warmup", "window", "top-k"]
    assert all(name not in help_text.lower() for name in forbidden)


def test_current_e_and_ensemble_reports_remain_self_hashed() -> None:
    for path, label in (
        ("reports/research/ssq_challenger_e_v1.json", "E1当前报告"),
        ("reports/research/ssq_ensemble_v1.json", "ensemble报告"),
    ):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        prospective._verify_self_hashed_report(document, label)


def test_real_reports_register_snapshot_and_status_in_process(tmp_path: Path) -> None:
    """用临时密钥完整重放D8→E登记和首快照，不触碰生产状态。"""

    canonical_csv = Path("data/ssq/official_history.csv")
    ensemble_report = Path("reports/research/ssq_ensemble_v1.json")
    e_report = Path("reports/research/ssq_challenger_e_v1.json")
    d8_state = tmp_path / "d8"
    e_state = tmp_path / "e"
    d8_key = b"d" * 32
    e_key = b"e" * 32
    before_deadline = datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc)

    registered_d8 = d8_prospective.register_prospective(
        canonical_csv,
        ensemble_report,
        d8_state,
        hmac_key=d8_key,
        now=before_deadline,
    )
    snapshot_d8 = d8_prospective.create_snapshot(
        canonical_csv,
        ensemble_report,
        d8_state,
        hmac_key=d8_key,
        now=before_deadline,
    )
    assert registered_d8["targetIssue"] == snapshot_d8["targetIssue"]

    registered_e = prospective.register_prospective(
        canonical_csv,
        e_report,
        ensemble_report,
        e_state,
        d8_state_dir=d8_state,
        hmac_key=e_key,
        d8_hmac_key=d8_key,
        now=before_deadline,
    )
    snapshot_e = prospective.create_snapshot(
        canonical_csv,
        e_report,
        ensemble_report,
        e_state,
        d8_state_dir=d8_state,
        hmac_key=e_key,
        d8_hmac_key=d8_key,
        now=before_deadline,
    )
    status = prospective.prospective_status(
        e_state,
        canonical_csv=canonical_csv,
        hmac_key=e_key,
    )

    assert registered_e["targetIssue"] == snapshot_e["targetIssue"]
    assert status["completed"] == 0
    assert status["remaining"] == 500
    assert status["pendingExactOneIssueUpdate"] is False
    assert status["formalRecommendationStatus"] == "uniform_abstain"
