# -*- coding: utf-8 -*-
"""大乐透C5历史执行诊断报告测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.dlt_7plus2_c5_diagnostic_v1 import _load_development_prefix
from src.analysis.dlt_7plus2_c5_diagnostic_v1 import summarize_c5_diagnostic
from src.analysis.dlt_7plus2_c5_hedge_v1 import C5BlockPrediction
from src.analysis.dlt_ridge_candidates_v1 import CandidateScores


@dataclass(frozen=True)
class Draw:
    issue: str
    front: tuple[int, ...]
    back: tuple[int, ...]


def test_diagnostic_is_self_hashed_and_never_claims_independence() -> None:
    scores = CandidateScores((0.0,) * 35, (0.0,) * 12)
    draws = [Draw(str(index), (1, 2, 3, 4, 5), (1, 2)) for index in range(700)]
    predictions = tuple(
        C5BlockPrediction(
            target_index=index,
            fit_cutoff=index - index % 25,
            weights=(1 / 3, 1 / 3, 1 / 3),
            scores=scores,
            expert_scores=(scores, scores, scores),
        )
        for index in range(600, 700)
    )

    report = summarize_c5_diagnostic(
        draws,
        predictions,
        start_index=600,
        stop_index=700,
        protocol_sha256="a" * 64,
    )

    assert report["evidenceStatus"] == "exploratory_reused_development"
    assert report["independentEvidence"] is False
    assert report["formalGate"] is False
    assert report["formalOutput"] == "uniform_abstain"
    assert report["portfolioChangedCountVsC4"] == 0
    assert report["developmentDisposition"] == "retired_no_portfolio_change"
    assert report["boundary"]["maximumConsumedIndex"] == 699
    assert report["boundary"]["frozenRowsAccessed"] == 0
    assert report["metrics"]["meanJointLogLossImprovement"] == pytest.approx(0.0)
    assert report["metrics"]["meanHFImprovementVsC4"] == pytest.approx(0.0)
    assert report["metrics"]["meanHBImprovementVsC4"] == pytest.approx(0.0)
    assert report["metrics"]["meanUImprovementVsC4"] == pytest.approx(0.0)
    assert sum(report["metrics"]["frontHitDistribution"].values()) == 100
    assert sum(report["metrics"]["backHitDistribution"].values()) == 100
    assert len(report["metrics"]["jointBlockImprovements"]) == 1
    assert len(report["perIssue"]) == 100
    assert len(report["reportSha256"]) == 64


def test_prefix_loader_never_parses_tail_after_limit(tmp_path: Path) -> None:
    path = tmp_path / "dlt.csv"
    path.write_text(
        "issue,date,front,back,source_url,raw_hash\n"
        "1,2026-01-01,01 02 03 04 05,01 02,official," + "a" * 64 + "\n"
        "2,2026-01-02,06 07 08 09 10,03 04,official," + "b" * 64 + "\n"
        "BROKEN_TAIL\n",
        encoding="utf-8",
    )
    draws = _load_development_prefix(path, limit=2)
    assert [draw.issue for draw in draws] == ["1", "2"]
