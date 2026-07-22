# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pytest

from src.analysis.digit_model_scoreboard import (
    build_model_scoreboard,
    render_model_scoreboard_markdown,
    write_model_scoreboard,
)


def _behavior_report(version: str, lottery: str, hits: int) -> dict[str, object]:
    return {
        "modelVersion": version,
        "lottery": lottery,
        "frozenTestRead": False,
        "currentDailyModelReplaced": False,
        "developmentCoverage": {
            "allCompleteFixedBlocksEvaluated": True,
            "evaluatedOuterPeriods": 6500,
        },
        "groups": {
            "C": {
                "periods": 6500,
                "hits": hits,
                "hitRate": hits / 6500,
                "top50PValue": 0.2,
                "meanLogLoss": 6.91,
                "meanBrier": 0.999,
                "fixed500BlockHitRates": [0.05] * 13,
                "abstainedPeriods": 6400,
                "behavioralBoundaryContribution": {
                    "mean": -0.2,
                    "fixedBlockMeans": [-0.2] * 13,
                    "allFixedBlocksPositive": False,
                },
            }
        },
        "comparisons": {
            "CvsA": {
                "pairedTop50PValue": 0.3,
                "meanLogLossImprovement": -0.001,
                "meanBrierImprovement": -0.000001,
            }
        },
        "gate": {"passed": False, "newShadowStateAllowed": False},
    }


def test_scoreboard_keeps_all_comparable_models_and_never_selects_failed_winner():
    reports = [
        _behavior_report("behavioral_context_v3", "fc3d", 333),
        _behavior_report("behavioral_context_v3", "pl3", 331),
        _behavior_report("behavioral_context_v4", "fc3d", 314),
        _behavior_report("behavioral_context_v4", "pl3", 362),
    ]

    scoreboard = build_model_scoreboard(reports)

    assert scoreboard["schemaVersion"] == "digit_model_scoreboard_v1"
    assert len(scoreboard["directTop50Evidence"]) == 12
    assert {row["lottery"] for row in scoreboard["directTop50Evidence"]} == {
        "fc3d",
        "pl3",
    }
    assert all(row["periods"] >= 6000 for row in scoreboard["directTop50Evidence"])
    assert all(
        row["holmAdjustedRandomPValue"] >= row["randomPValue"]
        for row in scoreboard["directTop50Evidence"]
    )
    assert scoreboard["decision"]["selectedModel"] is None
    assert scoreboard["decision"]["productionMode"] == "uniform_abstain"
    assert scoreboard["decision"]["behaviorFamilyRetired"] is True
    assert scoreboard["decision"]["coreProspectiveCheckpoints"] == [50, 100, 200]
    assert scoreboard["nonComparableDiagnostics"]
    assert len(scoreboard["incompleteEvidence"]) == 3


def test_scoreboard_rejects_behavior_report_that_reads_frozen_or_skips_blocks():
    frozen = _behavior_report("behavioral_context_v4", "fc3d", 314)
    frozen["frozenTestRead"] = True
    with pytest.raises(ValueError, match="不得读取Frozen"):
        build_model_scoreboard([frozen])

    incomplete = _behavior_report("behavioral_context_v4", "fc3d", 314)
    incomplete["developmentCoverage"] = {
        "allCompleteFixedBlocksEvaluated": False,
        "evaluatedOuterPeriods": 500,
    }
    with pytest.raises(ValueError, match="全部完整固定块"):
        build_model_scoreboard([incomplete])


def test_write_scoreboard_adds_deterministic_content_hash(tmp_path: Path):
    scoreboard = build_model_scoreboard(
        [
            _behavior_report("behavioral_context_v3", "fc3d", 333),
            _behavior_report("behavioral_context_v3", "pl3", 331),
            _behavior_report("behavioral_context_v4", "fc3d", 314),
            _behavior_report("behavioral_context_v4", "pl3", 362),
        ]
    )

    destination = write_model_scoreboard(scoreboard, tmp_path / "scoreboard.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert len(payload["scoreboardSha256"]) == 64
    assert payload["decision"]["selectedModel"] is None


def test_markdown_scoreboard_exposes_no_winner_and_all_direct_rows():
    scoreboard = build_model_scoreboard(
        [
            _behavior_report("behavioral_context_v3", "fc3d", 333),
            _behavior_report("behavioral_context_v3", "pl3", 331),
            _behavior_report("behavioral_context_v4", "fc3d", 314),
            _behavior_report("behavioral_context_v4", "pl3", 362),
        ]
    )

    markdown = render_model_scoreboard_markdown(scoreboard)

    assert "无模型入选" in markdown
    assert markdown.count("| direct_top50 |") == 12
    assert "uniform_abstain" in markdown
