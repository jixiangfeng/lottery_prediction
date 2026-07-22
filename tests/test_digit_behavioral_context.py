# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from scripts import digit_behavioral_context as behavioral_cli
from src.analysis.digit_behavioral_context import (
    BehavioralContextConfig,
    run_behavioral_context_challenge,
)
from src.analysis.digit_online_gradient import OnlineGradientConfig
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "期数": [str(2026001 + index) for index in range(periods)],
            "百位": [(index * 7) % 10 for index in range(periods)],
            "十位": [(index * 3 + 1) % 10 for index in range(periods)],
            "个位": [(index * 9 + 2) % 10 for index in range(periods)],
        }
    )


def test_behavioral_context_challenge_runs_paired_prior_only_groups():
    online = OnlineGradientConfig(
        development_end=130,
        outer_periods=30,
        warmup_history=50,
        search_lookback=30,
        validation_lookback=20,
        calibration_interval=10,
        learning_rates=(0.0, 0.02),
        shrinkages=(0.0, 0.25),
    )

    report = run_behavioral_context_challenge(
        _history(130),
        get_lottery_rule("fc3d"),
        BehavioralContextConfig(
            online=online,
            paired_permutations=99,
        ),
    )

    assert report["modelVersion"] == "behavioral_context_v4"
    assert len(report["dataSha256"]) == 64
    assert len(report["sourceFingerprint"]) == 64
    assert report["frozenTestRead"] is False
    assert report["groups"]["A"]["periods"] == 30
    assert report["groups"]["B"]["periods"] == 30
    assert report["groups"]["C"]["periods"] == 30
    assert set(report["groups"]) == {"A", "B", "C"}
    assert report["comparison"]["pairedPeriods"] == 30
    assert report["comparison"] == report["comparisons"]["CvsA"]
    assert report["comparisons"]["BvsA"]["pairedPeriods"] == 30
    assert 0 <= report["comparison"]["pairedTop50PValue"] <= 1
    assert report["behavioralFeatureL2Multiplier"] == 2.0
    assert report["behavioralGradientClip"] == 0.25
    assert report["behavioralFeatureProfile"] == "minimal_v4"
    assert report["behavioralFeatures"] == [
        "exact_recency_risk",
        "last_position_overlap_risk",
    ]
    assert report["behavioralFeatureNormalization"] == "per_query_centered_bounded_z"
    assert set(report["behavioralFeatureSemantics"]) == set(
        report["behavioralFeatures"]
    )
    assert report["primaryChallengerGroup"] == "C"
    assert report["winnerSelectionAllowed"] is False
    assert report["candidatePolicy"] == {
        "excludeLatestExact": True,
        "topK": 50,
        "baselineMaximumTop50Triples": 1,
        "challengerMaximumTop50Triples": 0,
    }
    assert report["groups"]["A"]["dailyCandidatePolicyApplied"] is True
    assert report["groups"]["A"]["maximumTop50TriplesAllowed"] == 1
    assert report["groups"]["B"]["maximumTop50TriplesAllowed"] == 0
    assert report["groups"]["B"]["maximumTop50Triples"] == 0
    assert report["groups"]["C"]["allFixedBlocksAtOrAboveRandom"] is False
    assert 0 <= report["groups"]["C"]["shapeDistributionTotalVariation"] <= 1
    assert all(
        report["groups"]["C"]["finalWeights"][name] <= 0
        for name in report["behavioralFeatures"]
    )
    assert set(report["groups"]["C"]["featureAttribution"]) == set(
        report["groups"]["C"]["finalWeights"]
    )
    assert report["groups"]["A"]["behavioralBoundaryContribution"] is None
    assert report["groups"]["C"]["behavioralBoundaryContribution"] is not None
    assert report["developmentCoverage"] == {
        "availableOuterPeriods": 30,
        "completeDevelopmentPeriods": 0,
        "evaluatedOuterPeriods": 30,
        "expectedFirstTargetIndex": 100,
        "actualFirstTargetIndex": 100,
        "allCompleteFixedBlocksEvaluated": False,
    }
    assert report["gate"]["passed"] is False
    assert report["gate"]["minimumPeriods"] == 500
    assert report["gate"]["maximumShapeTotalVariation"] == 0.10
    assert "完整固定时间块未全部达到随机基线" in report["gate"]["reasons"]
    assert "未覆盖开发区全部完整固定时间块" in report["gate"]["reasons"]
    assert "行为特征Top50边界贡献为负或跨块不稳定" in report["gate"]["reasons"]


def test_behavioral_context_config_defaults_to_minimal_v4_package_and_rejects_invalid_subset():
    online = OnlineGradientConfig(development_end=600)

    config = BehavioralContextConfig(online=online)
    assert config.behavioral_feature_names == (
        "exact_recency_risk",
        "last_position_overlap_risk",
    )
    assert config.baseline_maximum_top50_triples == 1
    assert config.challenger_maximum_top50_triples == 0
    with pytest.raises(ValueError, match="必须非空"):
        BehavioralContextConfig(online=online, behavioral_feature_names=())
    with pytest.raises(ValueError, match="未知特征"):
        BehavioralContextConfig(
            online=online,
            behavioral_feature_names=("unknown_behavior",),
        )


def test_behavioral_cli_excludes_frozen_and_can_scan_complete_development_blocks(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        behavioral_cli,
        "load_digit_development_csv",
        lambda *args, **kwargs: (_history(1200), 1700),
    )

    def fake_run(history, rule, config, *, progress_callback=None):
        captured["historyPeriods"] = len(history)
        captured["lottery"] = rule.code
        captured["outerPeriods"] = config.online.outer_periods
        captured["developmentEnd"] = config.online.development_end
        captured["behavioralFeatures"] = config.behavioral_feature_names
        captured["progressCallback"] = callable(progress_callback)
        return {}

    def fake_write(report, path):
        captured["report"] = report
        return tmp_path / "behavior.json"

    monkeypatch.setattr(behavioral_cli, "run_behavioral_context_challenge", fake_run)
    monkeypatch.setattr(behavioral_cli, "write_behavioral_context_report", fake_write)

    exit_code = behavioral_cli.main(
        [
            "--lottery",
            "pl3",
            "--csv",
            "history.csv",
            "--output",
            str(tmp_path / "behavior.json"),
            "--all-development-blocks",
            "--behavior-features",
            "exact_recency_risk",
            "shape_run_excess_risk",
        ]
    )

    assert exit_code == 0
    assert captured["historyPeriods"] == 1200
    assert captured["lottery"] == "pl3"
    assert captured["outerPeriods"] == 500
    assert captured["developmentEnd"] == 1050
    assert captured["behavioralFeatures"] == (
        "exact_recency_risk",
        "shape_run_excess_risk",
    )
    assert captured["progressCallback"] is True
    assert captured["report"] == {
        "dataBoundary": {
            "fullPeriods": 1700,
            "developmentPeriods": 1200,
            "frozenPeriodsExcluded": 500,
            "evaluationEndIndex": 1050,
            "trailingDevelopmentPeriodsExcluded": 150,
            "outerPeriods": 500,
            "allDevelopmentBlocks": True,
        }
    }
