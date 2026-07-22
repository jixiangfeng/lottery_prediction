# -*- coding: utf-8 -*-
"""learned_ranker_v4 完整化验收测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.digit_learned_ranker import main as learned_ranker_main
from src.analysis import digit_learned_ranker as ranker_module
from src.analysis import digit_learned_ranker_search as ranker_search
from src.analysis.digit_data import canonical_digit_data_sha256
from src.analysis.digit_learned_features import (
    FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
)
from src.analysis.digit_learned_ranker import (
    LearnedRankerParams,
    aggregate_group_candidates,
    generate_learned_ranker_daily,
    learned_ranker_source_fingerprint,
    load_params_artifact_fingerprint,
    params_fingerprint,
    partition_plan_by_activation,
    payload_fingerprint,
    process_learned_ranker_live_evaluations,
    resolve_activation,
    save_params,
)
from src.analysis.digit_learned_ranker_search import (
    FULL_HALF_LIVES,
    FULL_OMISSION_CAPS,
    FULL_TEMPERATURES,
    OBJECTIVE_PROFILES,
    LearnedSearchTrial,
    LearnedSplit,
    build_search_space_manifest,
    sample_feature_configs,
)
from src.analysis.digit_learned_ranker_walk_forward import (
    build_gate_result,
    run_learned_ranker_walk_forward,
    write_walk_forward_report,
)
from src.lotteries import get_lottery_rule
from tests.test_digit_learned_features import _history
from tests.test_digit_learned_ranker_cli import _write_csv


def test_window_weights_are_canonical_fingerprintable_and_change_scores():
    rule = get_lottery_rule("fc3d")
    recent = LearnedFeatureConfig(
        windows=(30, 300, "all"),
        window_weights=(("all", 1.0), ("30", 6.0), ("300", 2.0)),
    )
    canonical = LearnedFeatureConfig(
        windows=(30, 300, "all"),
        window_weights=(("30", 6), ("300", 2), ("all", 1)),
    )
    long_term = replace(canonical, window_weights=(("30", 1), ("300", 6), ("all", 2)))

    assert recent.window_weights == canonical.window_weights
    assert params_fingerprint(LearnedRankerParams(), recent) == params_fingerprint(
        LearnedRankerParams(), canonical
    )
    assert params_fingerprint(LearnedRankerParams(), recent) != params_fingerprint(
        LearnedRankerParams(), long_term
    )

    weighted_history = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": 9 if index >= 50 else index % 4,
                "十位": 8 if index >= 55 else (index * 3) % 7,
                "个位": 7 if index >= 60 else (index * 5) % 6,
            }
            for index in range(80)
        ]
    )
    candidates = ("012", "345", "678", "987", "999")
    recent_features = build_candidate_features(
        build_history_state(weighted_history, rule, recent), rule, candidates=candidates
    )
    long_features = build_candidate_features(
        build_history_state(weighted_history, rule, long_term),
        rule,
        candidates=candidates,
    )
    assert not np.allclose(
        recent_features["position_frequency"], long_features["position_frequency"]
    )


def test_source_fingerprint_bytes_ignore_platform_line_endings(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_bytes(b"first\nsecond\n")
    lf = ranker_module._canonical_source_bytes(source)
    source.write_bytes(b"first\r\nsecond\r\n")

    assert ranker_module._canonical_source_bytes(source) == lf


def test_active_feature_set_is_compact_and_finite():
    rule = get_lottery_rule("pl3")
    config = LearnedFeatureConfig(windows=(30, 50, 300, "all"))
    features = build_candidate_features(
        build_history_state(_history(90), rule, config), rule, candidates=("012", "987")
    )

    assert set(FEATURE_NAMES) == {
        "position_frequency",
        "position_omission",
        "pair_frequency",
        "sum_distribution",
        "span_distribution",
        "recent_trend",
        "position_trend",
        "pair_trend",
        "shape_transition",
        "shape_recent_deviation",
        "constraint_penalty",
    }
    assert set(FEATURE_NAMES).issubset(features.columns)
    assert np.isfinite(features[list(FEATURE_NAMES)].to_numpy()).all()


def test_canonical_data_hash_ignores_row_order_and_csv_formatting():
    rule = get_lottery_rule("fc3d")
    history = _history(20)
    shuffled = history.sample(frac=1.0, random_state=7).reset_index(drop=True)
    string_digits = shuffled.astype(str)

    assert canonical_digit_data_sha256(history, rule) == canonical_digit_data_sha256(
        string_digits, rule
    )
    changed = history.copy()
    changed.loc[0, "个位"] = 9
    assert canonical_digit_data_sha256(history, rule) != canonical_digit_data_sha256(
        changed, rule
    )


def test_split_supports_explicit_500_period_frozen_test():
    split = LearnedSplit.from_length(1000, frozen_test_periods=500)

    assert split.to_dict() == {"searchEnd": 250, "validationEnd": 500, "testEnd": 1000}


def test_objective_profiles_are_explicit_and_searchable():
    assert {
        "balanced",
        "direct_focus",
        "group_focus",
        "pool_focus",
        "direct_hit_only",
        "group_hit_only",
        "pool_hit_only",
    }.issubset(OBJECTIVE_PROFILES)
    assert OBJECTIVE_PROFILES["direct_hit_only"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert OBJECTIVE_PROFILES["group_hit_only"] == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert OBJECTIVE_PROFILES["pool_hit_only"] == (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def test_full_search_manifest_records_all_required_dimensions_and_profiles():
    manifest = build_search_space_manifest(smoke=False)

    assert (
        manifest["halfLife"]
        == list(FULL_HALF_LIVES)
        == [
            None,
            20,
            30,
            50,
            80,
            100,
            150,
            200,
        ]
    )
    assert manifest["omissionCap"] == list(FULL_OMISSION_CAPS) == [20, 30, 50, 80]
    assert (
        manifest["temperature"] == list(FULL_TEMPERATURES) == [0.5, 1.0, 2.0, 5.0, 10.0]
    )
    assert len(manifest["windowWeightProfiles"]) >= 4
    assert len(manifest["windowSets"]) >= 1
    assert len(manifest["alpha"]) >= 1
    assert manifest["groupAggregation"] == [
        "sum_prob",
        "max_perm",
        "mean_top_perm",
    ]
    assert manifest["uniformShrinkage"] == [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    assert manifest["abstentionUniformShrinkage"] == 0.0
    assert set(manifest["featureWeightRanges"]) == set(FEATURE_NAMES)
    assert manifest["objectiveProfiles"] == {
        name: list(weights) for name, weights in OBJECTIVE_PROFILES.items()
    }
    assert manifest["featureNormalization"] == "robust_zscore"
    assert manifest["recommendationConfig"] == {
        "directTopK": [50],
        "groupTopK": [10],
        "positionPoolSize": [5],
        "groupDigitPoolSize": [7],
    }
    assert manifest["sampling"]["evaluatedFeatureConfigs"] == []
    assert manifest["sampling"]["deterministic"] is True
    assert manifest["sampling"]["materializesFullCartesianProduct"] is False
    assert build_search_space_manifest(smoke=True)["space"] == manifest["space"]


def test_feature_config_sampling_is_seeded_bounded_and_uses_full_dimensions():
    first = sample_feature_configs(seed=17, smoke=False, limit=12)
    second = sample_feature_configs(seed=17, smoke=False, limit=12)

    assert first == second
    assert len(first) == 12
    assert first[0] == LearnedFeatureConfig()
    assert {item.half_life for item in first}.issubset(set(FULL_HALF_LIVES))
    assert {item.omission_cap for item in first}.issubset(set(FULL_OMISSION_CAPS))
    assert len({item.window_weights for item in first}) >= 3
    assert len(sample_feature_configs(seed=17, smoke=True)) == 1


def test_calibration_selection_uses_proper_scores_in_declared_order():
    better_log_loss = {
        "meanLogLoss": 6.8,
        "meanBrierScore": 0.999,
        "topKExpectedCalibrationError": 0.2,
    }
    better_brier_only = {
        "meanLogLoss": 6.9,
        "meanBrierScore": 0.8,
        "topKExpectedCalibrationError": 0.0,
    }

    assert ranker_search._calibration_selection_key(
        better_log_loss
    ) > ranker_search._calibration_selection_key(better_brier_only)


def test_strict_confirmation_requires_viability_and_positive_proper_scores(
    monkeypatch,
):
    hits = []
    for block_hits, block_size in ((17, 166), (17, 167), (16, 167)):
        hits.extend([True] * block_hits)
        hits.extend([False] * (block_size - block_hits))
    passing = ranker_search._ConfirmationSequences(
        kind="direct",
        hits=tuple(hits),
        random_probabilities=(0.05,) * 500,
        log_loss_improvements=(0.01,) * 500,
        brier_improvements=(0.001,) * 500,
    )
    monkeypatch.setattr(ranker_search, "_confirmation_sequences", lambda *args: passing)

    passed, reasons, diagnostics = ranker_search._validation_confirmation(
        (), LearnedRankerParams(direct_top_k=50), "direct_hit_only", seed=7
    )

    assert passed is True
    assert reasons == ()
    assert diagnostics["viability"]["conditions"] == {
        "enoughPeriods": True,
        "significant": True,
        "relativeLift": True,
        "confidenceLowerBound": True,
        "stableAcrossBlocks": True,
    }

    failing_scores = replace(
        passing,
        log_loss_improvements=(-0.01,) * 500,
    )
    monkeypatch.setattr(
        ranker_search, "_confirmation_sequences", lambda *args: failing_scores
    )
    passed, reasons, _ = ranker_search._validation_confirmation(
        (), LearnedRankerParams(direct_top_k=50), "direct_hit_only", seed=7
    )
    assert passed is False
    assert any("LogLoss" in reason for reason in reasons)


def test_search_trial_serializes_canonical_window_weights():
    config = LearnedFeatureConfig(
        windows=(30, 300, "all"),
        window_weights={"30": 4, "300": 2, "all": 1},
    )
    payload = LearnedSearchTrial(LearnedRankerParams(), config, 0.1, 0.2).to_dict()

    assert payload["featureConfig"]["windowWeights"] == {
        "30": 4.0,
        "300": 2.0,
        "all": 1.0,
    }


def test_partial_gate_activation_is_independent_for_three_outputs():
    direct_only = resolve_activation(
        common_passed=True,
        direct_passed=True,
        group_passed=False,
        position_passed=False,
    )
    group_only = resolve_activation(
        common_passed=True,
        direct_passed=False,
        group_passed=True,
        position_passed=False,
    )
    common_failed = resolve_activation(
        common_passed=False,
        direct_passed=True,
        group_passed=True,
        position_passed=True,
    )

    assert direct_only == {
        "commonPassed": True,
        "directPassed": True,
        "groupPassed": False,
        "positionPassed": False,
        "activeDirect": True,
        "activeGroup": False,
        "activePosition": False,
        "overallPassed": False,
        "overallSemantics": (
            "commonPassed && directPassed && groupPassed && positionPassed"
        ),
    }
    assert group_only["activeDirect"] is False
    assert group_only["activeGroup"] is True
    assert common_failed["activeDirect"] is False
    assert common_failed["activeGroup"] is False


def test_frozen_gate_report_splits_common_direct_and_group():
    gate = build_gate_result(
        mean_rank=420.0,
        mean_log_loss=6.8,
        direct_p_value=0.01,
        group_p_value=0.20,
        position_p_value=0.01,
        stable_blocks=2,
    )

    assert gate["common"]["passed"] is True
    assert gate["direct"]["passed"] is True
    assert gate["group"]["passed"] is False
    assert gate["passed"] is False
    assert gate["activation"]["activeDirect"] is True
    assert gate["activation"]["activeGroup"] is False


def test_daily_plan_separates_active_main_recommendation_from_research():
    plan = {
        "directCandidates": [{"text": "123"}],
        "groupCandidates": [{"group_key": "123"}],
        "positionPools": [[1], [2], [3]],
        "groupDigitPool": [1, 2, 3],
    }
    activation = resolve_activation(
        common_passed=True,
        direct_passed=False,
        group_passed=True,
        position_passed=False,
    )

    partitioned = partition_plan_by_activation(plan, activation)

    assert partitioned["activeDirect"] is False
    assert partitioned["activeGroup"] is True
    assert partitioned["mainRecommendation"] == {
        "directCandidates": [],
        "groupCandidates": [{"group_key": "123"}],
        "positionPools": [],
        "groupDigitPool": [1, 2, 3],
    }
    assert partitioned["research"]["directCandidates"] == [{"text": "123"}]
    assert partitioned["research"]["groupCandidates"] == []


def test_non_sum_group_aggregation_is_score_not_probability():
    candidates = [f"{value:03d}" for value in range(1000)]
    probabilities = np.full(1000, 0.001)

    summed = aggregate_group_candidates(
        candidates, probabilities, aggregation="sum_prob"
    )[0]
    maximum = aggregate_group_candidates(
        candidates, probabilities, aggregation="max_perm"
    )[0]

    assert summed.aggregation == "sum_prob"
    assert summed.probability is not None
    assert summed.score is None
    assert maximum.aggregation == "max_perm"
    assert maximum.probability is None
    assert maximum.score is not None
    assert "probability" not in maximum.to_dict()
    assert maximum.to_dict()["aggregation"] == "max_perm"


def test_daily_snapshot_is_self_describing_timezone_aware_and_immutable(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)
    save_params(
        LearnedRankerParams(),
        params_path,
        metadata={"experimentId": "exp_a", "ruleCode": "fc3d"},
    )

    _, _, snapshot_path = generate_learned_ranker_daily(
        "fc3d", csv_path, params_path, output_dir=output_dir
    )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert payload["schemaVersion"] >= 2
    assert payload["modelVersion"] == "learned_ranker_v4"
    assert payload["experimentId"] == "exp_a"
    assert payload["ruleCode"] == "fc3d"
    assert payload["sourceIssue"] == "2026020"
    assert payload["targetIssue"] is None
    assert payload["nextIssueInterpretation"]
    assert payload["generatedAt"].endswith("+08:00")
    assert payload["immutable"] is True
    assert payload["paramsFingerprint"]
    assert payload["paramsArtifactFingerprint"]
    assert payload["canonicalDataSha256"]
    assert payload["sourceFingerprint"]
    assert payload["activation"]
    assert payload["positionPassed"] is False
    assert set(payload["strategyStatuses"]) == {"direct", "group", "position"}
    assert (output_dir / payload["strategyRegistry"]).exists()
    assert payload["candidates"] == payload["plan"]
    assert payload["snapshotFingerprint"]
    assert payload["featureConfig"]["windowWeights"]
    markdown = (output_dir / "daily" / "fc3d").glob("*.md")
    assert "研究分区（直选，未启用）" in next(markdown).read_text(encoding="utf-8")


def test_live_review_resolves_first_issue_after_source_and_deduplicates(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)
    save_params(
        LearnedRankerParams(),
        params_path,
        metadata={"experimentId": "exp_a", "ruleCode": "fc3d"},
    )
    _, _, snapshot_path = generate_learned_ranker_daily(
        "fc3d", csv_path, params_path, output_dir=output_dir
    )
    _write_csv(csv_path, periods=22)
    history = pd.read_csv(csv_path, dtype=str)
    rule = get_lottery_rule("fc3d")

    first, summaries = process_learned_ranker_live_evaluations(
        history,
        rule,
        output_dir / "picks" / "digit",
        output_dir / "evaluations" / "learned_ranker_v4_live",
    )
    second, second_summaries = process_learned_ranker_live_evaluations(
        history,
        rule,
        output_dir / "picks" / "digit",
        output_dir / "evaluations" / "learned_ranker_v4_live",
    )

    assert len(first) == len(second) == 1
    assert first[0]["sourceIssue"] == "2026020"
    assert first[0]["targetIssue"] == "2026021"
    assert first[0]["dedupKey"] == second[0]["dedupKey"]
    assert summaries == second_summaries
    assert (
        len(
            list(
                (output_dir / "evaluations" / "learned_ranker_v4_live").glob(
                    "*.evaluation.json"
                )
            )
        )
        == 1
    )
    assert snapshot_path.exists()


def test_live_review_rejects_mutable_or_corrupted_v4_snapshot(tmp_path: Path):
    rule = get_lottery_rule("pl3")
    history = _history(22)
    picks = tmp_path / "picks"
    evaluations = tmp_path / "evaluations"
    picks.mkdir()
    mutable = {
        "schemaVersion": 2,
        "rankingMode": "learned_ranker_v4",
        "ruleCode": "pl3",
        "sourceIssue": "2026020",
        "immutable": False,
    }
    (picks / "pl3_learned_ranker_v4_mutable.json").write_text(
        json.dumps(mutable), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="immutable"):
        process_learned_ranker_live_evaluations(history, rule, picks, evaluations)


@pytest.mark.parametrize("lottery", ["fc3d", "pl3"])
def test_cli_train_evaluate_daily_smoke_for_each_supported_rule(
    tmp_path: Path, lottery: str
):
    csv_path = tmp_path / f"{lottery}.csv"
    output_dir = tmp_path / lottery
    params_path = tmp_path / f"{lottery}_params.json"
    _write_csv(csv_path, periods=16)
    script = Path(__file__).resolve().parents[1] / "scripts" / "digit_learned_ranker.py"
    common = [
        "--lottery",
        lottery,
        "--csv",
        str(csv_path),
        "--output-dir",
        str(output_dir),
    ]

    trained = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(script),
            "train",
            *common,
            "--params",
            str(params_path),
            "--min-train-size",
            "3",
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert trained.returncode == 2
    assert not params_path.exists()
    search_path = (
        output_dir / "evaluations" / f"learned_ranker_v4_search_{lottery}.json"
    )
    metadata = json.loads(search_path.read_text(encoding="utf-8"))
    assert metadata["smoke"] is True
    assert metadata["validationEvaluated"] is True
    assert metadata["validationPassed"] is False


def test_two_fresh_processes_produce_same_core_daily_artifact(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    _write_csv(csv_path, periods=20)
    save_params(
        LearnedRankerParams(),
        params_path,
        metadata={"experimentId": "deterministic", "ruleCode": "fc3d"},
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "digit_learned_ranker.py"
    payloads = []
    for name in ("first", "second"):
        output_dir = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                str(script),
                "daily",
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--params",
                str(params_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        daily_path = next(
            (output_dir / "daily" / "fc3d").glob("fc3d*_daily_2026020.json")
        )
        payload = json.loads(daily_path.read_text(encoding="utf-8"))
        payload.pop("generatedAt")
        payload.pop("snapshotFingerprint")
        payloads.append(payload)
    assert payloads[0] == payloads[1]


def test_distinct_experiments_and_params_keep_separate_snapshots_and_summaries(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)
    for experiment, temperature in (("exp_a", 1.0), ("exp_b", 0.5)):
        params_path = tmp_path / f"{experiment}.json"
        save_params(
            replace(LearnedRankerParams(), temperature=temperature),
            params_path,
            metadata={"experimentId": experiment, "ruleCode": "fc3d"},
        )
        generate_learned_ranker_daily(
            "fc3d", csv_path, params_path, output_dir=output_dir
        )
    snapshots = list((output_dir / "picks" / "digit").glob("*.json"))
    assert len(snapshots) == 2

    _write_csv(csv_path, periods=21)
    history = pd.read_csv(csv_path, dtype=str)
    evaluations, summaries = process_learned_ranker_live_evaluations(
        history,
        get_lottery_rule("fc3d"),
        output_dir / "picks" / "digit",
        output_dir / "evaluations" / "learned_ranker_v4_live",
    )
    assert {item["experimentId"] for item in evaluations} == {"exp_a", "exp_b"}
    assert len({item["paramsFingerprint"] for item in evaluations}) == 2
    assert len(summaries) == 2


def test_daily_rejects_re_fingerprinted_evaluation_with_wrong_canonical_hash(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)
    rule = get_lottery_rule("fc3d")
    history = pd.read_csv(csv_path, dtype=str)
    feature_config = LearnedFeatureConfig(windows=(5, "all"))
    params = LearnedRankerParams()
    save_params(
        params,
        params_path,
        metadata={
            "ruleCode": "fc3d",
            "featureConfig": {
                "windows": [5, "all"],
                "alpha": 2.0,
                "halfLife": None,
                "omissionCap": 50,
                "windowWeights": {"5": 1.0, "all": 1.0},
            },
        },
    )
    report = run_learned_ranker_walk_forward(
        history,
        rule,
        params,
        LearnedSplit(search_end=10, validation_end=14, test_end=20),
        feature_config=feature_config,
        canonical_data_sha256=canonical_digit_data_sha256(history, rule),
        source_fingerprint=learned_ranker_source_fingerprint(),
        params_artifact_fingerprint=load_params_artifact_fingerprint(params_path),
    )
    _, evaluation_path = write_walk_forward_report(report, output_dir / "evaluations")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["canonicalDataSha256"] = "0" * 64
    evaluation["gate"]["passed"] = True
    evaluation["gate"]["activation"] = resolve_activation(
        common_passed=True,
        direct_passed=True,
        group_passed=True,
        position_passed=True,
    )
    evaluation.pop("reportFingerprint")
    evaluation["reportFingerprint"] = payload_fingerprint(evaluation)
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _, daily_path, _ = generate_learned_ranker_daily(
        "fc3d",
        csv_path,
        params_path,
        output_dir=output_dir,
        evaluation_path=evaluation_path,
    )
    daily = json.loads(daily_path.read_text(encoding="utf-8"))
    assert daily["activation"]["activeDirect"] is False
    assert daily["activation"]["activeGroup"] is False
    assert daily["evaluationValidation"]["canonicalMatched"] is False


def test_daily_keeps_frozen_evaluation_valid_when_history_only_appends(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)
    rule = get_lottery_rule("fc3d")
    history = pd.read_csv(csv_path, dtype=str)
    split = LearnedSplit(search_end=10, validation_end=14, test_end=20)
    feature_config = LearnedFeatureConfig(
        windows=(5, "all"), window_weights={"5": 1.0, "all": 1.0}
    )
    params = LearnedRankerParams()
    frozen_canonical = canonical_digit_data_sha256(history, rule)
    save_params(
        params,
        params_path,
        metadata={
            "ruleCode": "fc3d",
            "canonicalDataSha256": frozen_canonical,
            "split": split.to_dict(),
            "testSegmentUsedForSelection": False,
            "validationPassed": True,
            "sourceFingerprint": learned_ranker_source_fingerprint(),
            "featureConfig": {
                "windows": [5, "all"],
                "alpha": 2.0,
                "halfLife": None,
                "omissionCap": 50,
                "windowWeights": {"5": 1.0, "all": 1.0},
            },
        },
    )
    report = run_learned_ranker_walk_forward(
        history,
        rule,
        params,
        split,
        feature_config=feature_config,
        canonical_data_sha256=frozen_canonical,
        source_fingerprint=learned_ranker_source_fingerprint(),
        params_artifact_fingerprint=load_params_artifact_fingerprint(params_path),
    )
    _, evaluation_path = write_walk_forward_report(report, output_dir / "evaluations")

    appended = pd.concat(
        [
            history,
            pd.DataFrame([{history.columns[0]: "2026021", history.columns[1]: "123"}]),
        ],
        ignore_index=True,
    )
    appended.to_csv(csv_path, index=False)
    assert (
        learned_ranker_main(
            [
                "evaluate",
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--params",
                str(params_path),
            ]
        )
        == 0
    )
    _, daily_path, _ = generate_learned_ranker_daily(
        "fc3d",
        csv_path,
        params_path,
        output_dir=output_dir,
        evaluation_path=evaluation_path,
    )
    daily = json.loads(daily_path.read_text(encoding="utf-8"))

    assert daily["evaluationValidation"]["canonicalMatched"] is True
    assert daily["frozenDataCanonicalSha256"] == frozen_canonical
    assert daily["canonicalDataSha256"] != frozen_canonical


def test_prepared_target_disk_cache_round_trip(tmp_path):
    candidates = [f"{value:03d}" for value in range(1000)]
    features = pd.DataFrame(
        {
            "candidate": candidates,
            **{
                name: np.arange(1000, dtype=float) + feature_index
                for feature_index, name in enumerate(FEATURE_NAMES)
            },
        }
    )
    targets = (ranker_search._PreparedTarget(features, "123", "122"),)
    path = tmp_path / "prepared.npz"

    ranker_search._write_prepared_cache(path, targets)
    restored = ranker_search._read_prepared_cache(path)

    assert restored is not None
    assert restored[0].actual_text == "123"
    assert restored[0].latest_exact == "122"
    pd.testing.assert_frame_equal(restored[0].features, features)


def test_pool_hit_objective_uses_fixed_probability_mass_pool():
    candidates = [f"{value:03d}" for value in range(1000)]
    features = pd.DataFrame(
        {
            "candidate": candidates,
            **{name: np.zeros(1000) for name in FEATURE_NAMES},
        }
    )
    target = ranker_search._PreparedTarget(features=features, actual_text="999")

    objective = ranker_search._objective(
        (target,),
        LearnedRankerParams(position_pool_size=5),
        objective_profile="pool_hit_only",
    )

    assert objective == 0.0


def test_direct_objective_uses_the_same_daily_candidate_policy_as_output():
    candidates = [f"{number:03d}" for number in range(1000)]
    features = pd.DataFrame({"candidate": candidates})
    for name in FEATURE_NAMES:
        features[name] = 0.0
    features.loc[0, "position_frequency"] = 100.0
    params = LearnedRankerParams(
        weights={
            name: (1.0 if name == "position_frequency" else 0.0)
            for name in FEATURE_NAMES
        },
        direct_top_k=50,
    )

    raw = ranker_search._PreparedTarget(features, "000")
    policy = ranker_search._PreparedTarget(features, "000", latest_exact="000")

    assert (
        ranker_search._objective((raw,), params, objective_profile="direct_hit_only")
        == 1.0
    )
    assert (
        ranker_search._objective((policy,), params, objective_profile="direct_hit_only")
        == 0.0
    )


def test_profile_searches_share_prepared_target_cache(monkeypatch):
    history = _history(20)
    rule = get_lottery_rule("fc3d")
    split = LearnedSplit(search_end=10, validation_end=14, test_end=20)
    base = ranker_search.LearnedSearchConfig(
        split=split,
        min_train_size=6,
        random_trials=1,
        local_trials=0,
        evaluation_stride=2,
        feature_configs=(LearnedFeatureConfig(windows=(10, "all")),),
        objective_profile="direct_hit_only",
        direct_objective_top_k=50,
        position_objective_pool_size=3,
        require_search_qualification=False,
    )
    calls = 0
    original = ranker_search._prepare_targets

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ranker_search, "_prepare_targets", counted)
    cache = {}
    direct_result = ranker_search.search_learned_ranker_params(
        history, rule, base, prepared_target_cache=cache
    )
    ranker_search.search_learned_ranker_params(
        history,
        rule,
        replace(base, objective_profile="group_hit_only"),
        prepared_target_cache=cache,
    )

    assert calls == 2
    assert direct_result.params.direct_top_k == 50
    assert direct_result.params.position_pool_size == 3


def test_joint_budget_selector_uses_search_and_validates_stability():
    def metric(lift, blocks):
        return {
            "lift": lift,
            "timeBlocks": [{"lift": value} for value in blocks],
        }

    curves = {
        "fc3d": {
            "search": {
                "direct": {
                    "10": metric(1.5, [1.2, 0.8, 1.1]),
                    "20": metric(1.3, [1.1, 1.0, 1.2]),
                    "1000": metric(1.0, [1.0, 1.0, 1.0]),
                }
            },
            "validation": {
                "direct": {
                    "10": metric(1.1, [1.2, 0.7, 1.1]),
                    "20": metric(1.2, [1.1, 1.0, 1.1]),
                    "1000": metric(1.0, [1.0, 1.0, 1.0]),
                }
            },
        },
        "pl3": {
            "search": {
                "direct": {
                    "10": metric(1.4, [0.9, 1.2, 0.8]),
                    "20": metric(1.2, [1.0, 1.1, 1.2]),
                    "1000": metric(1.0, [1.0, 1.0, 1.0]),
                }
            },
            "validation": {
                "direct": {
                    "10": metric(1.3, [1.1, 0.9, 1.2]),
                    "20": metric(1.1, [1.0, 1.1, 1.0]),
                    "1000": metric(1.0, [1.0, 1.0, 1.0]),
                }
            },
        },
    }

    selected = ranker_search.select_joint_budget(
        curves, kind="direct", full_coverage_budget=1000
    )

    assert selected["selectedBudget"] == 20
    assert selected["searchQualified"] is True
    assert selected["validationConfirmed"] is True
    assert selected["validationUsedForSelection"] is False
