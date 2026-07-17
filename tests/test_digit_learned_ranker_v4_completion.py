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


def test_trend_features_keep_independent_components_and_named_horizons():
    rule = get_lottery_rule("pl3")
    config = LearnedFeatureConfig(windows=(30, 50, 300, "all"))
    features = build_candidate_features(
        build_history_state(_history(90), rule, config), rule, candidates=("012", "987")
    )

    component_names = ("position", "pair", "sum", "span", "shape")
    required = {
        "position_trend",
        "pair_trend",
        "sum_trend",
        "span_trend",
        "shape_trend",
        "trend_30_300",
        "trend_50_all",
        "trend_ratio_30_300",
        *(f"{name}_trend_30_300" for name in component_names),
        *(f"{name}_trend_50_all" for name in component_names),
        *(f"{name}_trend_ratio_30_300" for name in component_names),
    }
    assert required.issubset(FEATURE_NAMES)
    assert required.issubset(features.columns)
    assert np.isfinite(features[list(required)].to_numpy()).all()
    assert not np.allclose(
        features["position_trend"].to_numpy(), features["pair_trend"].to_numpy()
    )
    assert not np.allclose(features["trend_ratio_30_300"].to_numpy(), 0.0)
    assert any(
        not np.allclose(features[f"{name}_trend_ratio_30_300"].to_numpy(), 0.0)
        for name in component_names
    )


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
        manifest["temperature"] == list(FULL_TEMPERATURES) == [0.1, 0.2, 0.5, 1.0, 2.0]
    )
    assert len(manifest["windowWeightProfiles"]) >= 4
    assert len(manifest["windowSets"]) >= 1
    assert len(manifest["alpha"]) >= 1
    assert manifest["groupAggregation"] == [
        "sum_prob",
        "max_perm",
        "mean_top_perm",
    ]
    assert set(manifest["featureWeightRanges"]) == set(FEATURE_NAMES)
    assert manifest["featureNormalization"] == "robust_zscore"
    assert manifest["recommendationConfig"] == {
        "directTopK": [10],
        "groupTopK": [10],
        "positionPoolSize": [5],
        "groupDigitPoolSize": [7],
    }
    assert manifest["sampling"]["deterministic"] is True
    assert manifest["sampling"]["materializesFullCartesianProduct"] is False
    assert build_search_space_manifest(smoke=True)["space"] == manifest["space"]


def test_feature_config_sampling_is_seeded_bounded_and_uses_full_dimensions():
    first = sample_feature_configs(seed=17, smoke=False, limit=12)
    second = sample_feature_configs(seed=17, smoke=False, limit=12)

    assert first == second
    assert len(first) == 12
    assert {item.half_life for item in first}.issubset(set(FULL_HALF_LIVES))
    assert {item.omission_cap for item in first}.issubset(set(FULL_OMISSION_CAPS))
    assert len({item.window_weights for item in first}) >= 3
    assert len(sample_feature_configs(seed=17, smoke=True)) == 1


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


def test_partial_gate_activation_is_independent_and_overall_is_compatibility_only():
    direct_only = resolve_activation(
        common_passed=True, direct_passed=True, group_passed=False
    )
    group_only = resolve_activation(
        common_passed=True, direct_passed=False, group_passed=True
    )
    common_failed = resolve_activation(
        common_passed=False, direct_passed=True, group_passed=True
    )

    assert direct_only == {
        "commonPassed": True,
        "directPassed": True,
        "groupPassed": False,
        "activeDirect": True,
        "activeGroup": False,
        "overallPassed": False,
        "overallSemantics": "commonPassed && directPassed && groupPassed（兼容字段）",
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
        common_passed=True, direct_passed=False, group_passed=True
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
    assert payload["candidates"] == payload["plan"]
    assert payload["snapshotFingerprint"]
    assert payload["featureConfig"]["windowWeights"]
    markdown = (output_dir / "learned_ranker_v4_daily").glob("*.md")
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

    subprocess.run(
        [
            sys.executable,
            str(script),
            "train",
            *common,
            "--params",
            str(params_path),
            "--min-train-size",
            "3",
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate",
            *common,
            "--params",
            str(params_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, str(script), "daily", *common, "--params", str(params_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "evaluations" / f"learned_ranker_v4_{lottery}.json").exists()
    daily_matches = list(
        (output_dir / "learned_ranker_v4_daily").glob(
            f"{lottery}_learned_ranker_v4_*_daily_2026016.json"
        )
    )
    assert len(daily_matches) == 1


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
            (output_dir / "learned_ranker_v4_daily").glob("fc3d*_daily_2026020.json")
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
        common_passed=True, direct_passed=True, group_passed=True
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
    feature_config = LearnedFeatureConfig(windows=(5, "all"))
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
