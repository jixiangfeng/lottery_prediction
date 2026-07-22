# -*- coding: utf-8 -*-
"""learned_ranker_v4 CLI 与日报集成测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.digit_learned_ranker import main
from src.analysis.digit_learned_features import LearnedFeatureConfig
from src.analysis.digit_learned_ranker import (
    LearnedRankerParams,
    file_sha256,
    generate_learned_ranker_daily,
    learned_ranker_source_fingerprint,
    save_params,
)
from src.analysis.digit_learned_ranker_search import LearnedSplit


def _write_csv(path: Path, periods: int = 20) -> None:
    lines = ["期号,开奖号码"]
    for index in range(periods):
        lines.append(
            f"{2026001 + index},{index % 10}{(index * 3 + 1) % 10}{(index * 7 + 2) % 10}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _daily_json_path(output_dir: Path, issue: str = "2026020") -> Path:
    matches = sorted((output_dir / "daily" / "fc3d").glob(f"fc3d*_daily_{issue}.json"))
    assert len(matches) == 1
    return matches[0]


def _snapshot_path(output_dir: Path, issue: str = "2026020") -> Path:
    matches = sorted(
        (output_dir / "picks" / "digit").glob(f"fc3d_learned_ranker_v4_*_{issue}.json")
    )
    assert len(matches) == 1
    return matches[0]


def test_train_cli_records_explicit_frozen_window_and_objective_profile(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)

    assert (
        main(
            [
                "train",
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--frozen-test-periods",
                "10",
                "--random-trials",
                "4",
                "--local-trials",
                "0",
                "--evaluation-stride",
                "3",
                "--objective-profile",
                "direct_focus",
            ]
        )
        == 2
    )

    params_path = output_dir / "state" / "learned_ranker_v4" / "fc3d_params.json"
    assert not params_path.exists()
    search_path = output_dir / "evaluations" / "learned_ranker_v4_search_fc3d.json"
    metadata = json.loads(search_path.read_text(encoding="utf-8"))
    assert metadata["split"] == {"searchEnd": 5, "validationEnd": 10, "testEnd": 20}
    assert metadata["objectiveProfile"] == "direct_focus"
    assert metadata["search"]["objectiveProfile"] == "direct_focus"
    evaluated = metadata["search"]["searchSpaceManifest"]["sampling"][
        "evaluatedFeatureConfigs"
    ]
    assert len(evaluated) == 16
    assert len({json.dumps(item, sort_keys=True) for item in evaluated}) == 16


def test_train_cli_all_hit_profiles_do_not_write_unconfirmed_params(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path, periods=20)

    assert (
        main(
            [
                "train",
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--frozen-test-periods",
                "10",
                "--objective-profile",
                "all_hit_only",
                "--smoke",
            ]
        )
        == 2
    )

    state_dir = output_dir / "state" / "learned_ranker_v4"
    for profile in ("direct_hit_only", "group_hit_only", "pool_hit_only"):
        params_path = state_dir / f"fc3d_{profile}_params.json"
        assert not params_path.exists()
        search_path = (
            output_dir / "evaluations" / f"learned_ranker_v4_search_fc3d_{profile}.json"
        )
        metadata = json.loads(search_path.read_text(encoding="utf-8"))
        assert metadata["objectiveProfile"] == profile
        assert metadata["search"]["objectiveProfile"] == profile
        curves = metadata["search"]["budgetCurves"]
        assert set(curves) == {"search", "validation"}
        assert set(curves["validation"]) == {"direct", "group", "position"}
        assert curves["validation"]["direct"]["10"]["randomBaseline"] == 0.01
        assert curves["validation"]["position"]["5"]["randomBaseline"] == 0.5
        blocks = curves["validation"]["direct"]["10"]["timeBlocks"]
        assert len(blocks) == 3
        assert all(
            {"hits", "observations", "hitRate", "randomBaseline", "lift"} <= set(block)
            for block in blocks
        )


def test_daily_cli_writes_reproducible_fingerprinted_artifacts(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    params_path = output_dir / "state" / "learned_ranker_v4" / "fc3d_params.json"
    _write_csv(csv_path)
    params = LearnedRankerParams()
    save_params(params, params_path)
    saved_fingerprint = json.loads(params_path.read_text(encoding="utf-8"))[
        "paramsFingerprint"
    ]
    evaluation_path = output_dir / "evaluations" / "learned_ranker_v4_fc3d.json"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(
            {
                "ruleCode": "fc3d",
                "paramsFingerprint": saved_fingerprint,
                "gate": {"passed": False},
                "periods": [
                    {"direct_hit": True, "group_hit": False, "actual_rank": 8},
                    {"direct_hit": False, "group_hit": True, "actual_rank": 200},
                ],
            }
        ),
        encoding="utf-8",
    )

    args = [
        "daily",
        "--lottery",
        "fc3d",
        "--csv",
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--params",
        str(params_path),
    ]
    assert main(args) == 0
    snapshot = _snapshot_path(output_dir)
    first_snapshot = snapshot.read_bytes()
    assert main(args) == 0
    assert snapshot.read_bytes() == first_snapshot

    daily_json = _daily_json_path(output_dir)
    payload = json.loads(daily_json.read_text(encoding="utf-8"))
    assert payload["csvSha256"]
    assert payload["sourceFingerprint"]
    assert payload["paramsFingerprint"]
    assert payload["mode"] == "研究模式，不接入主推荐"
    assert len(payload["plan"]["directCandidates"]) == 10
    assert payload["recentEvaluation"] == {}


def test_uniform_daily_has_no_research_or_formal_ranking(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    _write_csv(csv_path)
    save_params(LearnedRankerParams(uniform_shrinkage=0.0), params_path)

    _, daily_path, _ = generate_learned_ranker_daily(
        "fc3d", csv_path, params_path, output_dir=output_dir
    )
    payload = json.loads(daily_path.read_text(encoding="utf-8"))

    assert payload["abstained"] is True
    assert payload["plan"]["directCandidates"] == []
    assert payload["plan"]["groupCandidates"] == []
    assert payload["plan"]["positionPools"] == []
    assert payload["plan"]["groupDigitPool"] == []
    assert payload["plan"]["mainRecommendation"]["directCandidates"] == []
    assert payload["plan"]["research"]["directCandidates"] == []


def test_daily_does_not_promote_mismatched_frozen_evaluation(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    params = LearnedRankerParams()
    params_path = tmp_path / "params.json"
    _write_csv(csv_path)
    save_params(params, params_path)
    saved_fingerprint = json.loads(params_path.read_text(encoding="utf-8"))[
        "paramsFingerprint"
    ]
    evaluation_path = tmp_path / "mismatched_evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "ruleCode": "pl3",
                "paramsFingerprint": saved_fingerprint,
                "gate": {"passed": True},
                "periods": [],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "daily",
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--params",
                str(params_path),
                "--evaluation",
                str(evaluation_path),
            ]
        )
        == 0
    )

    payload = json.loads((_daily_json_path(output_dir)).read_text(encoding="utf-8"))
    assert payload["gatePassed"] is False
    assert payload["mode"] == "研究模式，不接入主推荐"
    assert payload["evaluationValidation"] == {
        "exists": True,
        "readable": True,
        "ruleMatched": False,
        "paramsMatched": True,
        "paramsArtifactMatched": False,
        "sourceMatched": False,
        "canonicalMatched": False,
        "fingerprintValid": False,
        "frozenTestMatched": False,
        "promoted": False,
    }


def test_daily_rejects_forged_gate_without_frozen_evaluation_fingerprint(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    evaluation_path = tmp_path / "forged.json"
    _write_csv(csv_path)
    save_params(LearnedRankerParams(), params_path)
    model_fingerprint = json.loads(params_path.read_text(encoding="utf-8"))[
        "paramsFingerprint"
    ]
    evaluation_path.write_text(
        json.dumps(
            {
                "ruleCode": "fc3d",
                "paramsFingerprint": model_fingerprint,
                "gate": {"passed": True},
                "periods": [{"actual_rank": 1, "direct_hit": True, "group_hit": True}],
            }
        ),
        encoding="utf-8",
    )

    generate_learned_ranker_daily(
        "fc3d",
        csv_path,
        params_path,
        output_dir=output_dir,
        evaluation_path=evaluation_path,
    )
    payload = json.loads((_daily_json_path(output_dir)).read_text(encoding="utf-8"))

    assert payload["gatePassed"] is False
    assert payload["evaluationValidation"]["fingerprintValid"] is False
    assert payload["recentEvaluation"] == {}


def test_daily_degrades_to_research_mode_when_evaluation_is_corrupted(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "reports"
    evaluation_path = tmp_path / "broken.json"
    _write_csv(csv_path)
    save_params(LearnedRankerParams(), params_path)
    evaluation_path.write_text("{broken", encoding="utf-8")

    generate_learned_ranker_daily(
        "fc3d",
        csv_path,
        params_path,
        output_dir=output_dir,
        evaluation_path=evaluation_path,
    )
    payload = json.loads((_daily_json_path(output_dir)).read_text(encoding="utf-8"))

    assert payload["gatePassed"] is False
    assert payload["mode"] == "研究模式，不接入主推荐"
    assert payload["evaluationValidation"]["readable"] is False


def test_evaluate_uses_frozen_training_split_and_daily_accepts_only_matching_report(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    params_path = tmp_path / "params.json"
    _write_csv(csv_path)
    split = LearnedSplit(search_end=10, validation_end=14, test_end=20)
    feature_config = LearnedFeatureConfig(windows=(5, "all"))
    save_params(
        LearnedRankerParams(),
        params_path,
        metadata={
            "csvSha256": file_sha256(csv_path),
            "sourceFingerprint": learned_ranker_source_fingerprint(),
            "featureConfig": {
                "windows": list(feature_config.windows),
                "alpha": feature_config.alpha,
                "halfLife": feature_config.half_life,
                "omissionCap": feature_config.omission_cap,
            },
            "split": split.to_dict(),
            "testSegmentUsedForSelection": False,
            "validationPassed": True,
        },
    )

    assert (
        main(
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
    evaluation_path = output_dir / "evaluations" / "learned_ranker_v4_fc3d.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["testTargetIndices"] == list(range(14, 20))
    assert evaluation["testSegmentUsedForSelection"] is False
    assert evaluation["reportFingerprint"]

    generate_learned_ranker_daily(
        "fc3d",
        csv_path,
        params_path,
        output_dir=output_dir,
        evaluation_path=evaluation_path,
    )
    daily = json.loads((_daily_json_path(output_dir)).read_text(encoding="utf-8"))
    assert daily["evaluationValidation"]["fingerprintValid"] is True
    assert daily["evaluationValidation"]["frozenTestMatched"] is True
    assert daily["recentEvaluation"]["50"]["periods"] == 6


def test_evaluate_rejects_params_without_validation_confirmation(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    _write_csv(csv_path)
    save_params(
        LearnedRankerParams(),
        params_path,
        metadata={
            "split": LearnedSplit(
                search_end=10, validation_end=14, test_end=20
            ).to_dict(),
            "testSegmentUsedForSelection": False,
            "validationPassed": False,
            "validationReasons": ["Validation Top50命中率未优于随机基线"],
        },
    )

    with pytest.raises(ValueError, match="未通过Validation确认"):
        main(
            [
                "evaluate",
                "--lottery",
                "fc3d",
                "--csv",
                str(csv_path),
                "--params",
                str(params_path),
            ]
        )


def test_different_params_are_isolated_without_overwriting_daily_artifacts(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    first_params = tmp_path / "first.json"
    second_params = tmp_path / "second.json"
    _write_csv(csv_path)
    save_params(LearnedRankerParams(), first_params)
    save_params(replace(LearnedRankerParams(), temperature=0.5), second_params)
    _, first_daily_json, first_snapshot = generate_learned_ranker_daily(
        "fc3d", csv_path, first_params, output_dir=output_dir
    )
    original = first_daily_json.read_bytes()

    _, second_daily_json, second_snapshot = generate_learned_ranker_daily(
        "fc3d", csv_path, second_params, output_dir=output_dir
    )

    assert first_daily_json != second_daily_json
    assert first_snapshot != second_snapshot
    assert first_daily_json.read_bytes() == original
    assert first_daily_json.exists() and second_daily_json.exists()
    assert first_snapshot.exists() and second_snapshot.exists()
