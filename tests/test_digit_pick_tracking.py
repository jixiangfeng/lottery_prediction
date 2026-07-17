# -*- coding: utf-8 -*-

import json

import pandas as pd
import pytest

from src.analysis.digit_candidates import (
    DigitCandidateConfig,
    generate_digit_betting_candidates,
)
from src.analysis.digit_pick_tracking import (
    DigitPickEvaluation,
    build_digit_live_summary,
    derive_live_ensemble_weights,
    evaluate_digit_pick_snapshot,
    save_digit_pick_snapshot,
    write_digit_evaluation,
    write_digit_live_summary,
)
from src.analysis.digit_statistics import analyze_digit_history
from src.lotteries import get_lottery_rule


def _history(last_issue: int = 2026003) -> pd.DataFrame:
    rows = [
        {"期数": "2026001", "百位": 1, "十位": 2, "个位": 3},
        {"期数": "2026002", "百位": 4, "十位": 5, "个位": 6},
    ]
    if last_issue >= 2026003:
        rows.append({"期数": "2026003", "百位": 6, "十位": 5, "个位": 4})
    return pd.DataFrame(rows)


def test_digit_pick_snapshot_evaluates_first_later_issue(tmp_path):
    rule = get_lottery_rule("fc3d")
    train = _history(2026002)
    stats = analyze_digit_history(train, rule)
    plan = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )
    snapshot = save_digit_pick_snapshot(stats, plan, tmp_path, immutable=True)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    evaluation = evaluate_digit_pick_snapshot(_history(), rule, snapshot)

    assert payload["sourceIssue"] == "2026002"
    assert payload["schemaVersion"] == 3
    assert payload["rankingMode"] == "ensemble"
    assert payload["ensembleModelWeights"]
    assert payload["candidateConfig"]["count"] == 10
    assert len(payload["recommendationFingerprint"]) == 64
    assert payload["immutable"] is True
    assert set(payload["modelCandidates"]) == set(plan.model_candidates)
    assert evaluation is not None
    assert evaluation.source_issue == "2026002"
    assert evaluation.target_issue == "2026003"
    assert evaluation.actual_text == "654"
    assert evaluation.direct_hit == ("654" in payload["directCandidates"])
    assert evaluation.group_hit == ("456" in payload["groupCandidates"])
    assert evaluation.model_hits == {
        name: "654" in candidates
        for name, candidates in payload["modelCandidates"].items()
    }


def test_digit_pick_snapshot_is_idempotent_but_rejects_changed_recommendation(
    tmp_path,
):
    rule = get_lottery_rule("fc3d")
    stats = analyze_digit_history(_history(2026002), rule)
    original = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )
    snapshot = save_digit_pick_snapshot(stats, original, tmp_path, immutable=True)
    original_text = snapshot.read_text(encoding="utf-8")

    assert (
        save_digit_pick_snapshot(stats, original, tmp_path, immutable=True) == snapshot
    )
    assert snapshot.read_text(encoding="utf-8") == original_text

    changed = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=9, ranking_mode="ensemble"),
        group_count=9,
    )
    with pytest.raises(FileExistsError, match="禁止"):
        save_digit_pick_snapshot(stats, changed, tmp_path)


def test_digit_pick_snapshot_can_freeze_an_identical_daily_snapshot(tmp_path):
    rule = get_lottery_rule("fc3d")
    stats = analyze_digit_history(_history(2026002), rule)
    plan = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )
    snapshot = save_digit_pick_snapshot(stats, plan, tmp_path)
    generated_at = json.loads(snapshot.read_text(encoding="utf-8"))["generatedAt"]

    save_digit_pick_snapshot(stats, plan, tmp_path, immutable=True)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    assert payload["immutable"] is True
    assert payload["generatedAt"] == generated_at


def test_digit_pick_snapshot_allows_overwriting_non_frozen_daily_snapshot(tmp_path):
    rule = get_lottery_rule("fc3d")
    stats = analyze_digit_history(_history(2026002), rule)
    original = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )
    changed = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=9, ranking_mode="ensemble"),
        group_count=9,
    )

    snapshot = save_digit_pick_snapshot(stats, original, tmp_path)
    original_fingerprint = json.loads(snapshot.read_text(encoding="utf-8"))[
        "recommendationFingerprint"
    ]
    save_digit_pick_snapshot(stats, changed, tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    assert payload["immutable"] is False
    assert payload["candidateConfig"]["count"] == 9
    assert payload["recommendationFingerprint"] != original_fingerprint


def test_digit_pick_snapshot_experiment_id_keeps_parallel_strategies_separate(tmp_path):
    rule = get_lottery_rule("fc3d")
    stats = analyze_digit_history(_history(2026002), rule)
    plan = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )

    default_snapshot = save_digit_pick_snapshot(stats, plan, tmp_path)
    experiment_snapshot = save_digit_pick_snapshot(
        stats, plan, tmp_path, experiment_id="probability_v2"
    )
    payload = json.loads(experiment_snapshot.read_text(encoding="utf-8"))

    assert default_snapshot != experiment_snapshot
    assert experiment_snapshot.name == "fc3d_probability_v2_2026002.json"
    assert payload["experimentId"] == "probability_v2"
    with pytest.raises(ValueError, match="experiment_id"):
        save_digit_pick_snapshot(stats, plan, tmp_path, experiment_id="invalid-id")


def test_digit_pick_evaluation_and_live_summary_are_written(tmp_path):
    rule = get_lottery_rule("fc3d")
    stats = analyze_digit_history(_history(2026002), rule)
    plan = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=10, ranking_mode="ensemble"),
        group_count=10,
    )
    snapshot = save_digit_pick_snapshot(stats, plan, tmp_path / "picks")
    evaluation = evaluate_digit_pick_snapshot(_history(), rule, snapshot)
    assert evaluation is not None

    markdown_path, json_path = write_digit_evaluation(
        evaluation, tmp_path / "evaluations"
    )
    summary = build_digit_live_summary([evaluation])
    summary_markdown, summary_json = write_digit_live_summary(
        summary, tmp_path / "evaluations"
    )

    assert markdown_path.exists() and json_path.exists()
    assert "真实推荐复盘" in markdown_path.read_text(encoding="utf-8")
    assert summary.period_count == 1
    assert summary.direct_hits in {0, 1}
    assert summary_markdown.exists() and summary_json.exists()
    summary_text = summary_markdown.read_text(encoding="utf-8")
    assert "累计表现" in summary_text
    assert "随机命中率" in summary_text
    assert "至少 100 期" in summary_text
    assert summary.model_performance
    assert all(
        "randomHitRate" in values and "pValue" in values
        for values in summary.model_performance.values()
    )


def _model_evaluations(periods: int, hits: int) -> list[DigitPickEvaluation]:
    hit_candidates = [
        "123",
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
    ]
    miss_candidates = [
        "000",
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
    ]
    return [
        DigitPickEvaluation(
            rule_code="fc3d",
            display_name="福彩3D",
            source_issue=str(2026001 + index),
            target_issue=str(2026002 + index),
            ranking_mode="ensemble",
            actual_text="123",
            direct_candidates=[],
            group_candidates=[],
            direct_hit=False,
            group_hit=False,
            model_candidates={
                "position": hit_candidates if index < hits else miss_candidates
            },
            model_hits={"position": index < hits},
        )
        for index in range(periods)
    ]


def test_live_weights_stay_at_base_before_one_hundred_samples():
    base = DigitCandidateConfig().ensemble_model_weights

    adjusted = derive_live_ensemble_weights(_model_evaluations(99, 99), base)

    assert adjusted == base


def test_live_weights_do_not_reward_nonsignificant_lift_over_random():
    base = DigitCandidateConfig().ensemble_model_weights

    adjusted = derive_live_ensemble_weights(_model_evaluations(100, 2), base)

    assert adjusted == base


def test_live_weights_reward_only_significant_lift_with_ten_percent_cap():
    base = DigitCandidateConfig().ensemble_model_weights

    adjusted = derive_live_ensemble_weights(_model_evaluations(100, 5), base)

    assert len(adjusted) == len(base)
    assert sum(adjusted) == pytest.approx(sum(base))
    assert adjusted[0] > base[0]
    assert all(
        original * 0.9 <= value <= original * 1.1
        for original, value in zip(base, adjusted)
    )
