# -*- coding: utf-8 -*-

import json
import math

import pandas as pd
import pytest

from src.analysis.digit_probability import (
    DigitProbabilityCalibration,
    DigitProbabilityConfig,
    build_digit_probability_plan,
    fit_digit_probability_calibration,
)
from src.analysis.digit_probability_walk_forward import (
    build_digit_probability_walk_forward_markdown,
    run_digit_probability_walk_forward,
)
from src.analysis.digit_report import generate_digit_report_from_csv
from src.lotteries import get_lottery_rule


def _history(periods: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(periods)
        ]
    )


def _fixed_calibration(weight: float = 1.0) -> DigitProbabilityCalibration:
    return DigitProbabilityCalibration(
        applied_learned_weight=weight,
        selected_learned_weight=weight,
        selected_model="ensemble",
        temperature=0.2,
        validation_periods=90,
        selection_periods=60,
        holdout_periods=30,
        uniform_log_loss=math.log(1000),
        selection_log_loss=0.0,
        holdout_log_loss=0.0,
        block_log_losses=(0.0, 0.0, 0.0),
        passed=True,
        fallback_reason=None,
    )


def test_probability_plan_falls_back_to_exact_uniform_distribution_when_history_short():
    rule = get_lottery_rule("fc3d")
    plan = build_digit_probability_plan(
        _history(40),
        rule,
        candidate_count=10,
        probability_config=DigitProbabilityConfig(
            validation_periods=30,
            min_train_size=20,
            minimum_validation_periods=30,
        ),
    )

    assert plan.config.ranking_mode == "probability"
    assert plan.distribution.calibration.applied_learned_weight == 0.0
    assert plan.distribution.probability_sum == pytest.approx(1.0)
    assert len(plan.distribution.probabilities) == 1000
    assert all(
        value == pytest.approx(0.001)
        for value in plan.distribution.probabilities.values()
    )
    assert len({candidate.text for candidate in plan.direct_candidates}) == 10
    assert all(
        candidate.predicted_probability == pytest.approx(0.001)
        for candidate in plan.direct_candidates
    )
    assert all(candidate.shape == "组六" for candidate in plan.group_candidates)
    assert all(
        candidate.predicted_probability == pytest.approx(0.006)
        for candidate in plan.group_candidates
    )


def test_probability_plan_uses_pure_topk_and_exact_group_permutation_sum():
    rule = get_lottery_rule("fc3d")
    plan = build_digit_probability_plan(
        _history(100),
        rule,
        candidate_count=10,
        calibration=_fixed_calibration(),
    )
    probabilities = plan.distribution.probabilities
    selected = [
        float(candidate.predicted_probability or 0.0)
        for candidate in plan.direct_candidates
    ]

    assert selected == sorted(probabilities.values(), reverse=True)[:10]
    for candidate in plan.group_candidates:
        expected = sum(
            probability
            for text, probability in probabilities.items()
            if "".join(sorted(text)) == candidate.group_key
        )
        assert candidate.predicted_probability == pytest.approx(expected)
        assert candidate.ranking_model == "exact_permutation_probability_v2"


def test_probability_calibration_uses_uniform_fallback_when_validation_is_insufficient():
    rule = get_lottery_rule("fc3d")
    calibration = fit_digit_probability_calibration(
        _history(50),
        rule,
        probability_config=DigitProbabilityConfig(
            validation_periods=30,
            min_train_size=40,
            minimum_validation_periods=20,
        ),
    )

    assert calibration.passed is False
    assert calibration.applied_learned_weight == 0.0
    assert "不足" in str(calibration.fallback_reason)


def test_probability_walk_forward_freezes_calibration_before_evaluation_targets():
    rule = get_lottery_rule("fc3d")
    history = _history(100)
    changed = history.copy()
    changed.loc[changed.index[-5:], ["百位", "十位", "个位"]] = [9, 9, 9]
    options = dict(
        periods=5,
        min_train_size=30,
        candidate_count=10,
        probability_config=DigitProbabilityConfig(
            validation_periods=45,
            min_train_size=30,
            minimum_validation_periods=30,
        ),
    )

    original_report = run_digit_probability_walk_forward(history, rule, **options)
    changed_report = run_digit_probability_walk_forward(changed, rule, **options)

    assert original_report.calibration == changed_report.calibration
    assert original_report.calibration_train_end_issue == "2026095"
    assert all(issue.train_end_issue < issue.issue for issue in original_report.issues)
    assert original_report.development_only is True
    assert "只能用于开发诊断" in build_digit_probability_walk_forward_markdown(
        original_report
    )


def test_probability_daily_report_exposes_calibration_and_actual_probabilities(
    tmp_path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    rows = ["期号,开奖号码"]
    for row in _history(40).itertuples(index=False):
        rows.append(f"{row[0]},{row[1]}{row[2]}{row[3]}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    output = generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=10,
        ranking_mode="probability",
        probability_validation_periods=30,
        probability_min_train_size=20,
        probability_minimum_validation_periods=30,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026040.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (output_dir / "picks" / "digit" / "fc3d_probability_v2_2026040.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["candidateConfig"]["rankingMode"] == "probability"
    assert payload["advancedModels"] is None
    assert payload["probabilityCalibration"]["appliedLearnedWeight"] == 0.0
    assert payload["probabilitySum"] == pytest.approx(1.0)
    assert all(
        candidate["predictedProbability"] > 0 for candidate in payload["candidates"]
    )
    assert snapshot["probabilityModel"] == "uniform_shrunk_rank_softmax_v2"
    assert snapshot["experimentId"] == "probability_v2"
    assert snapshot["probabilityCalibration"]
    assert len(snapshot["probabilityDistributionFingerprint"]) == 64
    assert len(snapshot["directCandidateProbabilities"]) == 10
    assert len(snapshot["groupCandidateProbabilities"]) == 10
    assert "回退均匀分布" in output.read_text(encoding="utf-8")


def test_online_probability_daily_report_persists_state_and_snapshot(tmp_path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    rows = ["期号,开奖号码"]
    for row in _history(25).itertuples(index=False):
        rows.append(f"{row[0]},{row[1]}{row[2]}{row[3]}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    output = generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=10,
        ranking_mode="online_probability",
        online_probability_min_train_size=10,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026025.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (
            output_dir / "picks" / "digit" / "fc3d_probability_online_v3_2026025.json"
        ).read_text(encoding="utf-8")
    )
    original_fingerprint = snapshot["recommendationFingerprint"]
    original_generated_at = snapshot["generatedAt"]

    assert payload["candidateConfig"]["rankingMode"] == "online_probability"
    assert payload["onlineProbability"]["stateUpdate"]["mode"] == "full_rebuild"
    assert payload["onlineProbability"]["state"]["feedbackPeriods"] == 15
    assert payload["probabilitySum"] == pytest.approx(1.0)
    assert payload["advancedModels"] is None
    assert snapshot["experimentId"] == "probability_online_v3"
    assert snapshot["onlineProbability"]["state"]["processedPeriods"] == 25
    assert snapshot["directCandidateProbabilities"]
    assert "逐期开奖反馈" in output.read_text(encoding="utf-8")

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=10,
        ranking_mode="online_probability",
        online_probability_min_train_size=10,
    )
    repeated_payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026025.json").read_text(encoding="utf-8")
    )
    repeated_snapshot = json.loads(
        (
            output_dir / "picks" / "digit" / "fc3d_probability_online_v3_2026025.json"
        ).read_text(encoding="utf-8")
    )
    assert repeated_payload["onlineProbability"]["stateUpdate"]["mode"] == "cache_hit"
    assert repeated_snapshot["recommendationFingerprint"] == original_fingerprint
    assert repeated_snapshot["generatedAt"] == original_generated_at
