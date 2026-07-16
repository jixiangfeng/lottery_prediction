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
    snapshot = save_digit_pick_snapshot(stats, plan, tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    evaluation = evaluate_digit_pick_snapshot(_history(), rule, snapshot)

    assert payload["sourceIssue"] == "2026002"
    assert payload["schemaVersion"] == 2
    assert payload["rankingMode"] == "ensemble"
    assert payload["ensembleModelWeights"]
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
    assert "累计表现" in summary_markdown.read_text(encoding="utf-8")
    assert summary.model_performance


def test_live_model_performance_derives_conservative_weights():
    evaluations = []
    for index in range(8):
        evaluations.append(
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
                model_candidates={"position": ["123"], "pair": ["456"]},
                model_hits={"position": True, "pair": False},
            )
        )
    base = DigitCandidateConfig().ensemble_model_weights

    adjusted = derive_live_ensemble_weights(evaluations, base, min_samples=5)

    assert len(adjusted) == len(base)
    assert sum(adjusted) == pytest.approx(sum(base))
    assert adjusted[0] > base[0]
    assert adjusted[1] < base[1]
    assert all(
        original * 0.8 <= value <= original * 1.2
        for original, value in zip(base, adjusted)
    )
