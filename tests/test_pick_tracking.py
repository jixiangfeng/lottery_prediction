# -*- coding: utf-8 -*-

import json

import pandas as pd

from src.analysis.daily_report import CandidateGroup, compute_basic_stats
from src.analysis.pick_tracking import (
    build_evaluation_markdown,
    evaluate_pick_snapshot,
    find_evaluable_snapshots,
    next_issue,
    save_pick_snapshot,
)


def _history() -> pd.DataFrame:
    rows = [
        {"期数": 2024002, **{f"红球_{i+1}": n for i, n in enumerate(range(1, 21))}},
        {"期数": 2024001, **{f"红球_{i+1}": n for i, n in enumerate(range(21, 41))}},
    ]
    return pd.DataFrame(rows)


def _groups() -> list[CandidateGroup]:
    return [
        CandidateGroup(numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], score=9.0, odd_count=5, big_count=0, repeat_last_count=0, zone_distribution=[10, 0, 0, 0, 0, 0, 0, 0]),
        CandidateGroup(numbers=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30], score=8.0, odd_count=5, big_count=0, repeat_last_count=0, zone_distribution=[0, 0, 10, 0, 0, 0, 0, 0]),
    ]


def test_next_issue_increments_numeric_issue():
    assert next_issue("2024001") == "2024002"


def test_save_pick_snapshot_writes_json_for_next_issue(tmp_path):
    stats = compute_basic_stats(_history(), windows=(1, 2))

    path = save_pick_snapshot(stats, _groups(), parameter_name="test_param", output_dir=tmp_path)

    assert path.name == "kl8_2024003.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_issue"] == "2024003"
    assert payload["source_issue"] == "2024002"
    assert payload["parameter_name"] == "test_param"
    assert payload["groups"][0] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_evaluate_pick_snapshot_calculates_hits_and_prize(tmp_path):
    stats = compute_basic_stats(_history(), windows=(1, 2))
    path = save_pick_snapshot(stats, _groups(), parameter_name="test_param", output_dir=tmp_path, target_issue="2024002")

    evaluation = evaluate_pick_snapshot(_history(), path)

    assert evaluation is not None
    assert evaluation.target_issue == "2024002"
    assert evaluation.total_cost == 4
    assert evaluation.total_prize == 5_000_002
    assert evaluation.group_results[0].hit_count == 10
    assert evaluation.group_results[1].hit_count == 0

    markdown = build_evaluation_markdown(evaluation)
    assert "## 快乐8推荐复盘" in markdown
    assert "2024002" in markdown
    assert "总返奖" in markdown


def test_find_evaluable_snapshots_only_returns_opened_issues(tmp_path):
    stats = compute_basic_stats(_history(), windows=(1, 2))
    ready = save_pick_snapshot(stats, _groups(), parameter_name="test_param", output_dir=tmp_path, target_issue="2024002")
    save_pick_snapshot(stats, _groups(), parameter_name="test_param", output_dir=tmp_path, target_issue="2024999")

    matches = find_evaluable_snapshots(_history(), tmp_path)

    assert matches == [ready]
