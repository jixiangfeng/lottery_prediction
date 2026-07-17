# -*- coding: utf-8 -*-

import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.digit_backtest import backtest_digit_candidates
from src.analysis.digit_candidates import (
    DigitCandidateConfig,
    generate_digit_candidates,
)
from src.analysis.digit_report import (
    build_digit_report_data,
    build_digit_report_markdown,
    generate_digit_report_from_csv,
)
from src.analysis.digit_statistics import analyze_digit_history
from src.lotteries import get_lottery_rule


def test_build_digit_report_markdown_contains_core_sections():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026002", "百位": 1, "十位": 1, "个位": 2},
            {"期数": "2026001", "百位": 9, "十位": 9, "个位": 9},
        ]
    )
    stats = analyze_digit_history(df, rule)

    markdown = build_digit_report_markdown(stats)

    assert "# 福彩3D 数字彩分析日报" in markdown
    assert "## 最新开奖" in markdown
    assert "## 位置频率 Top" in markdown
    assert "## 当前遗漏 Top" in markdown
    assert "## 多窗口遗漏" in markdown
    assert "## 和值 / 跨度 / 形态" in markdown
    assert "仅做历史统计" in markdown


def test_generate_digit_report_from_csv_writes_report(tmp_path: Path):
    csv_path = tmp_path / "pl5.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text(
        "期号,开奖号码\n2026002,01234\n2026001,99887\n", encoding="utf-8"
    )

    output = generate_digit_report_from_csv("pl5", csv_path, output_dir=output_dir)

    assert output == output_dir / "pl5_daily_2026002.md"
    text = output.read_text(encoding="utf-8")
    assert "# 排列五 数字彩分析日报" in text
    assert "2026002" in text
    assert "0 1 2 3 4" in text
    assert "全不同" in text
    assert "二二一" in text
    assert "## 统计候选" in text
    assert "和值" in text
    assert "## 历史回放迁移说明" in text
    assert "hindsight 回放" in text
    assert "开奖前已保存的 prediction snapshot" in text


def test_build_digit_report_data_is_json_friendly():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026002", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026001", "百位": 1, "十位": 1, "个位": 2},
        ]
    )
    stats = analyze_digit_history(df, rule)
    candidates = generate_digit_candidates(
        stats, rule, config=DigitCandidateConfig(count=3), seed=1
    )
    backtest = backtest_digit_candidates(df, rule, candidates)

    payload = build_digit_report_data(
        stats, candidates, backtest, markdown_path=Path("reports/fc3d_daily_2026002.md")
    )

    assert payload["lottery"]["code"] == "fc3d"
    assert payload["latestIssue"] == "2026002"
    assert payload["latestNumbers"] == [1, 2, 3]
    assert "positionFrequency" in payload
    assert len(payload["candidates"]) == 3
    assert payload["directCandidates"] == payload["candidates"]
    assert "groupCandidates" in payload
    assert payload["backtest"]["candidateCount"] == 3
    assert payload["artifacts"]["markdown"] == "reports/fc3d_daily_2026002.md"
    assert payload["candidates"][0]["modelWeight"] > 0
    assert payload["candidates"][0]["compositeModelWeight"] > 0
    assert payload["candidates"][0]["jointProbabilityDeprecated"] is True
    assert payload["candidateConfig"]["rankingMode"] == "composite"


def test_generate_digit_report_from_csv_writes_json_when_enabled(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text("期号,开奖号码\n2026002,123\n2026001,112\n", encoding="utf-8")

    output = generate_digit_report_from_csv(
        "fc3d", csv_path, output_dir=output_dir, write_json=True, candidate_count=3
    )
    json_path = output_dir / "data" / "fc3d_daily_2026002.json"

    assert output == output_dir / "fc3d_daily_2026002.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["lottery"]["displayName"] == "福彩3D"
    assert payload["schemaVersion"] == 2
    assert payload["latestIssue"] == "2026002"
    assert len(payload["candidates"]) == 3
    assert payload["backtest"]["candidateCount"] == 3
    assert payload["candidateConfig"]["rankingMode"] == "ensemble"
    assert payload["candidates"][0]["modelRankPercentiles"]
    assert "集成投票" in output.read_text(encoding="utf-8")
    assert payload["advancedModels"]["monteCarloEnabled"] is True
    assert payload["modelCandidates"]
    assert "高级模型状态" in output.read_text(encoding="utf-8")
    assert Path(payload["artifacts"]["pickSnapshot"]).exists()


def test_digit_report_evaluates_previous_snapshot_when_new_issue_arrives(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text("期号,开奖号码\n2026002,123\n2026001,456\n", encoding="utf-8")
    generate_digit_report_from_csv(
        "fc3d", csv_path, output_dir=output_dir, candidate_count=5
    )

    csv_path.write_text(
        "期号,开奖号码\n2026003,654\n2026002,123\n2026001,456\n", encoding="utf-8"
    )
    generate_digit_report_from_csv(
        "fc3d", csv_path, output_dir=output_dir, candidate_count=5
    )

    assert (output_dir / "evaluations" / "fc3d_2026003.md").exists()
    assert (output_dir / "evaluations" / "fc3d_live_summary.md").exists()


def test_digit_report_integrates_trained_ml_and_monte_carlo_votes(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    lines = ["期号,开奖号码"]
    for index in range(40):
        lines.append(
            f"{2026001 + index},{index % 5}{(index * 3 + 1) % 10}{(index * 7 + 2) % 10}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=5,
        monte_carlo_simulations=500,
        ml_training_periods=5,
        ml_negative_samples=2,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026040.json").read_text(encoding="utf-8")
    )

    assert payload["advancedModels"]["mlTrained"] is True
    assert payload["advancedModels"]["monteCarloAccepted"] > 0
    assert payload["advancedModels"]["monteCarloPairConditioned"] is True
    assert payload["advancedModels"]["monteCarloStructureConditioned"] is True
    assert all(
        "mlRanker" in candidate["modelRankPercentiles"]
        and "monteCarlo" in candidate["modelRankPercentiles"]
        for candidate in payload["candidates"]
    )
    assert payload["advancedModels"]["activeModelCount"] == 16
    assert payload["advancedModels"]["availableModelCount"] == 16
    assert payload["advancedModels"]["activeModelNames"][-2:] == [
        "monteCarlo",
        "mlRanker",
    ]


def test_digit_report_model_switches_are_reflected_in_active_models(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    lines = ["期号,开奖号码"]
    for index in range(60):
        lines.append(
            f"{2026001 + index},{index % 5}{(index * 3 + 1) % 10}{(index * 7 + 2) % 10}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=5,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026060.json").read_text(encoding="utf-8")
    )

    assert payload["advancedModels"]["activeModelCount"] == 14
    assert "monteCarlo" not in payload["advancedModels"]["activeModelNames"]
    assert "mlRanker" not in payload["advancedModels"]["activeModelNames"]
    assert "实际启用模型（14/16）" in (output_dir / "fc3d_daily_2026060.md").read_text(
        encoding="utf-8"
    )


def test_digit_report_defaults_to_incremental_statistics_and_exposes_baseline(
    tmp_path: Path,
):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text(
        "期号,开奖号码\n2026003,123\n2026002,456\n2026001,789\n",
        encoding="utf-8",
    )

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=3,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    first_payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026003.json").read_text(encoding="utf-8")
    )
    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=3,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    second_payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026003.json").read_text(encoding="utf-8")
    )
    markdown = (output_dir / "fc3d_daily_2026003.md").read_text(encoding="utf-8")

    expected_snapshot = output_dir / "state" / "fc3d_statistics_snapshot.json"
    assert expected_snapshot.exists()
    assert first_payload["statisticsUpdate"]["mode"] == "full_rebuild"
    assert second_payload["statisticsUpdate"]["mode"] == "cache_hit"
    assert second_payload["statisticsUpdate"]["processedRows"] == 0
    assert second_payload["statisticsUpdate"]["snapshotPath"] == str(expected_snapshot)
    assert second_payload["theoreticalProbabilities"]["shape"] == {
        "豹子": 0.01,
        "组三": 0.27,
        "组六": 0.72,
    }
    assert second_payload["theoreticalProbabilities"]["isPrediction"] is False
    assert "数学基线（不是预测）" in markdown
    assert "统计更新：`cache_hit`" in markdown


def test_digit_report_default_disables_hindsight_backtest_and_keeps_json_shape(
    tmp_path: Path, monkeypatch
):
    import src.analysis.digit_report as report_module

    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text("期号,开奖号码\n2026002,123\n2026001,456\n", encoding="utf-8")

    def fail_hindsight(*args, **kwargs):
        raise AssertionError("默认日报不应回放当前候选到全部历史")

    monkeypatch.setattr(report_module, "backtest_digit_candidates", fail_hindsight)
    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=3,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026002.json").read_text(encoding="utf-8")
    )

    assert payload["backtest"]["candidateCount"] == 3
    assert payload["backtest"]["totalChecks"] == 0
    assert payload["hindsightBacktest"]["enabled"] is False
    assert payload["hindsightBacktest"]["migration"]


def test_digit_report_can_explicitly_enable_hindsight_backtest(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text("期号,开奖号码\n2026002,123\n2026001,456\n", encoding="utf-8")

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=3,
        enable_monte_carlo=False,
        enable_ml=False,
        enable_hindsight_backtest=True,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026002.json").read_text(encoding="utf-8")
    )
    markdown = (output_dir / "fc3d_daily_2026002.md").read_text(encoding="utf-8")

    assert payload["hindsightBacktest"]["enabled"] is True
    assert payload["backtest"]["totalChecks"] > 0
    assert "## 数字彩候选回测" in markdown


def test_incremental_and_full_statistics_generate_identical_candidates(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    lines = ["期号,开奖号码"]
    for index in range(45):
        lines.append(
            f"{2026001 + index},{index % 10}{(index * 3) % 10}{(index * 7) % 10}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    incremental_dir = tmp_path / "incremental"
    full_dir = tmp_path / "full"

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=incremental_dir,
        write_json=True,
        candidate_count=6,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=full_dir,
        write_json=True,
        candidate_count=6,
        enable_monte_carlo=False,
        enable_ml=False,
        incremental_stats=False,
    )

    incremental = json.loads(
        (incremental_dir / "data" / "fc3d_daily_2026045.json").read_text(
            encoding="utf-8"
        )
    )
    full = json.loads(
        (full_dir / "data" / "fc3d_daily_2026045.json").read_text(encoding="utf-8")
    )
    assert incremental["candidates"] == full["candidates"]
    assert incremental["modelCandidates"] == full["modelCandidates"]
    assert full["statisticsUpdate"]["rebuildReason"] == "incremental_disabled"


def test_digit_report_cli_accepts_statistics_snapshot_flags(
    tmp_path: Path, monkeypatch
):
    import src.analysis.digit_report as report_module

    captured = {}

    def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "report.md"

    monkeypatch.setattr(report_module, "generate_digit_report_from_csv", fake_generate)

    exit_code = report_module.main(
        [
            "--lottery",
            "fc3d",
            "--csv",
            str(tmp_path / "input.csv"),
            "--stats-snapshot-path",
            str(tmp_path / "state.json"),
            "--rebuild-stats",
            "--no-incremental-stats",
        ]
    )

    assert exit_code == 0
    assert captured["stats_snapshot_path"] == str(tmp_path / "state.json")
    assert captured["rebuild_stats"] is True
    assert captured["incremental_stats"] is False


def test_digit_report_appending_one_issue_uses_incremental_statistics(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    output_dir = tmp_path / "reports"
    csv_path.write_text("期号,开奖号码\n2026002,123\n2026001,456\n", encoding="utf-8")
    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=3,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    csv_path.write_text(
        "期号,开奖号码\n2026003,789\n2026002,123\n2026001,456\n",
        encoding="utf-8",
    )

    generate_digit_report_from_csv(
        "fc3d",
        csv_path,
        output_dir=output_dir,
        write_json=True,
        candidate_count=3,
        enable_monte_carlo=False,
        enable_ml=False,
    )
    payload = json.loads(
        (output_dir / "data" / "fc3d_daily_2026003.json").read_text(encoding="utf-8")
    )

    assert payload["statisticsUpdate"]["mode"] == "incremental"
    assert payload["statisticsUpdate"]["addedIssues"] == 1
    assert payload["statisticsUpdate"]["processedRows"] == 1


def test_atomic_text_write_preserves_previous_file_when_replace_fails(
    tmp_path: Path, monkeypatch
):
    import src.analysis.digit_report as report_module

    path = tmp_path / "report.md"
    path.write_text("旧报告", encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("模拟 replace 失败")

    monkeypatch.setattr(report_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="模拟 replace 失败"):
        report_module._atomic_write_text(path, "新报告")

    assert path.read_text(encoding="utf-8") == "旧报告"
    assert not list(tmp_path.glob(".report.md.*.tmp"))


def test_atomic_text_write_never_exposes_partial_content_to_concurrent_reader(
    tmp_path: Path,
):
    import src.analysis.digit_report as report_module

    path = tmp_path / "report.json"
    old_content = json.dumps({"version": "old", "payload": "a" * 200_000})
    new_content = json.dumps({"version": "new", "payload": "b" * 200_000})
    report_module._atomic_write_text(path, old_content)
    stop = threading.Event()
    observed: list[str] = []

    def read_repeatedly():
        while not stop.is_set():
            observed.append(path.read_text(encoding="utf-8"))

    reader = threading.Thread(target=read_repeatedly)
    reader.start()
    try:
        for _ in range(20):
            report_module._atomic_write_text(path, new_content)
            report_module._atomic_write_text(path, old_content)
    finally:
        stop.set()
        reader.join(timeout=5)

    assert not reader.is_alive()
    assert observed
    assert set(observed) <= {old_content, new_content}
