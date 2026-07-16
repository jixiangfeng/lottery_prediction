# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd

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
    assert "## 数字彩候选回测" in text
    assert "直选命中" in text


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
