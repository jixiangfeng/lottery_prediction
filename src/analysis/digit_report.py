# -*- coding: utf-8 -*-
"""数字彩 Markdown 分析报告生成器。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.analysis.digit_backtest import DigitBacktestSummary, build_digit_backtest_markdown, backtest_digit_candidates
from src.analysis.betting_plan import BettingPlan, build_betting_plan_markdown, build_digit_betting_plan
from src.analysis.digit_candidates import (
    DigitBettingCandidateResult,
    DigitCandidateConfig,
    DigitCandidateResult,
    generate_digit_betting_candidates,
)
from src.analysis.digit_data import load_digit_csv
from src.analysis.digit_statistics import DigitStatisticsResult, analyze_digit_history
from src.lotteries import get_lottery_rule


def _top_items(counter: Counter, limit: int = 5) -> list[tuple[object, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _format_numbers(numbers: list[int]) -> str:
    return " ".join(str(number) for number in numbers)


def build_digit_report_markdown(
    stats: DigitStatisticsResult,
    *,
    top_n: int = 5,
    candidates: DigitCandidateResult | None = None,
    backtest_markdown: str | None = None,
    betting_plan: BettingPlan | None = None,
    candidate_plan: DigitBettingCandidateResult | None = None,
) -> str:
    """根据数字彩统计结果生成 Markdown 报告。"""

    lines = [
        f"# {stats.display_name} 数字彩分析日报",
        "",
        "## 最新开奖",
        "",
        f"- 最新期号：`{stats.latest_issue}`",
        f"- 最新号码：`{_format_numbers(stats.latest_numbers)}`",
        f"- 样本期数：`{stats.total_issues}`",
        "- 说明：本报告仅做历史统计和娱乐参考，不保证中奖。",
        "",
        "## 位置频率 Top",
        "",
    ]
    for position, counter in stats.position_frequency.items():
        top = "，".join(f"{digit}:{count}" for digit, count in _top_items(counter, top_n))
        lines.append(f"- {position}：{top}")

    lines.extend(["", "## 当前遗漏 Top", ""])
    for position, omission in stats.current_omission.items():
        top = sorted(omission.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        text = "，".join(f"{digit}:{miss}" for digit, miss in top)
        lines.append(f"- {position}：{text}")

    lines.extend(
        [
            "",
            "## 和值 / 跨度 / 形态",
            "",
            "### 和值 Top",
            "",
        ]
    )
    for value, count in _top_items(stats.sum_distribution, top_n):
        lines.append(f"- {value}：{count}期")

    lines.extend(["", "### 跨度 Top", ""])
    for value, count in _top_items(stats.span_distribution, top_n):
        lines.append(f"- {value}：{count}期")

    lines.extend(["", "### 形态分布", ""])
    for shape, count in _top_items(stats.shape_distribution, top_n):
        lines.append(f"- {shape}：{count}期")

    lines.extend(["", "## 奇偶 / 大小", "", "### 奇偶分布", ""])
    for label, count in _top_items(stats.parity_distribution, top_n):
        lines.append(f"- {label}：{count}期")

    lines.extend(["", "### 大小分布", ""])
    for label, count in _top_items(stats.big_small_distribution, top_n):
        lines.append(f"- {label}：{count}期")

    if candidates is not None and candidates.candidates:
        lines.extend(["", "## 统计候选", ""])
        lines.append("### 直选候选")
        lines.append("")
        lines.append("以下为按位置精确命中的直选候选；模型评分只表示历史统计排序质量，不是实际开奖概率。")
        lines.append("")
        for index, candidate in enumerate(candidates.candidates, 1):
            lines.append(
                f"{index}. `{candidate.text}` - 和值 {candidate.sum_value}，跨度 {candidate.span}，"
                f"形态 {candidate.shape}，评分 {candidate.score:.4f}"
            )
        if candidate_plan is not None and candidate_plan.group_candidates:
            lines.extend(["", "### 组选候选（复合模型质量聚合）", ""])
            lines.append("组选只比较数字集合，不要求位置顺序；过滤空间归一化模型质量不是实际开奖概率。")
            lines.append("")
            for index, candidate in enumerate(candidate_plan.group_candidates, 1):
                lines.append(
                    f"{index}. `{candidate.group_key}` - 形态 {candidate.shape}，"
                    f"过滤空间归一化模型质量 {candidate.composite_model_weight:.6f}，"
                    f"排列数 {candidate.permutations}（不是实际开奖概率）"
                )

    if betting_plan is not None:
        lines.extend(["", build_betting_plan_markdown(betting_plan).rstrip()])

    if backtest_markdown:
        lines.extend(["", backtest_markdown.rstrip()])

    lines.extend(
        [
            "",
            "## 理性提示",
            "",
            "数字彩开奖结果仍接近随机；位置频率、遗漏、和值、跨度和形态只能描述历史分布，不能保证预测下一期开奖。",
            "",
        ]
    )
    return "\n".join(lines)


def build_digit_report_data(
    stats: DigitStatisticsResult,
    candidates: DigitCandidateResult,
    backtest: DigitBacktestSummary,
    *,
    markdown_path: Path | None = None,
    betting_plan: BettingPlan | None = None,
    candidate_plan: DigitBettingCandidateResult | None = None,
) -> dict:
    """生成前端友好的数字彩 JSON 数据。"""

    return {
        "lottery": {
            "code": stats.code,
            "displayName": stats.display_name,
        },
        "latestIssue": stats.latest_issue,
        "latestNumbers": stats.latest_numbers,
        "totalIssues": stats.total_issues,
        "positionFrequency": {
            position: dict(sorted(counter.items())) for position, counter in stats.position_frequency.items()
        },
        "currentOmission": stats.current_omission,
        "sumDistribution": dict(sorted(stats.sum_distribution.items())),
        "spanDistribution": dict(sorted(stats.span_distribution.items())),
        "shapeDistribution": dict(stats.shape_distribution),
        "parityDistribution": dict(stats.parity_distribution),
        "bigSmallDistribution": dict(stats.big_small_distribution),
        "candidates": [candidate.to_dict() for candidate in candidates.candidates],
        "directCandidates": [candidate.to_dict() for candidate in candidates.candidates],
        "groupCandidates": (
            [candidate.to_dict() for candidate in candidate_plan.group_candidates]
            if candidate_plan is not None
            else []
        ),
        "candidateConfig": candidates.to_dict().get("config"),
        "backtest": backtest.to_dict(),
        "bettingPlan": betting_plan.to_dict() if betting_plan is not None else None,
        "artifacts": {
            "markdown": str(markdown_path) if markdown_path is not None else None,
        },
        "disclaimer": "数字彩开奖结果接近随机；本报告仅做历史统计、候选过滤和回测参考，不保证中奖。",
    }


def generate_digit_report_from_csv(
    lottery: str,
    csv_path: str | Path,
    *,
    output_dir: str | Path = "reports",
    top_n: int = 5,
    candidate_count: int = 10,
    write_json: bool = False,
) -> Path:
    """从数字彩 CSV 生成 Markdown 分析报告。"""

    rule = get_lottery_rule(lottery)
    df = load_digit_csv(csv_path, rule)
    stats = analyze_digit_history(df, rule)
    candidate_plan = generate_digit_betting_candidates(
        stats,
        rule,
        config=DigitCandidateConfig(count=candidate_count),
        group_count=candidate_count,
    )
    candidates = DigitCandidateResult(
        candidate_plan.rule_code,
        candidate_plan.display_name,
        candidate_plan.direct_candidates,
        candidate_plan.config,
    )
    betting_plan = build_digit_betting_plan(candidates)
    backtest = backtest_digit_candidates(df, rule, candidates)
    backtest_markdown = build_digit_backtest_markdown(backtest)
    output_path = Path(output_dir) / f"{rule.code}_daily_{stats.latest_issue}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_digit_report_markdown(
            stats,
            top_n=top_n,
            candidates=candidates,
            backtest_markdown=backtest_markdown,
            betting_plan=betting_plan,
            candidate_plan=candidate_plan,
        ),
        encoding="utf-8",
    )
    if write_json:
        json_path = Path(output_dir) / "data" / f"{rule.code}_daily_{stats.latest_issue}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                build_digit_report_data(
                    stats,
                    candidates,
                    backtest,
                    markdown_path=output_path,
                    betting_plan=betting_plan,
                    candidate_plan=candidate_plan,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成福彩3D/排列三/排列五数字彩分析报告")
    parser.add_argument("--lottery", required=True, choices=["fc3d", "pl3", "pl5"], help="数字彩玩法代码")
    parser.add_argument("--csv", required=True, help="历史开奖 CSV 路径")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    parser.add_argument("--top-n", default=5, type=int, help="Top 项数量")
    parser.add_argument("--candidate-count", default=10, type=int, help="统计候选数量")
    parser.add_argument("--json", action="store_true", help="同时输出 reports/data/<lottery>_daily_<issue>.json")
    args = parser.parse_args(argv)

    output = generate_digit_report_from_csv(
        args.lottery,
        args.csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        candidate_count=args.candidate_count,
        write_json=args.json,
    )
    print(output)
    return 0
