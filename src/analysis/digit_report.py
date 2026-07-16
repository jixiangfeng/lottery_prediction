# -*- coding: utf-8 -*-
"""数字彩 Markdown 分析报告生成器。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.analysis.digit_backtest import (
    DigitBacktestSummary,
    build_digit_backtest_markdown,
    backtest_digit_candidates,
)
from src.analysis.digit_advanced_models import (
    DigitAdvancedModelDiagnostics,
    build_advanced_model_scores,
)
from src.analysis.betting_plan import (
    BettingPlan,
    build_betting_plan_markdown,
    build_digit_betting_plan,
)
from src.analysis.digit_candidates import (
    DigitBettingCandidateResult,
    DigitCandidateConfig,
    DigitCandidateResult,
    generate_digit_betting_candidates,
    with_all_history_window,
)
from src.analysis.digit_data import load_digit_csv
from src.analysis.digit_pick_tracking import (
    derive_live_ensemble_weights,
    process_digit_pick_evaluations,
    save_digit_pick_snapshot,
)
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
    pick_snapshot_path: Path | None = None,
    evaluated_pick_count: int = 0,
    advanced_model_diagnostics: DigitAdvancedModelDiagnostics | None = None,
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
        top = "，".join(
            f"{digit}:{count}" for digit, count in _top_items(counter, top_n)
        )
        lines.append(f"- {position}：{top}")

    lines.extend(["", "## 当前遗漏 Top", ""])
    for position, omission in stats.current_omission.items():
        top = sorted(omission.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        text = "，".join(f"{digit}:{miss}" for digit, miss in top)
        lines.append(f"- {position}：{text}")

    lines.extend(["", "## 多窗口遗漏", ""])
    for window, positions in sorted(stats.omission_windows.items()):
        values = []
        for position, omission in positions.items():
            top = sorted(omission.items(), key=lambda item: (-item[1], item[0]))[:3]
            values.append(
                f"{position} " + "/".join(f"{digit}:{miss}" for digit, miss in top)
            )
        lines.append(f"- {window} 期：" + "；".join(values))

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
        if candidates.config.ranking_mode == "ensemble":
            lines.append(
                "以下候选使用 14 个统计子模型加蒙特卡洛、ML 共 16 个投票位的排名分位做集成投票；集成分和票数只表示历史排序，不是实际开奖概率。"
            )
        else:
            lines.append(
                "以下为按位置精确命中的直选候选；模型评分只表示历史统计排序质量，不是实际开奖概率。"
            )
        lines.append("")
        for index, candidate in enumerate(candidates.candidates, 1):
            if candidates.config.ranking_mode == "ensemble":
                lines.append(
                    f"{index}. `{candidate.text}` - 和值 {candidate.sum_value}，跨度 {candidate.span}，"
                    f"形态 {candidate.shape}，集成分 {candidate.ensemble_score:.4f}，"
                    f"Top10% 票数 {candidate.top_decile_votes}/{len(candidate.model_rank_percentiles)}"
                )
            else:
                lines.append(
                    f"{index}. `{candidate.text}` - 和值 {candidate.sum_value}，跨度 {candidate.span}，"
                    f"形态 {candidate.shape}，评分 {candidate.score:.4f}"
                )
        if candidate_plan is not None and candidate_plan.group_candidates:
            if candidates.config.ranking_mode == "ensemble":
                lines.extend(["", "### 组选候选（形态内独立集成排名）", ""])
                lines.append(
                    "组三与组六分别对无序数字集合重新计算模型分位，不直接复用直选排名；该分数不是开奖概率。"
                )
            else:
                lines.extend(["", "### 组选候选（复合模型质量聚合）", ""])
                lines.append(
                    "组选只比较数字集合，不要求位置顺序；过滤空间归一化模型质量不是实际开奖概率。"
                )
            lines.append("")
            for index, candidate in enumerate(candidate_plan.group_candidates, 1):
                if candidates.config.ranking_mode == "ensemble":
                    lines.append(
                        f"{index}. `{candidate.group_key}` - 形态 {candidate.shape}，"
                        f"形态内集成分 {candidate.ensemble_score:.4f}，"
                        f"排列数 {candidate.permutations}（不是实际开奖概率）"
                    )
                else:
                    lines.append(
                        f"{index}. `{candidate.group_key}` - 形态 {candidate.shape}，"
                        f"过滤空间归一化模型质量 {candidate.composite_model_weight:.6f}，"
                        f"排列数 {candidate.permutations}（不是实际开奖概率）"
                    )

    if betting_plan is not None:
        lines.extend(["", build_betting_plan_markdown(betting_plan).rstrip()])

    if backtest_markdown:
        lines.extend(["", backtest_markdown.rstrip()])

    if advanced_model_diagnostics is not None:
        diagnostics = advanced_model_diagnostics
        lines.extend(
            [
                "",
                "## 高级模型状态",
                "",
                f"- 蒙特卡洛：`{'启用' if diagnostics.monte_carlo_enabled else '关闭'}`，"
                f"模拟 `{diagnostics.monte_carlo_simulations}` 次，过滤后接受 `{diagnostics.monte_carlo_accepted}` 次",
                f"- 联合蒙特卡洛：位置对条件 `{'开启' if diagnostics.monte_carlo_pair_conditioned else '关闭'}`，"
                f"形态分布接受 `{'开启' if diagnostics.monte_carlo_structure_conditioned else '关闭'}`",
                f"- 轻量机器学习排序：`{'已训练' if diagnostics.ml_trained else '未训练'}`，"
                f"逐期训练目标 `{diagnostics.ml_training_targets}`，训练样本 `{diagnostics.ml_training_samples}`",
                "- 两类分数只参与候选排序，不解释为实际开奖概率。",
            ]
        )

    if pick_snapshot_path is not None:
        lines.extend(
            [
                "",
                "## 推荐留痕",
                "",
                f"- 本期推荐快照：`{pick_snapshot_path}`",
                f"- 已自动复盘历史快照：`{evaluated_pick_count}` 期",
                "- 逐模型表现满 5 期后会以基础权重为中心保守调权，单模型浮动不超过 20%。",
                "- 新快照将在源期之后的第一期开奖数据出现时自动复盘。",
            ]
        )

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
    pick_snapshot_path: Path | None = None,
    live_summary_path: Path | None = None,
    advanced_model_diagnostics: DigitAdvancedModelDiagnostics | None = None,
) -> dict:
    """生成前端友好的数字彩 JSON 数据。"""

    return {
        "schemaVersion": 2,
        "lottery": {
            "code": stats.code,
            "displayName": stats.display_name,
        },
        "latestIssue": stats.latest_issue,
        "latestNumbers": stats.latest_numbers,
        "totalIssues": stats.total_issues,
        "positionFrequency": {
            position: dict(sorted(counter.items()))
            for position, counter in stats.position_frequency.items()
        },
        "currentOmission": stats.current_omission,
        "omissionWindows": {
            str(window): positions
            for window, positions in stats.omission_windows.items()
        },
        "sumDistribution": dict(sorted(stats.sum_distribution.items())),
        "spanDistribution": dict(sorted(stats.span_distribution.items())),
        "shapeDistribution": dict(stats.shape_distribution),
        "parityDistribution": dict(stats.parity_distribution),
        "bigSmallDistribution": dict(stats.big_small_distribution),
        "candidates": [candidate.to_dict() for candidate in candidates.candidates],
        "modelCandidates": candidates.model_candidates,
        "directCandidates": [
            candidate.to_dict() for candidate in candidates.candidates
        ],
        "groupCandidates": (
            [candidate.to_dict() for candidate in candidate_plan.group_candidates]
            if candidate_plan is not None
            else []
        ),
        "candidateConfig": candidates.to_dict().get("config"),
        "advancedModels": (
            advanced_model_diagnostics.to_dict()
            if advanced_model_diagnostics is not None
            else None
        ),
        "backtest": backtest.to_dict(),
        "bettingPlan": betting_plan.to_dict() if betting_plan is not None else None,
        "artifacts": {
            "markdown": str(markdown_path) if markdown_path is not None else None,
            "pickSnapshot": (
                str(pick_snapshot_path) if pick_snapshot_path is not None else None
            ),
            "liveSummary": (
                str(live_summary_path) if live_summary_path is not None else None
            ),
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
    ranking_mode: str = "ensemble",
    enable_monte_carlo: bool = True,
    monte_carlo_simulations: int = 20_000,
    enable_ml: bool = True,
    ml_training_periods: int = 60,
    ml_negative_samples: int = 9,
    constraint_mode: str = "soft",
    constraint_probability_floor: float = 0.02,
    constraint_penalty_weight: float = 0.05,
) -> Path:
    """从数字彩 CSV 生成 Markdown 分析报告。"""

    rule = get_lottery_rule(lottery)
    df = load_digit_csv(csv_path, rule)
    report_dir = Path(output_dir)
    evaluations, live_summary_path = process_digit_pick_evaluations(
        df,
        rule,
        report_dir / "picks" / "digit",
        report_dir / "evaluations",
    )
    base_config = with_all_history_window(
        DigitCandidateConfig(
            count=candidate_count,
            ranking_mode=ranking_mode,
            constraint_mode=constraint_mode,
            constraint_probability_floor=constraint_probability_floor,
            constraint_penalty_weight=constraint_penalty_weight,
        ),
        len(df),
    )
    candidate_config = replace(
        base_config,
        ensemble_model_weights=derive_live_ensemble_weights(
            evaluations, base_config.ensemble_model_weights
        ),
    )
    stats = analyze_digit_history(
        df,
        rule,
        frequency_windows=candidate_config.frequency_windows,
    )
    external_scores, advanced_model_diagnostics = build_advanced_model_scores(
        df,
        stats,
        rule,
        candidate_config,
        enable_monte_carlo=enable_monte_carlo,
        monte_carlo_simulations=monte_carlo_simulations,
        enable_ml=enable_ml,
        ml_training_periods=ml_training_periods,
        ml_negative_samples=ml_negative_samples,
        seed=int(stats.latest_issue),
    )
    candidate_plan = generate_digit_betting_candidates(
        stats,
        rule,
        config=candidate_config,
        group_count=candidate_count,
        external_scores=external_scores,
    )
    candidates = DigitCandidateResult(
        candidate_plan.rule_code,
        candidate_plan.display_name,
        candidate_plan.direct_candidates,
        candidate_plan.config,
        candidate_plan.model_candidates,
    )
    betting_plan = build_digit_betting_plan(candidates)
    backtest = backtest_digit_candidates(df, rule, candidates)
    backtest_markdown = build_digit_backtest_markdown(backtest)
    pick_snapshot_path = save_digit_pick_snapshot(
        stats,
        candidate_plan,
        report_dir / "picks" / "digit",
    )
    output_path = report_dir / f"{rule.code}_daily_{stats.latest_issue}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_digit_report_markdown(
            stats,
            top_n=top_n,
            candidates=candidates,
            backtest_markdown=backtest_markdown,
            betting_plan=betting_plan,
            candidate_plan=candidate_plan,
            pick_snapshot_path=pick_snapshot_path,
            evaluated_pick_count=len(evaluations),
            advanced_model_diagnostics=advanced_model_diagnostics,
        ),
        encoding="utf-8",
    )
    if write_json:
        json_path = (
            Path(output_dir) / "data" / f"{rule.code}_daily_{stats.latest_issue}.json"
        )
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
                    pick_snapshot_path=pick_snapshot_path,
                    live_summary_path=live_summary_path,
                    advanced_model_diagnostics=advanced_model_diagnostics,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成福彩3D/排列三/排列五数字彩分析报告"
    )
    parser.add_argument(
        "--lottery",
        required=True,
        choices=["fc3d", "pl3", "pl5"],
        help="数字彩玩法代码",
    )
    parser.add_argument(
        "--no-monte-carlo", action="store_true", help="关闭蒙特卡洛投票器"
    )
    parser.add_argument(
        "--monte-carlo-simulations", type=int, default=20_000, help="蒙特卡洛模拟次数"
    )
    parser.add_argument("--no-ml", action="store_true", help="关闭 sklearn 轻量排序器")
    parser.add_argument(
        "--ml-training-periods", type=int, default=60, help="机器学习逐期训练目标数"
    )
    parser.add_argument(
        "--ml-negative-samples", type=int, default=9, help="每个正样本对应的负样本数"
    )
    parser.add_argument("--csv", required=True, help="历史开奖 CSV 路径")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    parser.add_argument("--top-n", default=5, type=int, help="Top 项数量")
    parser.add_argument("--candidate-count", default=10, type=int, help="统计候选数量")
    parser.add_argument(
        "--json",
        action="store_true",
        help="同时输出 reports/data/<lottery>_daily_<issue>.json",
    )
    parser.add_argument(
        "--ranking-mode",
        default="ensemble",
        choices=("ensemble", "composite"),
        help="候选排序模式，默认使用多模型集成投票",
    )
    parser.add_argument(
        "--constraint-mode",
        default="soft",
        choices=("off", "soft", "hard"),
        help="结构罕见度约束模式",
    )
    parser.add_argument(
        "--constraint-probability-floor",
        type=float,
        default=0.02,
        help="奇偶/大小/质合结构多窗口概率下限",
    )
    parser.add_argument(
        "--constraint-penalty-weight",
        type=float,
        default=0.05,
        help="soft 模式罕见结构惩罚权重",
    )
    args = parser.parse_args(argv)

    output = generate_digit_report_from_csv(
        args.lottery,
        args.csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        candidate_count=args.candidate_count,
        write_json=args.json,
        ranking_mode=args.ranking_mode,
        enable_monte_carlo=not args.no_monte_carlo,
        monte_carlo_simulations=args.monte_carlo_simulations,
        enable_ml=not args.no_ml,
        ml_training_periods=args.ml_training_periods,
        ml_negative_samples=args.ml_negative_samples,
        constraint_mode=args.constraint_mode,
        constraint_probability_floor=args.constraint_probability_floor,
        constraint_penalty_weight=args.constraint_penalty_weight,
    )
    print(output)
    return 0
