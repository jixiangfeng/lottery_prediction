# -*- coding: utf-8 -*-
"""快乐8每日统计报告生成器。

该模块聚焦“可读报告”，不承诺预测准确性。它基于历史开奖 CSV 计算基础统计，
生成结构约束较均衡的选十候选组，并导出 Markdown 报告。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.analysis.backtest import BacktestSummary, build_backtest_markdown, run_fixed_candidate_backtest
from src.analysis.betting_plan import BettingPlan, build_betting_plan_markdown, build_kl8_betting_plan
from src.analysis.strategy_compare import (
    StrategyComparisonResult,
    build_strategy_comparison_markdown,
    compare_strategies,
)
from src.analysis.sliding_window import (
    SlidingStrategySummary,
    build_sliding_window_markdown,
    run_sliding_window_comparison,
)
from src.analysis.parameter_search import (
    ParameterSearchResult,
    build_parameter_search_markdown,
    generate_parameter_groups,
    search_parameter_grid,
)
from src.analysis.pick_tracking import (
    evaluate_pick_snapshot,
    find_evaluable_snapshots,
    save_pick_snapshot,
    write_evaluation_markdown,
)
from src.analysis.live_summary import compute_live_summary, write_live_summary
from src.analysis.html_report import write_html_report
from src.analysis.report_data import write_report_data
from src.analysis.data_quality import DataQualityResult, check_kl8_data_quality, write_data_quality_reports
from src.analysis.candidate_coverage import (
    CandidateCoverageResult,
    analyze_candidate_coverage,
    build_candidate_coverage_markdown,
)
from src.analysis.candidate_portfolio_score import (
    CandidatePortfolioScore,
    build_portfolio_score_markdown,
    score_candidate_portfolio,
)
from src.analysis.candidate_batch_optimizer import CandidateBatchOptimizationResult, select_best_candidate_batch
from src.analysis.strategy_mode import select_parameter_result
from src.analysis.data_source_status import build_data_source_status, read_download_meta, write_data_source_status
from src.analysis.live_parameter_weights import apply_live_parameter_weights
from src.analysis.review_feedback import ReviewFeedback, apply_review_feedback
from src.analysis.walk_forward_parameter_weights import apply_walk_forward_parameter_weights
from src.analysis.walk_forward_validation import (
    WalkForwardValidation,
    build_walk_forward_markdown,
    validate_parameter_walk_forward,
)
from src.config import data_file_name, name_path

NUMBER_RANGE = range(1, 81)


@dataclass(frozen=True)
class Kl8Stats:
    """快乐8基础统计结果。"""

    latest_issue: str
    latest_numbers: list[int]
    frequency_by_window: dict[int, Counter[int]]
    current_omission: dict[int, int]
    hot_numbers: list[int]
    cold_numbers: list[int]
    zone_distribution: list[int]
    tail_distribution: list[int]


@dataclass(frozen=True)
class CandidateGroup:
    """一组候选号码及其结构特征。"""

    numbers: list[int]
    score: float
    odd_count: int
    big_count: int
    repeat_last_count: int
    zone_distribution: list[int]


def _number_columns(df: pd.DataFrame) -> list[str]:
    columns = [f"红球_{idx}" for idx in range(1, 21)]
    missing = [column for column in columns if column not in df.columns]
    if "期数" not in df.columns or missing:
        raise ValueError("历史数据必须包含【期数】和【红球_1】到【红球_20】列")
    return columns


def _row_numbers(row: pd.Series, columns: Sequence[str]) -> list[int]:
    return sorted(int(row[column]) for column in columns)


def _format_numbers(numbers: Iterable[int]) -> str:
    return " ".join(f"{int(number):02d}" for number in numbers)


def _zone_distribution(numbers: Iterable[int]) -> list[int]:
    buckets = [0] * 8
    for number in numbers:
        buckets[(int(number) - 1) // 10] += 1
    return buckets


def _tail_distribution(numbers: Iterable[int]) -> list[int]:
    buckets = [0] * 10
    for number in numbers:
        buckets[int(number) % 10] += 1
    return buckets


def compute_basic_stats(df: pd.DataFrame, windows: Sequence[int] = (30, 100, 300)) -> Kl8Stats:
    """计算热冷号、遗漏、区间/尾数等基础统计。"""

    if df.empty:
        raise ValueError("历史数据为空，无法生成报告")
    columns = _number_columns(df)
    sorted_df = df.copy()
    sorted_df["期数"] = sorted_df["期数"].astype(str)
    sorted_df = sorted_df.sort_values("期数", ascending=False).reset_index(drop=True)

    latest_row = sorted_df.iloc[0]
    latest_numbers = _row_numbers(latest_row, columns)
    frequency_by_window: dict[int, Counter[int]] = {}
    for window in windows:
        counter: Counter[int] = Counter()
        for _, row in sorted_df.head(window).iterrows():
            counter.update(_row_numbers(row, columns))
        frequency_by_window[int(window)] = counter

    current_omission: dict[int, int] = {}
    for number in NUMBER_RANGE:
        miss = 0
        for _, row in sorted_df.iterrows():
            if number in _row_numbers(row, columns):
                break
            miss += 1
        current_omission[number] = miss

    primary_window = int(windows[0]) if windows else min(30, len(sorted_df))
    primary_frequency = frequency_by_window[primary_window]
    hot_numbers = [number for number, _ in primary_frequency.most_common(20)]
    cold_numbers = sorted(NUMBER_RANGE, key=lambda n: (primary_frequency[n], -current_omission[n], n))[:20]

    recent_numbers: list[int] = []
    for _, row in sorted_df.head(primary_window).iterrows():
        recent_numbers.extend(_row_numbers(row, columns))

    return Kl8Stats(
        latest_issue=str(latest_row["期数"]),
        latest_numbers=latest_numbers,
        frequency_by_window=frequency_by_window,
        current_omission=current_omission,
        hot_numbers=hot_numbers,
        cold_numbers=cold_numbers,
        zone_distribution=_zone_distribution(recent_numbers),
        tail_distribution=_tail_distribution(recent_numbers),
    )


def generate_candidate_groups(
    stats: Kl8Stats,
    count: int = 10,
    group_size: int = 10,
    seed: int | None = None,
) -> list[CandidateGroup]:
    """基于基础统计生成结构均衡的候选组。"""

    rng = random.Random(seed if seed is not None else int(stats.latest_issue))
    freq_counter = stats.frequency_by_window[min(stats.frequency_by_window)]
    max_freq = max(freq_counter.values(), default=1)
    max_omit = max(stats.current_omission.values(), default=1)
    latest_set = set(stats.latest_numbers)

    weights = {
        number: 0.55 * (freq_counter[number] / max_freq)
        + 0.25 * (min(stats.current_omission[number], max_omit) / max_omit)
        + 0.20 * (1.0 if number in stats.hot_numbers[:30] else 0.2)
        for number in NUMBER_RANGE
    }

    def score_group(numbers: list[int]) -> float:
        odd_count = sum(number % 2 for number in numbers)
        big_count = sum(number >= 41 for number in numbers)
        repeat_count = len(set(numbers) & latest_set)
        zones = _zone_distribution(numbers)
        balance_penalty = 0.08 * abs(odd_count - group_size / 2) + 0.08 * abs(big_count - group_size / 2)
        repeat_penalty = 0.04 * abs(repeat_count - 2)
        zone_penalty = 0.05 * sum(max(0, value - 2) for value in zones)
        return round(sum(weights[number] for number in numbers) - balance_penalty - repeat_penalty - zone_penalty, 4)

    groups: list[CandidateGroup] = []
    seen: set[tuple[int, ...]] = set()
    population = list(NUMBER_RANGE)
    weighted_population = [max(weights[number], 0.01) ** 2 for number in population]
    attempts = 0
    while len(groups) < count and attempts < count * 2000:
        attempts += 1
        selected: set[int] = set()
        while len(selected) < group_size:
            selected.add(rng.choices(population, weights=weighted_population, k=1)[0])
        numbers = sorted(selected)
        odd_count = sum(number % 2 for number in numbers)
        big_count = sum(number >= 41 for number in numbers)
        repeat_count = len(set(numbers) & latest_set)
        zones = _zone_distribution(numbers)
        tails = _tail_distribution(numbers)
        key = tuple(numbers)
        if key in seen:
            continue
        if not (3 <= odd_count <= 7 and 3 <= big_count <= 7 and repeat_count <= 5):
            continue
        if max(zones) > 3 or max(tails) > 3:
            continue
        if any(len(set(numbers) & set(group.numbers)) > 6 for group in groups):
            continue
        seen.add(key)
        groups.append(
            CandidateGroup(
                numbers=numbers,
                score=score_group(numbers),
                odd_count=odd_count,
                big_count=big_count,
                repeat_last_count=repeat_count,
                zone_distribution=zones,
            )
        )

    return sorted(groups, key=lambda group: group.score, reverse=True)


def _candidate_from_numbers(stats: Kl8Stats, numbers: Sequence[int]) -> CandidateGroup:
    """将已有号码组转换为带结构评分的 CandidateGroup。"""

    group = sorted(int(number) for number in numbers)
    freq_counter = stats.frequency_by_window[min(stats.frequency_by_window)]
    max_freq = max(freq_counter.values(), default=1)
    max_omit = max(stats.current_omission.values(), default=1)
    latest_set = set(stats.latest_numbers)
    weights = {
        number: 0.55 * (freq_counter[number] / max_freq)
        + 0.25 * (min(stats.current_omission[number], max_omit) / max_omit)
        + 0.20 * (1.0 if number in stats.hot_numbers[:30] else 0.2)
        for number in NUMBER_RANGE
    }
    odd_count = sum(number % 2 for number in group)
    big_count = sum(number >= 41 for number in group)
    repeat_count = len(set(group) & latest_set)
    zones = _zone_distribution(group)
    balance_penalty = 0.08 * abs(odd_count - len(group) / 2) + 0.08 * abs(big_count - len(group) / 2)
    repeat_penalty = 0.04 * abs(repeat_count - 2)
    zone_penalty = 0.05 * sum(max(0, value - 2) for value in zones)
    score = round(sum(weights[number] for number in group) - balance_penalty - repeat_penalty - zone_penalty, 4)
    return CandidateGroup(
        numbers=group,
        score=score,
        odd_count=odd_count,
        big_count=big_count,
        repeat_last_count=repeat_count,
        zone_distribution=zones,
    )


def select_best_parameter_candidates(
    stats: Kl8Stats,
    parameter_search_results: Sequence[ParameterSearchResult],
) -> tuple[list[CandidateGroup], str]:
    """选择参数搜索第一名作为日报主推荐候选组。"""

    if not parameter_search_results:
        raise ValueError("parameter_search_results 不能为空")
    best = parameter_search_results[0]
    return [_candidate_from_numbers(stats, group) for group in best.groups], best.config.name


def build_markdown_report(
    stats: Kl8Stats,
    groups: Sequence[CandidateGroup],
    title: str = "快乐8每日分析报告",
    backtest_summary: BacktestSummary | None = None,
    strategy_comparison: dict[str, StrategyComparisonResult] | None = None,
    sliding_window_summary: dict[str, SlidingStrategySummary] | None = None,
    parameter_search_results: Sequence[ParameterSearchResult] | None = None,
    parameter_name: str | None = None,
    data_quality: DataQualityResult | None = None,
    candidate_coverage: CandidateCoverageResult | None = None,
    candidate_portfolio_score: CandidatePortfolioScore | None = None,
    candidate_batch_optimization: CandidateBatchOptimizationResult | None = None,
    walk_forward_validation: WalkForwardValidation | None = None,
    strategy_mode: str | None = None,
    live_parameter_weights: dict | None = None,
    walk_forward_parameter_weights: dict | None = None,
    review_feedback: ReviewFeedback | None = None,
    betting_plan: BettingPlan | None = None,
) -> str:
    """生成 Markdown 报告文本。"""

    primary_window = min(stats.frequency_by_window)
    primary_frequency = stats.frequency_by_window[primary_window]
    hot = sorted(NUMBER_RANGE, key=lambda n: (-primary_frequency[n], n))[:20]
    cold = sorted(NUMBER_RANGE, key=lambda n: (primary_frequency[n], -stats.current_omission[n], n))[:20]
    omission_top = sorted(NUMBER_RANGE, key=lambda n: (-stats.current_omission[n], n))[:15]

    lines = [
        f"# {title}",
        "",
        f"- 最新期号：`{stats.latest_issue}`",
        f"- 最新开奖号码：`{_format_numbers(stats.latest_numbers)}`",
        f"- 数据质量：`{'正常' if data_quality is None or data_quality.ok else '异常'}`",
        f"- 策略模式：`{strategy_mode or 'auto'}`",
        f"- 实盘加权：`{'启用' if live_parameter_weights and live_parameter_weights.get('enabled') else '未启用'}`",
        f"- 前推加权：`{'启用' if walk_forward_parameter_weights and walk_forward_parameter_weights.get('enabled') else '未启用'}`"
        + (f"（最稳策略：{walk_forward_parameter_weights.get('bestStrategy')}）" if walk_forward_parameter_weights and walk_forward_parameter_weights.get('enabled') else ""),
        f"- 复盘反馈：`{'启用' if review_feedback and review_feedback.enabled else '未启用'}`"
        + (f"（{review_feedback.reason}）" if review_feedback else ""),
        f"- 二次优化：`{candidate_batch_optimization.trial_count if candidate_batch_optimization else 1}批候选`",
        "- 说明：本报告仅做历史数据统计和娱乐参考，不保证中奖。",
        "",
        "## 热号 / 冷号 / 遗漏",
        "",
        f"- 最近{primary_window}期热号：`{_format_numbers(hot)}`",
        f"- 最近{primary_window}期冷号：`{_format_numbers(cold)}`",
        "- 当前高遗漏：" + "；".join(f"{num:02d}({stats.current_omission[num]}期)" for num in omission_top),
        "",
        "## 分布概览",
        "",
        "- 最近窗口区间分布 01-10 ~ 71-80：`" + "-".join(str(v) for v in stats.zone_distribution) + "`",
        "- 最近窗口尾数分布 0尾~9尾：`" + "-".join(str(v) for v in stats.tail_distribution) + "`",
        "",
        "## 候选组选十",
        "",
        f"- 主推荐参数：`{parameter_name}`" if parameter_name else "- 主推荐参数：`default_hybrid`",
        "",
        "| 序号 | 号码 | 评分 | 奇偶 | 大小 | 上期重号 | 区间分布 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for idx, group in enumerate(groups, 1):
        lines.append(
            f"| {idx} | `{_format_numbers(group.numbers)}` | {group.score:.3f} | "
            f"{group.odd_count}:{len(group.numbers) - group.odd_count} | "
            f"{group.big_count}:{len(group.numbers) - group.big_count} | "
            f"{group.repeat_last_count} | {'-'.join(str(v) for v in group.zone_distribution)} |"
        )
    lines.append("")
    if candidate_coverage is not None:
        lines.append(build_candidate_coverage_markdown(candidate_coverage).rstrip())
        lines.append("")
    if candidate_portfolio_score is not None:
        lines.append(build_portfolio_score_markdown(candidate_portfolio_score).rstrip())
        lines.append("")
    if betting_plan is not None:
        lines.append(build_betting_plan_markdown(betting_plan).rstrip())
        lines.append("")
    if backtest_summary is not None:
        lines.append(build_backtest_markdown(backtest_summary).rstrip())
        lines.append("")
    if strategy_comparison is not None:
        lines.append(build_strategy_comparison_markdown(strategy_comparison).rstrip())
        lines.append("")
    if sliding_window_summary is not None:
        lines.append(build_sliding_window_markdown(sliding_window_summary).rstrip())
        lines.append("")
    if parameter_search_results is not None:
        lines.append(build_parameter_search_markdown(parameter_search_results).rstrip())
        lines.append("")
    if walk_forward_validation is not None:
        lines.append(build_walk_forward_markdown(walk_forward_validation).rstrip())
        lines.append("")
    lines.extend(
        [
            "",
            "## 回测提示",
            "",
            "- 回测是把当前候选组固定后回放历史，不代表未来会重复。",
            "- 使用时优先比较多期命中分布，不建议只看单期结果。",
            "- 小额娱乐，避免倍投和追损。",
            "",
        ]
    )
    return "\n".join(lines)


def write_daily_report(markdown: str, output_dir: Path | str, issue: str) -> Path:
    """将 Markdown 报告写入 reports 目录。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"kl8_daily_{issue}.md"
    output.write_text(markdown, encoding="utf-8")
    return output


def load_history_csv(path: Path | str | None = None) -> pd.DataFrame:
    """加载本地快乐8 CSV。"""

    csv_path = Path(path) if path else Path(name_path["kl8"]["path"]) / data_file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到快乐8历史数据：{csv_path}，请先运行 scripts/get_data.py")
    return pd.read_csv(csv_path)


def generate_report_from_csv(
    data_path: Path | str | None = None,
    output_dir: Path | str = "reports",
    candidate_count: int = 10,
    group_size: int = 10,
    seed: int | None = None,
    mode: str = "auto",
    strategy: str | None = None,
    batch_trials: int = 30,
) -> Path:
    """从 CSV 生成并写入每日报告。"""

    df = load_history_csv(data_path)
    report_dir = Path(output_dir)
    meta_path = Path(name_path["kl8"]["path"]) / "download_meta.json"
    data_source = build_data_source_status(read_download_meta(meta_path))
    write_data_source_status(data_source, report_dir)
    data_quality = check_kl8_data_quality(df)
    write_data_quality_reports(data_quality, report_dir)
    stats = compute_basic_stats(df)
    parameter_search_results = search_parameter_grid(
        df,
        stats,
        count=candidate_count,
        group_size=group_size,
        seed=seed,
        window_size=100,
        step=100,
        max_windows=6,
    )
    parameter_search_results, live_weight_meta = apply_live_parameter_weights(
        parameter_search_results,
        report_dir / "live_summary.md",
    )
    parameter_search_results, walk_forward_weight_meta = apply_walk_forward_parameter_weights(
        parameter_search_results,
        report_dir / "data" / "walk_forward_kl8.json",
    )
    historical_evaluations = []
    for snapshot_path in find_evaluable_snapshots(df, report_dir / "picks"):
        evaluation = evaluate_pick_snapshot(df, snapshot_path)
        if evaluation is not None:
            historical_evaluations.append(evaluation)
    walk_forward_validation = validate_parameter_walk_forward(parameter_search_results)
    selected_result, resolved_mode = select_parameter_result(
        parameter_search_results,
        mode=mode,
        strategy=strategy,
        live_summary_path=report_dir / "live_summary.md",
    )
    selected_result, resolved_mode, review_feedback = apply_review_feedback(
        selected_result,
        parameter_search_results,
        mode=resolved_mode,
        history=df,
        evaluations=historical_evaluations,
    )
    def _batch_seed(trial_index: int) -> int:
        base = seed if seed is not None else int(str(stats.latest_issue))
        return base * 1000 + trial_index

    raw_batches = [selected_result.groups]
    for trial_index in range(2, max(1, batch_trials) + 1):
        raw_batches.append(
            generate_parameter_groups(
                stats,
                selected_result.config,
                count=candidate_count,
                group_size=group_size,
                seed=_batch_seed(trial_index),
            )
        )
    candidate_batches = [[_candidate_from_numbers(stats, group) for group in batch] for batch in raw_batches]

    def _score_batch(batch: Sequence[CandidateGroup]) -> CandidatePortfolioScore:
        return score_candidate_portfolio(batch, analyze_candidate_coverage(batch))

    candidate_batch_optimization = select_best_candidate_batch(candidate_batches, _score_batch)
    groups = list(candidate_batch_optimization.groups)
    parameter_name = selected_result.config.name
    candidate_coverage = analyze_candidate_coverage(groups)
    candidate_portfolio_score = candidate_batch_optimization.best_score
    betting_plan = build_kl8_betting_plan(groups)
    backtest_summary = run_fixed_candidate_backtest(df, [group.numbers for group in groups], window=100)
    strategy_comparison = compare_strategies(
        df,
        stats,
        groups,
        count=candidate_count,
        group_size=group_size,
        seed=seed,
        window=100,
    )
    sliding_window_summary = run_sliding_window_comparison(
        df,
        {strategy: result.groups for strategy, result in strategy_comparison.items()},
        window_size=100,
        step=100,
        max_windows=6,
    )
    markdown = build_markdown_report(
        stats,
        groups,
        backtest_summary=backtest_summary,
        strategy_comparison=strategy_comparison,
        sliding_window_summary=sliding_window_summary,
        parameter_search_results=parameter_search_results,
        parameter_name=parameter_name,
        data_quality=data_quality,
        candidate_coverage=candidate_coverage,
        candidate_portfolio_score=candidate_portfolio_score,
        candidate_batch_optimization=candidate_batch_optimization,
        walk_forward_validation=walk_forward_validation,
        strategy_mode=resolved_mode,
        live_parameter_weights=live_weight_meta,
        walk_forward_parameter_weights=walk_forward_weight_meta,
        review_feedback=review_feedback,
        betting_plan=betting_plan,
    )
    report_path = write_daily_report(markdown, output_dir, stats.latest_issue)
    html_path = write_html_report(markdown, output_dir, stats.latest_issue)
    pick_snapshot_path = save_pick_snapshot(stats, groups, parameter_name=parameter_name, output_dir=report_dir / "picks")
    write_report_data(
        stats=stats,
        groups=groups,
        parameter_name=parameter_name,
        backtest_summary=backtest_summary,
        strategy_comparison=strategy_comparison,
        sliding_window_summary=sliding_window_summary,
        parameter_search_results=parameter_search_results,
        output_dir=output_dir,
        pick_snapshot_path=pick_snapshot_path,
        html_path=html_path,
        data_quality=data_quality,
        candidate_coverage=candidate_coverage,
        candidate_portfolio_score=candidate_portfolio_score,
        candidate_batch_optimization=candidate_batch_optimization,
        walk_forward_validation=walk_forward_validation,
        strategy_mode=resolved_mode,
        data_source=data_source,
        live_parameter_weights=live_weight_meta,
        walk_forward_parameter_weights=walk_forward_weight_meta,
        review_feedback=review_feedback,
        betting_plan=betting_plan,
    )
    evaluations_by_issue = {evaluation.target_issue: evaluation for evaluation in historical_evaluations}
    for snapshot_path in find_evaluable_snapshots(df, report_dir / "picks"):
        evaluation = evaluate_pick_snapshot(df, snapshot_path)
        if evaluation is not None:
            evaluations_by_issue[evaluation.target_issue] = evaluation
            write_evaluation_markdown(evaluation, report_dir / "evaluations")
    evaluations = list(evaluations_by_issue.values())
    if evaluations:
        write_live_summary(compute_live_summary(evaluations), report_dir)
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(description="生成快乐8每日 Markdown 统计报告")
    parser.add_argument("--data", default=None, help="历史开奖 CSV 路径，默认 data/kl8/data.csv")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    parser.add_argument("--count", default=10, type=int, help="候选组数量")
    parser.add_argument("--group-size", default=10, type=int, help="每组号码数量，默认选十")
    parser.add_argument("--seed", default=None, type=int, help="随机种子，默认使用最新期号")
    parser.add_argument("--mode", default="auto", choices=["auto", "manual", "stable"], help="主推荐策略模式")
    parser.add_argument("--strategy", default=None, help="manual 模式下固定使用的参数名")
    parser.add_argument("--batch-trials", default=30, type=int, help="候选批次二次优化尝试次数")
    args = parser.parse_args(argv)

    output = generate_report_from_csv(
        data_path=args.data,
        output_dir=args.output_dir,
        candidate_count=args.count,
        group_size=args.group_size,
        seed=args.seed,
        mode=args.mode,
        strategy=args.strategy,
        batch_trials=args.batch_trials,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
