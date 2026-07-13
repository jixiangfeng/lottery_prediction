# -*- coding: utf-8 -*-
"""快乐8逐期前推策略回测。

每一期只使用开奖前已存在的历史数据生成候选，再对比目标期开奖。
这比“固定今天候选回放历史”更接近真实使用场景，但仍然只是历史模拟，
不代表未来可预测或可盈利。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.analysis.backtest import KL8_SELECT10_PRIZE_TABLE
from src.analysis.daily_report import compute_basic_stats
from src.analysis.parameter_search import ParameterConfig, default_parameter_grid, generate_parameter_groups


@dataclass(frozen=True)
class WalkForwardIssueResult:
    """单个目标期的前推回测结果。"""

    issue: str
    strategy: str
    groups: list[list[int]]
    draw_numbers: list[int]
    hit_counts: list[int]
    total_cost: int
    total_prize: int
    roi: float

    @property
    def max_hit(self) -> int:
        return max(self.hit_counts, default=0)

    @property
    def hit5_plus_count(self) -> int:
        return sum(1 for value in self.hit_counts if value >= 5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "strategy": self.strategy,
            "groups": self.groups,
            "drawNumbers": self.draw_numbers,
            "hitCounts": self.hit_counts,
            "maxHit": self.max_hit,
            "hit5PlusCount": self.hit5_plus_count,
            "totalCost": self.total_cost,
            "totalPrize": self.total_prize,
            "roi": self.roi,
        }


@dataclass(frozen=True)
class WalkForwardStrategySummary:
    """单个策略多期前推汇总。"""

    strategy: str
    period_count: int
    group_count: int
    total_cost: int
    total_prize: int
    capped_total_prize: int
    roi: float
    capped_roi: float
    average_hit: float
    hit_distribution: Counter[int]
    hit5_plus_count: int
    hit6_plus_count: int
    issue_hit5_plus_count: int
    max_losing_streak: int
    best_issue: str
    best_issue_max_hit: int
    recent_roi: float
    score: float
    issue_results: list[WalkForwardIssueResult]

    def to_dict(self, *, include_issues: bool = True) -> dict[str, Any]:
        payload = {
            "strategy": self.strategy,
            "periodCount": self.period_count,
            "groupCount": self.group_count,
            "totalCost": self.total_cost,
            "totalPrize": self.total_prize,
            "cappedTotalPrize": self.capped_total_prize,
            "roi": self.roi,
            "cappedRoi": self.capped_roi,
            "averageHit": self.average_hit,
            "hitDistribution": {str(key): int(value) for key, value in sorted(self.hit_distribution.items())},
            "hit5PlusCount": self.hit5_plus_count,
            "hit6PlusCount": self.hit6_plus_count,
            "issueHit5PlusCount": self.issue_hit5_plus_count,
            "maxLosingStreak": self.max_losing_streak,
            "bestIssue": self.best_issue,
            "bestIssueMaxHit": self.best_issue_max_hit,
            "recentRoi": self.recent_roi,
            "score": self.score,
        }
        if include_issues:
            payload["issues"] = [item.to_dict() for item in self.issue_results]
        return payload


@dataclass(frozen=True)
class WalkForwardBacktestReport:
    """多策略前推回测报告。"""

    lottery: str
    play: str
    period_count: int
    min_train_size: int
    group_count: int
    ticket_price: int
    best_strategy: str
    summaries: list[WalkForwardStrategySummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "lottery": self.lottery,
            "play": self.play,
            "periodCount": self.period_count,
            "minTrainSize": self.min_train_size,
            "groupCount": self.group_count,
            "ticketPrice": self.ticket_price,
            "bestStrategy": self.best_strategy,
            "summaries": [summary.to_dict(include_issues=True) for summary in self.summaries],
            "disclaimer": "逐期前推回测是历史模拟，不保证未来中奖或盈利。",
        }


def _number_columns(history: pd.DataFrame) -> list[str]:
    columns = [f"红球_{idx}" for idx in range(1, 21)]
    if "期数" not in history.columns or any(column not in history.columns for column in columns):
        raise ValueError("历史数据必须包含【期数】和【红球_1】到【红球_20】列")
    return columns


def _sorted_history(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df["期数"] = df["期数"].astype(str)
    return df.sort_values("期数", ascending=False).reset_index(drop=True)


def _row_numbers(row: pd.Series, columns: Sequence[str]) -> list[int]:
    return sorted(int(row[column]) for column in columns)


def _evaluate_groups(
    issue: str,
    strategy: str,
    groups: Sequence[Sequence[int]],
    draw_numbers: Sequence[int],
    *,
    ticket_price: int,
    prize_table: dict[int, int],
) -> WalkForwardIssueResult:
    draw_set = set(int(number) for number in draw_numbers)
    normalized_groups = [sorted(int(number) for number in group) for group in groups]
    hit_counts = [len(set(group) & draw_set) for group in normalized_groups]
    total_cost = len(normalized_groups) * ticket_price
    total_prize = sum(int(prize_table.get(hit, 0)) for hit in hit_counts)
    roi = round((total_prize - total_cost) / total_cost, 4) if total_cost else 0.0
    return WalkForwardIssueResult(
        issue=issue,
        strategy=strategy,
        groups=normalized_groups,
        draw_numbers=sorted(draw_set),
        hit_counts=hit_counts,
        total_cost=total_cost,
        total_prize=total_prize,
        roi=roi,
    )


def _summarize_strategy(strategy: str, issue_results: Sequence[WalkForwardIssueResult], *, group_count: int) -> WalkForwardStrategySummary:
    if not issue_results:
        raise ValueError("issue_results 不能为空")
    total_cost = sum(item.total_cost for item in issue_results)
    total_prize = sum(item.total_prize for item in issue_results)
    capped_total_prize = sum(min(KL8_SELECT10_PRIZE_TABLE.get(hit, 0), 800) for item in issue_results for hit in item.hit_counts)
    hit_distribution: Counter[int] = Counter()
    total_hits = 0
    total_bets = 0
    hit5_plus_count = 0
    hit6_plus_count = 0
    issue_hit5_plus_count = 0
    losing_streak = 0
    max_losing_streak = 0
    best = issue_results[0]
    for item in issue_results:
        if item.total_prize > 0:
            losing_streak = 0
        else:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        if item.max_hit > best.max_hit:
            best = item
        if item.hit5_plus_count > 0:
            issue_hit5_plus_count += 1
        for hit in item.hit_counts:
            hit_distribution[hit] += 1
            total_hits += hit
            total_bets += 1
            if hit >= 5:
                hit5_plus_count += 1
            if hit >= 6:
                hit6_plus_count += 1
    roi = round((total_prize - total_cost) / total_cost, 4) if total_cost else 0.0
    capped_roi = round((capped_total_prize - total_cost) / total_cost, 4) if total_cost else 0.0
    average_hit = round(total_hits / total_bets, 4) if total_bets else 0.0
    recent_items = list(issue_results)[: min(50, len(issue_results))]
    recent_cost = sum(item.total_cost for item in recent_items)
    recent_prize = sum(item.total_prize for item in recent_items)
    recent_roi = round((recent_prize - recent_cost) / recent_cost, 4) if recent_cost else 0.0
    score = round(capped_roi + hit5_plus_count / max(total_bets, 1) + 0.1 * recent_roi - 0.01 * max_losing_streak, 6)
    return WalkForwardStrategySummary(
        strategy=strategy,
        period_count=len(issue_results),
        group_count=group_count,
        total_cost=total_cost,
        total_prize=total_prize,
        capped_total_prize=capped_total_prize,
        roi=roi,
        capped_roi=capped_roi,
        average_hit=average_hit,
        hit_distribution=hit_distribution,
        hit5_plus_count=hit5_plus_count,
        hit6_plus_count=hit6_plus_count,
        issue_hit5_plus_count=issue_hit5_plus_count,
        max_losing_streak=max_losing_streak,
        best_issue=best.issue,
        best_issue_max_hit=best.max_hit,
        recent_roi=recent_roi,
        score=score,
        issue_results=list(issue_results),
    )


def run_walk_forward_strategy_backtest(
    history: pd.DataFrame,
    *,
    configs: Sequence[ParameterConfig] | None = None,
    periods: int = 300,
    min_train_size: int = 200,
    group_count: int = 10,
    group_size: int = 10,
    ticket_price: int = 2,
    prize_table: dict[int, int] | None = None,
) -> WalkForwardBacktestReport:
    """逐期前推回测多个参数策略。"""

    if periods <= 0:
        raise ValueError("periods 必须为正整数")
    if min_train_size <= 0:
        raise ValueError("min_train_size 必须为正整数")
    sorted_df = _sorted_history(history)
    columns = _number_columns(sorted_df)
    max_periods = max(0, len(sorted_df) - min_train_size)
    actual_periods = min(periods, max_periods)
    if actual_periods <= 0:
        raise ValueError("历史数据不足以进行逐期前推回测")
    grid = list(configs) if configs is not None else default_parameter_grid()
    prizes = prize_table or KL8_SELECT10_PRIZE_TABLE
    results_by_strategy: dict[str, list[WalkForwardIssueResult]] = {config.name: [] for config in grid}

    for target_index in range(actual_periods):
        target_row = sorted_df.iloc[target_index]
        train_df = sorted_df.iloc[target_index + 1 :].copy()
        if len(train_df) < min_train_size:
            continue
        stats = compute_basic_stats(train_df)
        issue = str(target_row["期数"])
        draw_numbers = _row_numbers(target_row, columns)
        for config_index, config in enumerate(grid):
            seed = int(issue) * 1009 + config_index * 9173
            groups = generate_parameter_groups(stats, config, count=group_count, group_size=group_size, seed=seed)
            results_by_strategy[config.name].append(
                _evaluate_groups(
                    issue,
                    config.name,
                    groups,
                    draw_numbers,
                    ticket_price=ticket_price,
                    prize_table=prizes,
                )
            )

    summaries = sorted(
        (_summarize_strategy(strategy, items, group_count=group_count) for strategy, items in results_by_strategy.items() if items),
        key=lambda item: item.score,
        reverse=True,
    )
    if not summaries:
        raise ValueError("没有生成任何前推回测结果")
    return WalkForwardBacktestReport(
        lottery="kl8",
        play="select10",
        period_count=actual_periods,
        min_train_size=min_train_size,
        group_count=group_count,
        ticket_price=ticket_price,
        best_strategy=summaries[0].strategy,
        summaries=summaries,
    )


def build_walk_forward_strategy_markdown(report: WalkForwardBacktestReport, *, top_issues: int = 8) -> str:
    """生成前推回测 Markdown 报告。"""

    lines = [
        "# 快乐8逐期前推策略回测",
        "",
        f"- 回测期数：`{report.period_count}`",
        f"- 最小训练期数：`{report.min_train_size}`",
        f"- 每期候选组数：`{report.group_count}`",
        f"- 最稳策略：`{report.best_strategy}`",
        "- 说明：每期只使用开奖前历史数据生成候选；这是历史模拟，不保证未来中奖。",
        "- 封顶ROI：单注返奖按最高 800 元封顶，用于降低偶发中8/中9对策略排序的干扰。",
        "",
        "## 策略汇总",
        "",
        "| 排名 | 策略 | 综合分 | ROI | 封顶ROI | 最近ROI | 平均命中 | 中5+注数 | 中6+注数 | 中5+期数 | 最大连亏 | 最好期号/命中 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, summary in enumerate(report.summaries, 1):
        lines.append(
            f"| {rank} | `{summary.strategy}` | {summary.score:.4f} | {summary.roi:.2%} | {summary.capped_roi:.2%} | {summary.recent_roi:.2%} | "
            f"{summary.average_hit:.3f} | {summary.hit5_plus_count} | {summary.hit6_plus_count} | "
            f"{summary.issue_hit5_plus_count} | {summary.max_losing_streak} | `{summary.best_issue}` / 中{summary.best_issue_max_hit} |"
        )
    lines.extend(["", "## 最优策略近期明细", ""])
    best = report.summaries[0]
    lines.extend(["| 期号 | ROI | 最高命中 | 中5+组数 | 命中分布 |", "|---|---:|---:|---:|---|"])
    for item in best.issue_results[:top_issues]:
        distribution = Counter(item.hit_counts)
        dist_text = " ".join(f"中{hit}:{distribution[hit]}" for hit in sorted(distribution, reverse=True))
        lines.append(f"| `{item.issue}` | {item.roi:.2%} | 中{item.max_hit} | {item.hit5_plus_count} | {dist_text} |")
    lines.extend(["", "## 理性提示", "", "前推回测用于比较策略稳定性和过拟合风险，不代表未来开奖规律。", ""])
    return "\n".join(lines)


def write_walk_forward_strategy_reports(report: WalkForwardBacktestReport, output_dir: Path | str) -> tuple[Path, Path]:
    """写入 Markdown 与 JSON 前推回测报告。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "walk_forward_kl8.md"
    json_dir = directory / "data"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "walk_forward_kl8.json"
    markdown_path.write_text(build_walk_forward_strategy_markdown(report), encoding="utf-8")
    import json

    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path
