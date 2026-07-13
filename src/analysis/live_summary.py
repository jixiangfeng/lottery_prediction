# -*- coding: utf-8 -*-
"""快乐8实盘累计表现统计。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.analysis.pick_tracking import PickEvaluation


@dataclass(frozen=True)
class LiveSummary:
    """多期真实推荐复盘的累计统计。"""

    period_count: int
    latest_issue: str
    total_cost: int
    total_prize: int
    roi: float
    average_hit: float
    hit5_plus_count: int
    max_losing_streak: int
    best_parameter: str
    parameter_roi: dict[str, float]
    issue_rows: list[dict[str, object]]


def _hit5_plus(evaluation: PickEvaluation) -> int:
    return sum(1 for item in evaluation.group_results if item.hit_count >= 5)


def compute_live_summary(evaluations: Sequence[PickEvaluation]) -> LiveSummary:
    """汇总多期推荐复盘结果。"""

    if not evaluations:
        raise ValueError("至少需要一条复盘记录")
    sorted_items = sorted(evaluations, key=lambda item: item.target_issue)
    total_cost = sum(item.total_cost for item in sorted_items)
    total_prize = sum(item.total_prize for item in sorted_items)
    total_groups = sum(len(item.group_results) for item in sorted_items)
    total_hits = sum(group.hit_count for item in sorted_items for group in item.group_results)
    hit5_plus_count = sum(_hit5_plus(item) for item in sorted_items)
    roi = round((total_prize - total_cost) / total_cost, 4) if total_cost else 0.0
    average_hit = round(total_hits / total_groups, 4) if total_groups else 0.0

    losing_streak = 0
    max_losing_streak = 0
    by_parameter: dict[str, dict[str, int]] = defaultdict(lambda: {"cost": 0, "prize": 0})
    issue_rows: list[dict[str, object]] = []
    for item in sorted_items:
        if item.total_prize < item.total_cost:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0
        by_parameter[item.parameter_name]["cost"] += item.total_cost
        by_parameter[item.parameter_name]["prize"] += item.total_prize
        issue_rows.append(
            {
                "issue": item.target_issue,
                "parameter": item.parameter_name,
                "cost": item.total_cost,
                "prize": item.total_prize,
                "roi": item.roi,
                "hit5_plus": _hit5_plus(item),
            }
        )

    parameter_roi = {
        parameter: round((values["prize"] - values["cost"]) / values["cost"], 4)
        for parameter, values in by_parameter.items()
        if values["cost"] > 0
    }
    best_parameter = max(parameter_roi, key=parameter_roi.get) if parameter_roi else ""
    return LiveSummary(
        period_count=len(sorted_items),
        latest_issue=sorted_items[-1].target_issue,
        total_cost=total_cost,
        total_prize=total_prize,
        roi=roi,
        average_hit=average_hit,
        hit5_plus_count=hit5_plus_count,
        max_losing_streak=max_losing_streak,
        best_parameter=best_parameter,
        parameter_roi=parameter_roi,
        issue_rows=issue_rows,
    )


def build_live_summary_markdown(summary: LiveSummary) -> str:
    """生成实盘累计表现 Markdown。"""

    lines = [
        "# 快乐8实盘累计表现",
        "",
        f"- 统计期数：`{summary.period_count}`",
        f"- 最近期号：`{summary.latest_issue}`",
        f"- 总投入：`{summary.total_cost}` 元",
        f"- 总返奖：`{summary.total_prize}` 元",
        f"- 累计收益率：`{summary.roi:.2%}`",
        f"- 平均命中：`{summary.average_hit}` 个/注",
        f"- 中5+次数：`{summary.hit5_plus_count}`",
        f"- 最大连续亏损期数：`{summary.max_losing_streak}`",
        f"- 最佳参数：`{summary.best_parameter}`",
        "",
        "## 参数表现",
        "",
        "| 参数 | 累计收益率 |",
        "|---|---:|",
    ]
    for parameter, roi in sorted(summary.parameter_roi.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {parameter} | {roi:.2%} |")
    lines.extend(["", "## 逐期明细", "", "| 期号 | 参数 | 投入 | 返奖 | 收益率 | 中5+ |", "|---|---|---:|---:|---:|---:|"])
    for row in summary.issue_rows:
        lines.append(
            f"| {row['issue']} | {row['parameter']} | {row['cost']} | {row['prize']} | "
            f"{float(row['roi']):.2%} | {row['hit5_plus']} |"
        )
    lines.extend(["", "说明：本汇总仅统计已保存推荐快照且目标期已开奖的真实复盘。", ""])
    return "\n".join(lines)


def write_live_summary(summary: LiveSummary, output_dir: Path | str) -> Path:
    """写入实盘累计表现 Markdown。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "live_summary.md"
    output.write_text(build_live_summary_markdown(summary), encoding="utf-8")
    return output
