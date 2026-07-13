# -*- coding: utf-8 -*-
"""快乐8每日推荐留痕与开奖后复盘。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.analysis.backtest import KL8_SELECT10_PRIZE_TABLE


@dataclass(frozen=True)
class GroupEvaluation:
    """单组推荐的开奖复盘。"""

    index: int
    numbers: list[int]
    hit_numbers: list[int]
    hit_count: int
    prize: int


@dataclass(frozen=True)
class PickEvaluation:
    """一期推荐快照的复盘汇总。"""

    target_issue: str
    source_issue: str
    parameter_name: str
    draw_numbers: list[int]
    group_results: list[GroupEvaluation]
    total_cost: int
    total_prize: int
    roi: float


def next_issue(issue: str | int) -> str:
    """数字期号 +1。"""

    return str(int(issue) + 1)


def _group_numbers(group: Any) -> list[int]:
    numbers = getattr(group, "numbers", group)
    return sorted(int(number) for number in numbers)


def save_pick_snapshot(
    stats: Any,
    groups: Sequence[Any],
    *,
    parameter_name: str,
    output_dir: Path | str,
    target_issue: str | None = None,
) -> Path:
    """保存当日推荐快照，默认面向下一期。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    source_issue = str(stats.latest_issue)
    issue = target_issue or next_issue(source_issue)
    payload = {
        "lottery": "kl8",
        "play": "select10",
        "source_issue": source_issue,
        "target_issue": issue,
        "parameter_name": parameter_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_numbers": [int(number) for number in stats.latest_numbers],
        "groups": [_group_numbers(group) for group in groups],
    }
    output = directory / f"kl8_{issue}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_pick_snapshot(path: Path | str) -> dict[str, Any]:
    """读取推荐快照 JSON。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number_columns(df: pd.DataFrame) -> list[str]:
    columns = [f"红球_{idx}" for idx in range(1, 21)]
    missing = [column for column in columns if column not in df.columns]
    if "期数" not in df.columns or missing:
        raise ValueError("历史数据必须包含【期数】和【红球_1】到【红球_20】列")
    return columns


def _draw_for_issue(history: pd.DataFrame, issue: str) -> list[int] | None:
    columns = _number_columns(history)
    issue_series = history["期数"].astype(str)
    matches = history[issue_series == str(issue)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return sorted(int(row[column]) for column in columns)


def evaluate_pick_snapshot(
    history: pd.DataFrame,
    snapshot_path: Path | str,
    *,
    ticket_price: int = 2,
    prize_table: dict[int, int] | None = None,
) -> PickEvaluation | None:
    """如果目标期已开奖，则复盘推荐快照；否则返回 None。"""

    payload = load_pick_snapshot(snapshot_path)
    target_issue = str(payload["target_issue"])
    draw_numbers = _draw_for_issue(history, target_issue)
    if draw_numbers is None:
        return None

    draw_set = set(draw_numbers)
    prizes = prize_table or KL8_SELECT10_PRIZE_TABLE
    group_results: list[GroupEvaluation] = []
    for index, group in enumerate(payload["groups"], 1):
        numbers = sorted(int(number) for number in group)
        hit_numbers = sorted(set(numbers) & draw_set)
        prize = int(prizes.get(len(hit_numbers), 0))
        group_results.append(
            GroupEvaluation(
                index=index,
                numbers=numbers,
                hit_numbers=hit_numbers,
                hit_count=len(hit_numbers),
                prize=prize,
            )
        )
    total_cost = len(group_results) * ticket_price
    total_prize = sum(item.prize for item in group_results)
    roi = round((total_prize - total_cost) / total_cost, 4) if total_cost else 0.0
    return PickEvaluation(
        target_issue=target_issue,
        source_issue=str(payload.get("source_issue", "")),
        parameter_name=str(payload.get("parameter_name", "")),
        draw_numbers=draw_numbers,
        group_results=group_results,
        total_cost=total_cost,
        total_prize=total_prize,
        roi=roi,
    )


def find_evaluable_snapshots(history: pd.DataFrame, picks_dir: Path | str) -> list[Path]:
    """找出目标期已开奖的推荐快照。"""

    directory = Path(picks_dir)
    if not directory.exists():
        return []
    available_issues = set(history["期数"].astype(str))
    matches: list[Path] = []
    for path in sorted(directory.glob("kl8_*.json")):
        payload = load_pick_snapshot(path)
        if str(payload.get("target_issue")) in available_issues:
            matches.append(path)
    return matches


def build_evaluation_markdown(evaluation: PickEvaluation) -> str:
    """生成单期复盘 Markdown。"""

    lines = [
        "## 快乐8推荐复盘",
        "",
        f"- 推荐期号：`{evaluation.target_issue}`",
        f"- 推荐参数：`{evaluation.parameter_name}`",
        f"- 开奖号码：`{' '.join(f'{number:02d}' for number in evaluation.draw_numbers)}`",
        f"- 总投入：`{evaluation.total_cost}` 元",
        f"- 总返奖：`{evaluation.total_prize}` 元",
        f"- 本期收益率：`{evaluation.roi:.2%}`",
        "",
        "| 序号 | 推荐号码 | 命中数 | 命中号码 | 返奖 |",
        "|---:|---|---:|---|---:|",
    ]
    for item in evaluation.group_results:
        lines.append(
            f"| {item.index} | `{' '.join(f'{number:02d}' for number in item.numbers)}` | "
            f"{item.hit_count} | `{' '.join(f'{number:02d}' for number in item.hit_numbers)}` | {item.prize} |"
        )
    lines.extend(["", "说明：复盘只记录真实推荐与开奖的对照，不代表未来收益。", ""])
    return "\n".join(lines)


def write_evaluation_markdown(evaluation: PickEvaluation, output_dir: Path | str) -> Path:
    """写入复盘 Markdown。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"kl8_{evaluation.target_issue}.md"
    output.write_text(build_evaluation_markdown(evaluation), encoding="utf-8")
    return output
