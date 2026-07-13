# -*- coding: utf-8 -*-
"""快乐8历史数据质量检查。

检查目标：确保官方历史开奖数据在进入统计、回测和 H5 展示前满足基本约束。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

NUMBER_COLUMNS = [f"红球_{idx}" for idx in range(1, 21)]


@dataclass(frozen=True)
class DataQualityIssue:
    """单条数据质量问题。"""

    code: str
    level: str
    issue: str
    message: str


@dataclass(frozen=True)
class DataQualityResult:
    """快乐8数据质量检查结果。"""

    ok: bool
    total_issues: int
    latest_issue: str | None
    error_count: int
    warning_count: int
    items: list[DataQualityIssue]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "totalIssues": self.total_issues,
            "latestIssue": self.latest_issue,
            "errorCount": self.error_count,
            "warningCount": self.warning_count,
            "items": [asdict(item) for item in self.items],
            "warnings": self.warnings,
        }


def _missing_columns(df: pd.DataFrame) -> list[str]:
    required = ["期数", *NUMBER_COLUMNS]
    return [column for column in required if column not in df.columns]


def _row_numbers(row: pd.Series, columns: Sequence[str]) -> list[int | None]:
    numbers: list[int | None] = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            numbers.append(None)
            continue
        try:
            numbers.append(int(value))
        except (TypeError, ValueError):
            numbers.append(None)
    return numbers


def check_kl8_data_quality(df: pd.DataFrame) -> DataQualityResult:
    """检查快乐8开奖数据是否满足 20 个唯一号码、范围 1-80、期号唯一等约束。"""

    items: list[DataQualityIssue] = []
    warnings: list[str] = []
    if df.empty:
        items.append(DataQualityIssue("EMPTY_DATA", "error", "", "历史数据为空"))
        return DataQualityResult(False, 0, None, 1, 0, items, warnings)

    missing = _missing_columns(df)
    if missing:
        items.append(DataQualityIssue("MISSING_COLUMN", "error", "", f"缺少必要字段：{', '.join(missing)}"))
        return DataQualityResult(False, len(df), None, 1, 0, items, warnings)

    work = df.copy()
    work["期数"] = work["期数"].astype(str)
    duplicated_issues = sorted(work.loc[work["期数"].duplicated(keep=False), "期数"].unique())
    for issue in duplicated_issues:
        items.append(DataQualityIssue("DUPLICATE_ISSUE", "error", str(issue), f"期号 {issue} 重复出现"))

    for _, row in work.iterrows():
        issue = str(row["期数"])
        numbers = _row_numbers(row, NUMBER_COLUMNS)
        if any(number is None for number in numbers):
            items.append(DataQualityIssue("INVALID_NUMBER", "error", issue, f"期号 {issue} 存在空值或非数字号码"))
            continue

        valid_numbers = [int(number) for number in numbers if number is not None]
        if len(valid_numbers) != 20:
            items.append(DataQualityIssue("NUMBER_COUNT", "error", issue, f"期号 {issue} 号码数量不是 20 个"))
        out_of_range = [number for number in valid_numbers if number < 1 or number > 80]
        if out_of_range:
            items.append(
                DataQualityIssue(
                    "NUMBER_OUT_OF_RANGE",
                    "error",
                    issue,
                    f"期号 {issue} 存在超出 1-80 的号码：{out_of_range}",
                )
            )
        if len(set(valid_numbers)) != len(valid_numbers):
            items.append(DataQualityIssue("DUPLICATE_NUMBER", "error", issue, f"期号 {issue} 存在重复开奖号码"))

    try:
        issue_numbers = [int(issue) for issue in work["期数"].tolist()]
        if issue_numbers != sorted(issue_numbers, reverse=True):
            warnings.append("期号不是严格倒序排列，统计前会按期号重新排序")
    except ValueError:
        warnings.append("存在非数字期号，无法检查期号顺序")

    error_count = sum(1 for item in items if item.level == "error")
    warning_count = len(warnings) + sum(1 for item in items if item.level == "warning")
    latest_issue = str(work["期数"].max())
    return DataQualityResult(error_count == 0, len(work), latest_issue, error_count, warning_count, items, warnings)


def build_data_quality_markdown(result: DataQualityResult) -> str:
    """生成数据质量 Markdown 报告。"""

    status = "数据质量正常" if result.ok else "数据质量异常"
    lines = [
        "# 快乐8数据质量检查",
        "",
        f"状态：{status}",
        f"总期数：{result.total_issues}",
        f"最新期号：{result.latest_issue or '-'}",
        f"错误数：{result.error_count}",
        f"警告数：{result.warning_count}",
        "",
    ]
    if result.items:
        lines.extend(["## 问题明细", "", "| 级别 | 期号 | 代码 | 说明 |", "|---|---:|---|---|"])
        for item in result.items:
            lines.append(f"| {item.level} | {item.issue or '-'} | {item.code} | {item.message} |")
        lines.append("")
    if result.warnings:
        lines.extend(["## 警告", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_data_quality_reports(result: DataQualityResult, output_dir: Path) -> tuple[Path, Path]:
    """写入 JSON 和 Markdown 数据质量报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_quality.json"
    md_path = output_dir / "data_quality.md"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_data_quality_markdown(result), encoding="utf-8")
    return json_path, md_path
