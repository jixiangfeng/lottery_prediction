# -*- coding: utf-8 -*-
"""数字型彩票数据加载与标准化。

把福彩3D、排列三、排列五等来源不一的 CSV/DataFrame 统一成：
- 三位：期数, 百位, 十位, 个位
- 五位：期数, 万位, 千位, 百位, 十位, 个位

本模块只做清洗和校验，不负责下载数据或预测。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.lotteries.base import LotteryRule, validate_numbers

ISSUE_COLUMN_ALIASES = ("期数", "期号", "issue", "Issue", "draw", "draw_no", "drawNo")
NUMBER_COLUMN_ALIASES = ("开奖号码", "号码", "number", "numbers", "result", "openCode", "drawNumbers")


def sort_digit_dataframe_by_issue(df: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
    """按数值期号稳定排序，并拒绝歧义或重复期号。"""

    output = df.copy()
    issues = output["期数"].astype(str).str.strip()
    invalid = ~issues.str.fullmatch(r"\d+")
    if invalid.any():
        values = issues[invalid].tolist()[:3]
        raise ValueError(f"数字彩期号必须为纯数字，收到：{values}")
    issue_order = issues.map(int)
    duplicated = issue_order.duplicated(keep=False)
    if duplicated.any():
        values = issues[duplicated].tolist()[:5]
        raise ValueError(f"数字彩数据包含重复期号：{values}")
    output["期数"] = issues
    output["__issue_order"] = issue_order
    return (
        output.sort_values("__issue_order", ascending=ascending, kind="mergesort")
        .drop(columns="__issue_order")
        .reset_index(drop=True)
    )


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    columns = {str(column).strip(): column for column in df.columns}
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    lowered = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in lowered:
            return lowered[key]
    return None


def _digits_from_value(value: Any, expected_len: int) -> list[int]:
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    digits = re.findall(r"\d", text)
    if len(digits) != expected_len:
        raise ValueError(f"开奖号码位数必须为 {expected_len}，收到：{value}")
    return [int(digit) for digit in digits]


def _issue_series(df: pd.DataFrame) -> pd.Series:
    issue_column = _find_column(df, ISSUE_COLUMN_ALIASES)
    if issue_column is None:
        raise ValueError("数字彩数据必须包含期号列，例如：期数、期号、issue")
    return df[issue_column].astype(str).str.strip()


def _normalize_from_position_columns(df: pd.DataFrame, rule: LotteryRule) -> pd.DataFrame | None:
    if not all(column in df.columns for column in rule.number_columns):
        return None
    output = pd.DataFrame({"期数": _issue_series(df)})
    for column in rule.number_columns:
        output[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    return output


def _normalize_from_number_column(df: pd.DataFrame, rule: LotteryRule) -> pd.DataFrame:
    number_column = _find_column(df, NUMBER_COLUMN_ALIASES)
    if number_column is None:
        raise ValueError(
            f"数字彩数据缺少号码列；需要位置列 {rule.number_columns}，或开奖号码/number 等合并号码列"
        )
    output = pd.DataFrame({"期数": _issue_series(df)})
    rows = [_digits_from_value(value, rule.draw_count) for value in df[number_column].tolist()]
    for index, column in enumerate(rule.number_columns):
        output[column] = [numbers[index] for numbers in rows]
    return output


def normalize_digit_dataframe(df: pd.DataFrame, rule: LotteryRule) -> pd.DataFrame:
    """把数字彩 DataFrame 标准化为统一列名并校验号码。"""

    if rule.category != "digit":
        raise ValueError(f"数字彩数据标准化不适用于玩法：{rule.display_name}")
    if df.empty:
        return pd.DataFrame(columns=["期数", *rule.number_columns])

    output = _normalize_from_position_columns(df, rule)
    if output is None:
        output = _normalize_from_number_column(df, rule)

    output["期数"] = output["期数"].astype(str).str.strip()
    for column in rule.number_columns:
        output[column] = pd.to_numeric(output[column], errors="raise").astype(int)

    for _, row in output.iterrows():
        validate_numbers(rule, [int(row[column]) for column in rule.number_columns])

    return sort_digit_dataframe_by_issue(output[["期数", *rule.number_columns]], ascending=False)


def load_digit_csv(path: str | Path, rule: LotteryRule, *, encoding: str = "utf-8") -> pd.DataFrame:
    """读取 CSV 并标准化数字彩数据。"""

    csv_path = Path(path)
    df = pd.read_csv(csv_path, dtype=str, encoding=encoding)
    return normalize_digit_dataframe(df, rule)
