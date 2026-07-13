# -*- coding: utf-8 -*-
"""数字彩候选回测模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.digit_candidates import DigitCandidateResult
from src.analysis.digit_data import normalize_digit_dataframe
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitBacktestRow:
    """单期回测结果。"""

    issue: str
    actual_text: str
    direct_hit_texts: list[str]
    group_hit_texts: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "actualText": self.actual_text,
            "directHitTexts": self.direct_hit_texts,
            "groupHitTexts": self.group_hit_texts,
        }


@dataclass(frozen=True)
class DigitBacktestSummary:
    """数字彩候选回测摘要。"""

    rule_code: str
    display_name: str
    draw_count: int
    candidate_count: int
    total_checks: int
    direct_hits: int
    direct_hit_rate: float
    group_hits: int | None
    group_hit_rate: float | None
    rows: list[DigitBacktestRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "drawCount": self.draw_count,
            "candidateCount": self.candidate_count,
            "totalChecks": self.total_checks,
            "directHits": self.direct_hits,
            "directHitRate": self.direct_hit_rate,
            "groupHits": self.group_hits,
            "groupHitRate": self.group_hit_rate,
            "rows": [row.to_dict() for row in self.rows],
        }


def _text(numbers: list[int]) -> str:
    return "".join(str(int(number)) for number in numbers)


def _group_key(text: str) -> str:
    return "".join(sorted(text))


def backtest_digit_candidates(
    history: pd.DataFrame,
    rule: LotteryRule,
    candidate_result: DigitCandidateResult,
) -> DigitBacktestSummary:
    """回放历史开奖，统计候选直选/组选命中。"""

    df = normalize_digit_dataframe(history, rule)
    candidates = candidate_result.candidates
    direct_texts = [candidate.text for candidate in candidates]
    group_enabled = rule.draw_count == 3
    group_keys = {candidate.text: _group_key(candidate.text) for candidate in candidates}
    rows: list[DigitBacktestRow] = []
    direct_hits = 0
    group_hits = 0

    for _, row in df.iterrows():
        numbers = [int(row[column]) for column in rule.number_columns]
        actual_text = _text(numbers)
        actual_group = _group_key(actual_text)
        direct = [text for text in direct_texts if text == actual_text]
        group = [text for text in direct_texts if group_keys[text] == actual_group] if group_enabled else None
        direct_hits += len(direct)
        if group is not None:
            group_hits += len(group)
        rows.append(
            DigitBacktestRow(
                issue=str(row["期数"]),
                actual_text=actual_text,
                direct_hit_texts=direct,
                group_hit_texts=group,
            )
        )

    total_checks = len(df) * len(candidates)
    return DigitBacktestSummary(
        rule_code=rule.code,
        display_name=rule.display_name,
        draw_count=len(df),
        candidate_count=len(candidates),
        total_checks=total_checks,
        direct_hits=direct_hits,
        direct_hit_rate=direct_hits / total_checks if total_checks else 0.0,
        group_hits=group_hits if group_enabled else None,
        group_hit_rate=(group_hits / total_checks if total_checks else 0.0) if group_enabled else None,
        rows=rows,
    )


def build_digit_backtest_markdown(summary: DigitBacktestSummary) -> str:
    """生成数字彩候选回测 Markdown。"""

    lines = [
        "## 数字彩候选回测",
        "",
        f"- 回测期数：`{summary.draw_count}`",
        f"- 候选数量：`{summary.candidate_count}`",
        f"- 直选命中：`{summary.direct_hits}` / `{summary.total_checks}`，命中率 `{summary.direct_hit_rate:.2%}`",
    ]
    if summary.group_hits is not None:
        lines.append(
            f"- 组选命中：`{summary.group_hits}` / `{summary.total_checks}`，命中率 `{summary.group_hit_rate:.2%}`"
        )
    lines.extend(
        [
            "",
            "说明：回测只是把当前候选放回历史开奖中检查命中情况，不能代表未来表现。",
            "",
        ]
    )
    return "\n".join(lines)
