# -*- coding: utf-8 -*-
"""基于最近真实复盘的策略反馈。

该模块只把最近一期复盘暴露的形态问题转换成保守的策略偏好，
不把它解释成可预测规律。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.analysis.pick_tracking import PickEvaluation


@dataclass(frozen=True)
class ReviewFeedback:
    """最近复盘反馈摘要。"""

    enabled: bool
    latest_issue: str | None
    source_parameter: str | None
    recent_repeat_count: int
    max_hit: int
    hit4_count: int
    hit5_plus_count: int
    promoted_parameter: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "latestIssue": self.latest_issue,
            "sourceParameter": self.source_parameter,
            "recentRepeatCount": self.recent_repeat_count,
            "maxHit": self.max_hit,
            "hit4Count": self.hit4_count,
            "hit5PlusCount": self.hit5_plus_count,
            "promotedParameter": self.promoted_parameter,
            "reason": self.reason,
        }


def _number_columns(history: pd.DataFrame) -> list[str]:
    columns = [f"红球_{idx}" for idx in range(1, 21)]
    if "期数" not in history.columns or any(column not in history.columns for column in columns):
        raise ValueError("历史数据必须包含【期数】和【红球_1】到【红球_20】列")
    return columns


def recent_adjacent_repeat_count(history: pd.DataFrame) -> int:
    """计算最新两期快乐8开奖号码的重号数量。"""

    if len(history) < 2:
        return 0
    columns = _number_columns(history)
    sorted_history = history.copy()
    sorted_history["期数"] = sorted_history["期数"].astype(str)
    sorted_history = sorted_history.sort_values("期数", ascending=False).reset_index(drop=True)
    latest = {int(sorted_history.iloc[0][column]) for column in columns}
    previous = {int(sorted_history.iloc[1][column]) for column in columns}
    return len(latest & previous)


def latest_evaluation(evaluations: Sequence[PickEvaluation]) -> PickEvaluation | None:
    """取目标期号最新的一条复盘。"""

    if not evaluations:
        return None
    return sorted(evaluations, key=lambda item: item.target_issue)[-1]


def build_review_feedback(
    history: pd.DataFrame,
    evaluations: Sequence[PickEvaluation],
    *,
    promoted_parameter: str | None = None,
) -> ReviewFeedback:
    """汇总最新开奖重号和最近复盘命中形态。"""

    repeat_count = recent_adjacent_repeat_count(history)
    evaluation = latest_evaluation(evaluations)
    if evaluation is None:
        return ReviewFeedback(False, None, None, repeat_count, 0, 0, 0, None, "暂无可用实盘复盘")

    hit_counts = [item.hit_count for item in evaluation.group_results]
    max_hit = max(hit_counts, default=0)
    hit4_count = sum(1 for value in hit_counts if value == 4)
    hit5_plus_count = sum(1 for value in hit_counts if value >= 5)
    enabled = promoted_parameter is not None
    reason = "未触发复盘策略切换"
    if enabled:
        reason = f"最近两期重号 {repeat_count} 个，且最近复盘有 {hit4_count} 组中4，启用重号桥接"
    return ReviewFeedback(
        enabled=enabled,
        latest_issue=evaluation.target_issue,
        source_parameter=evaluation.parameter_name,
        recent_repeat_count=repeat_count,
        max_hit=max_hit,
        hit4_count=hit4_count,
        hit5_plus_count=hit5_plus_count,
        promoted_parameter=promoted_parameter,
        reason=reason,
    )


def apply_review_feedback(
    selected_result: Any,
    results: Sequence[Any],
    *,
    mode: str,
    history: pd.DataFrame,
    evaluations: Sequence[PickEvaluation],
    score_tolerance: float = 0.18,
) -> tuple[Any, str, ReviewFeedback]:
    """根据最新复盘保守选择重号策略。

    仅在 auto 模式下生效；当最近两期重号较多，最近复盘多组停在中4，且
    最佳 repeat_* 参数和当前第一名历史分差不大时，才切到 repeat_* 参数。
    """

    if mode != "auto" or not results:
        feedback = build_review_feedback(history, evaluations)
        return selected_result, mode, feedback

    repeat_count = recent_adjacent_repeat_count(history)
    evaluation = latest_evaluation(evaluations)
    hit_counts = [item.hit_count for item in evaluation.group_results] if evaluation else []
    max_hit = max(hit_counts, default=0)
    hit4_count = sum(1 for value in hit_counts if value == 4)
    hit5_plus_count = sum(1 for value in hit_counts if value >= 5)
    repeat_candidates = [result for result in results if result.config.name.startswith("repeat_")]
    promoted_parameter = None

    if repeat_count >= 5 and hit5_plus_count == 0 and (hit4_count >= 2 or max_hit >= 4) and repeat_candidates:
        best_repeat = max(repeat_candidates, key=lambda result: result.score)
        if float(results[0].score) - float(best_repeat.score) <= score_tolerance:
            selected_result = best_repeat
            mode = "auto_review_feedback"
            promoted_parameter = best_repeat.config.name

    feedback = build_review_feedback(history, evaluations, promoted_parameter=promoted_parameter)
    return selected_result, mode, feedback
