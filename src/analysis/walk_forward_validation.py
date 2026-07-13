# -*- coding: utf-8 -*-
"""参数反过拟合前推验证。

把每个参数的滑动窗口结果拆成“训练窗口”和“测试窗口”：训练窗口用于观察历史拟合，
测试窗口用于观察前推泛化。差距越大，说明越可能只是碰巧拟合某段历史。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class WalkForwardRow:
    parameter: str
    trainWindowCount: int
    testWindowCount: int
    trainMeanRoi: float
    testMeanRoi: float
    generalizationGap: float
    testWinRate: float
    score: float
    riskLevel: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "trainWindowCount": self.trainWindowCount,
            "testWindowCount": self.testWindowCount,
            "trainMeanRoi": self.trainMeanRoi,
            "testMeanRoi": self.testMeanRoi,
            "generalizationGap": self.generalizationGap,
            "testWinRate": self.testWinRate,
            "score": self.score,
            "riskLevel": self.riskLevel,
        }


@dataclass(frozen=True)
class WalkForwardValidation:
    enabled: bool
    trainWindowCount: int
    testWindowCount: int
    bestParameter: str | None
    rows: list[WalkForwardRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "trainWindowCount": self.trainWindowCount,
            "testWindowCount": self.testWindowCount,
            "bestParameter": self.bestParameter,
            "rows": [row.to_dict() for row in self.rows],
        }


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _risk(gap: float, test_win_rate: float) -> str:
    if gap >= 0.35:
        return "高"
    if gap >= 0.18:
        return "中"
    return "低"


def validate_parameter_walk_forward(
    parameter_results: Sequence[Any],
    *,
    train_window_count: int = 3,
) -> WalkForwardValidation:
    """基于参数搜索的滑动窗口结果做训练/测试前推验证。"""

    rows: list[WalkForwardRow] = []
    for result in parameter_results:
        windows = list(result.sliding_summary.windows)
        if len(windows) < 2:
            continue
        split = min(max(1, train_window_count), len(windows) - 1)
        train = windows[:split]
        test = windows[split:]
        train_rois = [float(item.summary.roi) for item in train]
        test_rois = [float(item.summary.roi) for item in test]
        train_mean = _mean(train_rois)
        test_mean = _mean(test_rois)
        gap = round(abs(train_mean - test_mean), 4)
        test_win_rate = round(sum(1 for roi in test_rois if roi >= 0) / len(test_rois), 4) if test_rois else 0.0
        # 更看重测试窗口；收益率长期为负时也允许比较“相对更稳”。
        score = round(test_mean - 0.5 * gap + 0.1 * test_win_rate, 4)
        rows.append(
            WalkForwardRow(
                parameter=result.config.name,
                trainWindowCount=len(train),
                testWindowCount=len(test),
                trainMeanRoi=train_mean,
                testMeanRoi=test_mean,
                generalizationGap=gap,
                testWinRate=test_win_rate,
                score=score,
                riskLevel=_risk(gap, test_win_rate),
            )
        )
    rows.sort(key=lambda row: row.score, reverse=True)
    return WalkForwardValidation(
        enabled=bool(rows),
        trainWindowCount=rows[0].trainWindowCount if rows else 0,
        testWindowCount=rows[0].testWindowCount if rows else 0,
        bestParameter=rows[0].parameter if rows else None,
        rows=rows,
    )


def build_walk_forward_markdown(validation: WalkForwardValidation) -> str:
    """生成反过拟合前推验证 Markdown。"""

    lines = [
        "## 反过拟合前推验证",
        "",
        f"- 最稳参数：`{validation.bestParameter or '暂无'}`",
        f"- 窗口拆分：训练 `{validation.trainWindowCount}` / 测试 `{validation.testWindowCount}`",
        "",
        "| 参数 | 训练收益率 | 测试收益率 | 泛化差距 | 测试胜率 | 风险 | 评分 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in validation.rows:
        lines.append(
            f"| {row.parameter} | {row.trainMeanRoi:.2%} | {row.testMeanRoi:.2%} | "
            f"{row.generalizationGap:.2%} | {row.testWinRate:.2%} | {row.riskLevel} | {row.score:.4f} |"
        )
    lines.extend([
        "",
        "说明：前推验证用较早窗口模拟训练、后续窗口模拟测试；泛化差距越小、测试风险越低，参数越不容易只是历史碰巧拟合。",
        "",
    ])
    return "\n".join(lines)
