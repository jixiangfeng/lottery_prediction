# -*- coding: utf-8 -*-
"""概率 v2 的固定校准、严格前推开发评估。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.analysis.digit_candidates import DigitCandidateConfig
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_probability import (
    DigitProbabilityCalibration,
    DigitProbabilityConfig,
    build_digit_probability_plan,
    fit_digit_probability_calibration,
)
from src.analysis.prediction_viability import (
    PredictionViabilityReport,
    build_prediction_viability_report,
    calculate_group_random_probability,
)
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitProbabilityWalkForwardIssue:
    """概率 v2 对一个目标期的开奖前结果。"""

    issue: str
    train_end_issue: str
    train_size: int
    actual_text: str
    actual_probability: float
    log_loss: float
    brier_score: float
    actual_midrank: float
    actual_rank_percentile: float
    direct_candidates: list[str]
    group_candidates: list[str]
    direct_hit: bool
    group_hit: bool
    direct_random_probability: float
    group_random_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "trainEndIssue": self.train_end_issue,
            "trainSize": self.train_size,
            "actualText": self.actual_text,
            "actualProbability": self.actual_probability,
            "logLoss": self.log_loss,
            "brierScore": self.brier_score,
            "actualMidrank": self.actual_midrank,
            "actualRankPercentile": self.actual_rank_percentile,
            "directCandidates": self.direct_candidates,
            "groupCandidates": self.group_candidates,
            "directHit": self.direct_hit,
            "groupHit": self.group_hit,
            "directRandomProbability": self.direct_random_probability,
            "groupRandomProbability": self.group_random_probability,
        }


@dataclass(frozen=True)
class DigitProbabilityWalkForwardReport:
    """固定一次校准参数后，对后续目标区间做严格前推的报告。"""

    rule_code: str
    display_name: str
    development_only: bool
    calibration_train_end_issue: str
    calibration: DigitProbabilityCalibration
    issues: list[DigitProbabilityWalkForwardIssue]
    mean_log_loss: float
    uniform_log_loss: float
    log_loss_improvement: float
    mean_brier_score: float
    uniform_brier_score: float
    brier_improvement: float
    mean_actual_rank_percentile: float
    mean_reciprocal_rank: float
    viability: PredictionViabilityReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "experimentModel": "digit_probability_v2",
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "developmentOnly": self.development_only,
            "calibrationTrainEndIssue": self.calibration_train_end_issue,
            "calibration": self.calibration.to_dict(),
            "periods": len(self.issues),
            "meanLogLoss": self.mean_log_loss,
            "uniformLogLoss": self.uniform_log_loss,
            "logLossImprovement": self.log_loss_improvement,
            "meanBrierScore": self.mean_brier_score,
            "uniformBrierScore": self.uniform_brier_score,
            "brierImprovement": self.brier_improvement,
            "meanActualRankPercentile": self.mean_actual_rank_percentile,
            "meanReciprocalRank": self.mean_reciprocal_rank,
            "directHits": sum(issue.direct_hit for issue in self.issues),
            "groupHits": sum(issue.group_hit for issue in self.issues),
            "viability": self.viability.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def run_digit_probability_walk_forward(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    periods: int = 500,
    min_train_size: int = 100,
    candidate_count: int = 10,
    candidate_config: DigitCandidateConfig | None = None,
    probability_config: DigitProbabilityConfig | None = None,
) -> DigitProbabilityWalkForwardReport:
    """在最早目标期之前冻结校准参数，再逐期评估概率 v2。"""

    if periods <= 0 or min_train_size <= 0 or candidate_count <= 0:
        raise ValueError("periods、min_train_size、candidate_count 必须为正整数")
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    available = len(chronological) - min_train_size
    if available <= 0:
        raise ValueError(f"历史数据不足：至少需要 {min_train_size + 1} 期")
    selected_periods = min(periods, available)
    first_target_index = len(chronological) - selected_periods
    development = chronological.iloc[:first_target_index].copy()
    if len(development) < min_train_size:
        raise ValueError("最早目标期之前的校准历史不足")
    base_config = candidate_config or DigitCandidateConfig(
        count=candidate_count,
        ranking_mode="ensemble",
        exclude_latest=False,
        constraint_mode="off",
    )
    calibration = fit_digit_probability_calibration(
        development,
        rule,
        candidate_config=base_config,
        probability_config=probability_config,
    )
    calibration_train_end_issue = str(development.iloc[-1]["期数"])
    issues = []
    for target_index in range(first_target_index, len(chronological)):
        train = chronological.iloc[:target_index].copy()
        target = chronological.iloc[target_index]
        plan = build_digit_probability_plan(
            train,
            rule,
            candidate_count=candidate_count,
            candidate_config=base_config,
            probability_config=probability_config,
            calibration=calibration,
        )
        actual_text = "".join(
            str(int(target[column])) for column in rule.number_columns
        )
        actual_probability = plan.distribution.probability_for(actual_text)
        actual_midrank = plan.distribution.midrank(actual_text)
        direct = [candidate.text for candidate in plan.direct_candidates]
        groups = [candidate.group_key for candidate in plan.group_candidates]
        actual_group = "".join(sorted(actual_text))
        direct_random_probability = len(set(direct)) / (10**rule.draw_count)
        group_random_probability = calculate_group_random_probability(groups)
        issues.append(
            DigitProbabilityWalkForwardIssue(
                issue=str(target["期数"]),
                train_end_issue=str(train.iloc[-1]["期数"]),
                train_size=len(train),
                actual_text=actual_text,
                actual_probability=actual_probability,
                log_loss=-math.log(max(actual_probability, 1e-300)),
                brier_score=(
                    plan.distribution.squared_probability_sum
                    - 2.0 * actual_probability
                    + 1.0
                ),
                actual_midrank=actual_midrank,
                actual_rank_percentile=actual_midrank
                / len(plan.distribution.probabilities),
                direct_candidates=direct,
                group_candidates=groups,
                direct_hit=actual_text in direct,
                group_hit=actual_group in groups,
                direct_random_probability=direct_random_probability,
                group_random_probability=group_random_probability,
            )
        )
    uniform_probability = 1.0 / (10**rule.draw_count)
    uniform_log_loss = -math.log(uniform_probability)
    uniform_brier = 1.0 - uniform_probability
    mean_log_loss = mean(issue.log_loss for issue in issues)
    mean_brier = mean(issue.brier_score for issue in issues)
    viability = build_prediction_viability_report(
        [issue.direct_hit for issue in issues],
        [issue.direct_random_probability for issue in issues],
        group_hits=[issue.group_hit for issue in issues],
        group_random_probabilities=[issue.group_random_probability for issue in issues],
    )
    return DigitProbabilityWalkForwardReport(
        rule_code=rule.code,
        display_name=rule.display_name,
        development_only=True,
        calibration_train_end_issue=calibration_train_end_issue,
        calibration=calibration,
        issues=issues,
        mean_log_loss=mean_log_loss,
        uniform_log_loss=uniform_log_loss,
        log_loss_improvement=uniform_log_loss - mean_log_loss,
        mean_brier_score=mean_brier,
        uniform_brier_score=uniform_brier,
        brier_improvement=uniform_brier - mean_brier,
        mean_actual_rank_percentile=mean(
            issue.actual_rank_percentile for issue in issues
        ),
        mean_reciprocal_rank=mean(1.0 / issue.actual_midrank for issue in issues),
        viability=viability,
    )


def build_digit_probability_walk_forward_markdown(
    report: DigitProbabilityWalkForwardReport,
) -> str:
    """生成概率 v2 开发评估 Markdown。"""

    calibration = report.calibration
    direct_hits = sum(issue.direct_hit for issue in report.issues)
    group_hits = sum(issue.group_hit for issue in report.issues)
    lines = [
        f"# {report.display_name} 概率 v2 严格前推开发评估",
        "",
        "## 边界",
        "",
        "- 本报告使用已经查看过的历史，只能用于开发诊断，不能作为新的未见验证。",
        f"- 校准参数只使用 `{report.calibration_train_end_issue}` 及以前数据，之后逐期保持固定。",
        "- 真正可行性仍须建立新实验 ID，并使用未来开奖前冻结样本验证。",
        "",
        "## 校准闸门",
        "",
        f"- 验证期数：`{calibration.validation_periods}`；选参/守门："
        f"`{calibration.selection_periods}/{calibration.holdout_periods}`",
        f"- 候选学习权重：`{calibration.selected_learned_weight:.2f}`；"
        f"实际应用：`{calibration.applied_learned_weight:.2f}`；模型：`{calibration.selected_model}`；"
        f"温度：`{calibration.temperature:.2f}`",
        f"- 校准结果：`{'通过' if calibration.passed else '回退均匀分布'}`",
        f"- 回退原因：`{calibration.fallback_reason or '-'}`",
        "",
        "## 概率质量",
        "",
        "| 指标 | v2 | 均匀分布 | 改善（正数更好） |",
        "|---|---:|---:|---:|",
        f"| Log Loss | {report.mean_log_loss:.6f} | {report.uniform_log_loss:.6f} | {report.log_loss_improvement:+.6f} |",
        f"| Brier Score | {report.mean_brier_score:.6f} | {report.uniform_brier_score:.6f} | {report.brier_improvement:+.6f} |",
        "",
        f"- 平均真实开奖号排名分位：`{report.mean_actual_rank_percentile:.2%}`",
        f"- MRR：`{report.mean_reciprocal_rank:.6f}`",
        "",
        "## 命中与随机闸门",
        "",
        f"- 直选：`{direct_hits}/{len(report.issues)}`；组选：`{group_hits}/{len(report.issues)}`",
        f"- 整体：`{'通过' if report.viability.viable else '不通过'}`",
        f"- 原因：{report.viability.reason}",
        "",
        "即使开发评估通过，也不能替代未来开奖前验证，不保证中奖或盈利。",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_digit_probability_walk_forward_reports(
    report: DigitProbabilityWalkForwardReport,
    output_dir: str | Path,
    *,
    prefix: str = "digit_probability_v2",
) -> tuple[Path, Path]:
    """原子写入概率 v2 Markdown 与 JSON。"""

    directory = Path(output_dir)
    markdown_path = directory / f"{prefix}_{report.rule_code}.md"
    json_path = directory / f"{prefix}_{report.rule_code}.json"
    _atomic_write(markdown_path, build_digit_probability_walk_forward_markdown(report))
    _atomic_write(
        json_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
    )
    return markdown_path, json_path


__all__ = [
    "DigitProbabilityWalkForwardIssue",
    "DigitProbabilityWalkForwardReport",
    "build_digit_probability_walk_forward_markdown",
    "run_digit_probability_walk_forward",
    "write_digit_probability_walk_forward_reports",
]
