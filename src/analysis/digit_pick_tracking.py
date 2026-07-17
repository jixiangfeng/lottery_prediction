# -*- coding: utf-8 -*-
"""数字彩推荐留痕、开奖后复盘与累计表现。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.analysis.digit_candidates import (
    ENSEMBLE_MODEL_NAMES,
    DigitBettingCandidateResult,
    DigitCandidateConfig,
)
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_probability import DigitProbabilityPlan
from src.analysis.digit_probability_online import DigitOnlineProbabilityPlan
from src.analysis.digit_statistics import DigitStatisticsResult
from src.analysis.prediction_viability import poisson_binomial_right_tail
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitPickEvaluation:
    """一份真实推荐快照对应的下一期开奖复盘。"""

    rule_code: str
    display_name: str
    source_issue: str
    target_issue: str
    ranking_mode: str
    actual_text: str
    direct_candidates: list[str]
    group_candidates: list[str]
    direct_hit: bool
    group_hit: bool | None
    experiment_id: str | None = None
    model_candidates: dict[str, list[str]] = field(default_factory=dict)
    model_hits: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "sourceIssue": self.source_issue,
            "targetIssue": self.target_issue,
            "rankingMode": self.ranking_mode,
            "experimentId": self.experiment_id,
            "actualText": self.actual_text,
            "directCandidates": self.direct_candidates,
            "groupCandidates": self.group_candidates,
            "directHit": self.direct_hit,
            "groupHit": self.group_hit,
            "modelCandidates": self.model_candidates,
            "modelHits": self.model_hits,
        }


@dataclass(frozen=True)
class DigitLiveSummary:
    """数字彩多期真实推荐累计命中摘要。"""

    rule_code: str
    display_name: str
    period_count: int
    latest_issue: str
    direct_hits: int
    direct_hit_rate: float
    group_hits: int | None
    group_hit_rate: float | None
    ranking_mode_counts: dict[str, int]
    model_performance: dict[str, dict[str, float | int]] = field(default_factory=dict)
    recommended_model_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "periodCount": self.period_count,
            "latestIssue": self.latest_issue,
            "directHits": self.direct_hits,
            "directHitRate": self.direct_hit_rate,
            "groupHits": self.group_hits,
            "groupHitRate": self.group_hit_rate,
            "rankingModeCounts": self.ranking_mode_counts,
            "modelPerformance": self.model_performance,
            "recommendedModelWeights": self.recommended_model_weights,
        }


def save_digit_pick_snapshot(
    stats: DigitStatisticsResult,
    plan: (
        DigitBettingCandidateResult | DigitProbabilityPlan | DigitOnlineProbabilityPlan
    ),
    output_dir: str | Path,
    *,
    immutable: bool = False,
    experiment_id: str | None = None,
) -> Path:
    """保存基于当前最新一期生成的候选，可显式冻结为不可覆盖快照。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plan_payload = plan.to_dict()
    if experiment_id is not None and not re.fullmatch(r"[a-z0-9_]+", experiment_id):
        raise ValueError("experiment_id 只允许小写字母、数字和下划线")
    stable_payload = {
        "ruleCode": plan.rule_code,
        "displayName": plan.display_name,
        "sourceIssue": str(stats.latest_issue),
        "rankingMode": plan.config.ranking_mode,
        "candidateConfig": plan_payload["config"],
        "directCandidates": [candidate.text for candidate in plan.direct_candidates],
        "groupCandidates": [candidate.group_key for candidate in plan.group_candidates],
        "modelCandidates": plan.model_candidates,
        "ensembleModelWeights": {
            name: weight
            for name, weight in zip(
                ENSEMBLE_MODEL_NAMES, plan.config.ensemble_model_weights
            )
        },
    }
    if experiment_id is not None:
        stable_payload["experimentId"] = experiment_id
    if "probabilityModel" in plan_payload:
        stable_payload["probabilityModel"] = plan_payload["probabilityModel"]
        stable_payload["probabilityDistributionFingerprint"] = plan_payload[
            "probabilityDistributionFingerprint"
        ]
        stable_payload["directCandidateProbabilities"] = {
            str(candidate["text"]): candidate["predictedProbability"]
            for candidate in plan_payload["directCandidates"]
        }
        stable_payload["groupCandidateProbabilities"] = {
            str(candidate["groupKey"]): candidate["predictedProbability"]
            for candidate in plan_payload["groupCandidates"]
        }
        if "probabilityCalibration" in plan_payload:
            stable_payload["probabilityCalibration"] = plan_payload[
                "probabilityCalibration"
            ]
        if "onlineProbability" in plan_payload:
            online_payload = plan_payload["onlineProbability"]
            stable_payload["onlineProbability"] = {
                "config": online_payload["config"],
                "state": online_payload["state"],
            }
    serialized = json.dumps(
        stable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    payload = {
        "schemaVersion": 3,
        **stable_payload,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "recommendationFingerprint": fingerprint,
        "immutable": immutable,
    }
    experiment_segment = f"_{experiment_id}" if experiment_id else ""
    output = (
        directory / f"{plan.rule_code}{experiment_segment}_{stats.latest_issue}.json"
    )
    if output.exists():
        existing = _load_snapshot(output)
        existing_fingerprint = existing.get("recommendationFingerprint")
        legacy_fields = (
            "ruleCode",
            "displayName",
            "sourceIssue",
            "rankingMode",
            "directCandidates",
            "groupCandidates",
            "modelCandidates",
            "ensembleModelWeights",
        )
        if existing_fingerprint == fingerprint:
            if not immutable or bool(existing.get("immutable")):
                return output
            payload["generatedAt"] = str(
                existing.get("generatedAt", payload["generatedAt"])
            )
        elif not existing_fingerprint and all(
            existing.get(field) == payload.get(field) for field in legacy_fields
        ):
            payload["generatedAt"] = str(
                existing.get("generatedAt", payload["generatedAt"])
            )
            payload["immutable"] = bool(existing.get("immutable")) or immutable
        elif immutable or bool(existing.get("immutable")):
            raise FileExistsError(
                f"源期 {stats.latest_issue} 已存在不同推荐快照，禁止开奖前留痕被覆盖"
            )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_digit_pick_snapshot(
    history: pd.DataFrame,
    rule: LotteryRule,
    snapshot_path: str | Path,
) -> DigitPickEvaluation | None:
    """使用快照源期之后的第一期开奖复盘，兼容跨年期号。"""

    payload = _load_snapshot(snapshot_path)
    if str(payload.get("ruleCode")) != rule.code:
        raise ValueError(f"推荐快照玩法与当前玩法不一致：{snapshot_path}")
    source_issue = str(payload["sourceIssue"])
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    later = chronological[chronological["期数"].astype(int) > int(source_issue)]
    if later.empty:
        return None
    target = later.iloc[0]
    actual_text = "".join(str(int(target[column])) for column in rule.number_columns)
    direct_candidates = [str(value) for value in payload.get("directCandidates", [])]
    group_candidates = [str(value) for value in payload.get("groupCandidates", [])]
    model_candidates = {
        str(name): [str(value) for value in values]
        for name, values in payload.get("modelCandidates", {}).items()
    }
    model_hits = {
        name: actual_text in values for name, values in model_candidates.items()
    }
    group_hit = (
        "".join(sorted(actual_text)) in group_candidates
        if rule.draw_count == 3
        else None
    )
    return DigitPickEvaluation(
        rule_code=rule.code,
        display_name=rule.display_name,
        source_issue=source_issue,
        target_issue=str(target["期数"]),
        ranking_mode=str(payload.get("rankingMode", "composite")),
        experiment_id=(
            str(payload["experimentId"]) if payload.get("experimentId") else None
        ),
        actual_text=actual_text,
        direct_candidates=direct_candidates,
        group_candidates=group_candidates,
        direct_hit=actual_text in direct_candidates,
        group_hit=group_hit,
        model_candidates=model_candidates,
        model_hits=model_hits,
    )


def build_digit_evaluation_markdown(evaluation: DigitPickEvaluation) -> str:
    """生成单期数字彩真实推荐复盘。"""

    group_text = (
        "-"
        if evaluation.group_hit is None
        else ("命中" if evaluation.group_hit else "未命中")
    )
    lines = [
        f"# {evaluation.display_name}真实推荐复盘",
        "",
        f"- 推荐依据期：`{evaluation.source_issue}`",
        f"- 开奖期号：`{evaluation.target_issue}`",
        f"- 排序模式：`{evaluation.ranking_mode}`",
        f"- 实验 ID：`{evaluation.experiment_id or 'default'}`",
        f"- 开奖号码：`{evaluation.actual_text}`",
        f"- 直选结果：`{'命中' if evaluation.direct_hit else '未命中'}`",
        f"- 组选结果：`{group_text}`",
        f"- 直选候选数：`{len(evaluation.direct_candidates)}`",
        f"- 组选候选数：`{len(evaluation.group_candidates)}`",
        "",
        "说明：这里只评估开奖前已经保存的推荐，不使用开奖后的数据反向改写候选。",
        "",
    ]
    if evaluation.model_hits:
        lines.extend(["## 逐模型命中", ""])
        for name, hit in sorted(evaluation.model_hits.items()):
            lines.append(f"- `{name}`：`{'命中' if hit else '未命中'}`")
        lines.append("")
    return "\n".join(lines)


def _model_hit_evidence(
    evaluations: Sequence[DigitPickEvaluation], model_name: str
) -> tuple[list[bool], list[float]]:
    """提取逐期模型命中及其同候选数量下的精确随机概率。"""

    hits: list[bool] = []
    probabilities: list[float] = []
    for item in evaluations:
        if model_name not in item.model_hits or model_name not in item.model_candidates:
            continue
        draw_count = len(item.actual_text)
        if draw_count <= 0:
            continue
        candidates = set(item.model_candidates[model_name])
        hits.append(bool(item.model_hits[model_name]))
        probabilities.append(min(1.0, len(candidates) / (10**draw_count)))
    return hits, probabilities


def derive_live_ensemble_weights(
    evaluations: Sequence[DigitPickEvaluation],
    base_weights: Sequence[float],
    *,
    min_samples: int = 100,
) -> tuple[float, ...]:
    """按同候选空间随机基线保守调权，单模型最多浮动 10%。

    只有不少于 ``min_samples`` 期且精确单侧 p 值小于 0.01 的模型才会
    获得正向调整，避免把 Top10 约 1% 的随机命中率误当成 50%。
    """

    if len(base_weights) != len(ENSEMBLE_MODEL_NAMES):
        raise ValueError("基础权重数量与集成模型数量不一致")
    multipliers: list[float] = []
    for name in ENSEMBLE_MODEL_NAMES:
        values, probabilities = _model_hit_evidence(evaluations, name)
        if len(values) < min_samples:
            multipliers.append(1.0)
            continue
        observed_hits = sum(values)
        expected_hits = sum(probabilities)
        p_value = poisson_binomial_right_tail(probabilities, observed_hits)
        smoothed_relative_lift = (observed_hits + 1.0) / (expected_hits + 1.0) - 1.0
        if p_value >= 0.01 or smoothed_relative_lift <= 0:
            multipliers.append(1.0)
            continue
        multiplier = 1.0 + min(0.1, 0.1 * smoothed_relative_lift)
        multipliers.append(multiplier)
    if all(multiplier == 1.0 for multiplier in multipliers):
        return tuple(float(value) for value in base_weights)
    raw = [
        float(weight) * multiplier
        for weight, multiplier in zip(base_weights, multipliers)
    ]
    base = [float(value) for value in base_weights]
    base_total = sum(base)
    if sum(raw) <= 0:
        return tuple(float(value) for value in base_weights)
    lower = [value * 0.9 for value in base]
    upper = [value * 1.1 for value in base]
    adjusted = [0.0] * len(base)
    free = set(range(len(base)))
    remaining_total = base_total
    while free:
        free_raw_total = sum(raw[index] for index in free)
        if free_raw_total <= 0:
            for index in free:
                adjusted[index] = base[index]
            break
        scale = remaining_total / free_raw_total
        violations = []
        for index in free:
            proposed = raw[index] * scale
            if proposed < lower[index]:
                adjusted[index] = lower[index]
                violations.append(index)
            elif proposed > upper[index]:
                adjusted[index] = upper[index]
                violations.append(index)
        if not violations:
            for index in free:
                adjusted[index] = raw[index] * scale
            break
        for index in violations:
            remaining_total -= adjusted[index]
            free.remove(index)
    return tuple(adjusted)


def write_digit_evaluation(
    evaluation: DigitPickEvaluation,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写入单期复盘 Markdown 与 JSON。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    experiment_segment = (
        f"_{evaluation.experiment_id}" if evaluation.experiment_id else ""
    )
    stem = f"{evaluation.rule_code}{experiment_segment}_{evaluation.target_issue}"
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"
    markdown_path.write_text(
        build_digit_evaluation_markdown(evaluation), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown_path, json_path


def build_digit_live_summary(
    evaluations: Sequence[DigitPickEvaluation],
) -> DigitLiveSummary:
    """汇总同一玩法的真实推荐复盘。"""

    if not evaluations:
        raise ValueError("至少需要一条数字彩复盘记录")
    ordered = sorted(evaluations, key=lambda item: int(item.target_issue))
    rule_codes = {item.rule_code for item in ordered}
    if len(rule_codes) != 1:
        raise ValueError("累计汇总只能包含同一玩法")
    direct_hits = sum(item.direct_hit for item in ordered)
    group_values = [
        bool(item.group_hit) for item in ordered if item.group_hit is not None
    ]
    group_hits = sum(group_values) if group_values else None
    model_performance = {}
    for name in ENSEMBLE_MODEL_NAMES:
        values, probabilities = _model_hit_evidence(ordered, name)
        if values:
            hits = sum(values)
            expected_hits = sum(probabilities)
            p_value = poisson_binomial_right_tail(probabilities, hits)
            model_performance[name] = {
                "sampleCount": len(values),
                "hits": hits,
                "hitRate": hits / len(values),
                "randomHitRate": expected_hits / len(values),
                "expectedRandomHits": expected_hits,
                "pValue": p_value,
                "significantAboveRandom": (
                    len(values) >= 100
                    and p_value < 0.01
                    and hits + 1.0 > expected_hits + 1.0
                ),
            }
    recommended_weights = derive_live_ensemble_weights(
        ordered, DigitCandidateConfig().ensemble_model_weights
    )
    return DigitLiveSummary(
        rule_code=ordered[0].rule_code,
        display_name=ordered[0].display_name,
        period_count=len(ordered),
        latest_issue=ordered[-1].target_issue,
        direct_hits=direct_hits,
        direct_hit_rate=direct_hits / len(ordered),
        group_hits=group_hits,
        group_hit_rate=(
            group_hits / len(group_values) if group_hits is not None else None
        ),
        ranking_mode_counts=dict(Counter(item.ranking_mode for item in ordered)),
        model_performance=model_performance,
        recommended_model_weights={
            name: weight
            for name, weight in zip(ENSEMBLE_MODEL_NAMES, recommended_weights)
        },
    )


def build_digit_live_summary_markdown(summary: DigitLiveSummary) -> str:
    """生成数字彩累计真实推荐表现。"""

    group_text = (
        "-"
        if summary.group_hits is None
        else f"{summary.group_hits}/{summary.period_count} ({summary.group_hit_rate:.2%})"
    )
    modes = "，".join(
        f"{name}:{count}" for name, count in sorted(summary.ranking_mode_counts.items())
    )
    lines = [
        f"# {summary.display_name}真实推荐累计表现",
        "",
        f"- 复盘期数：`{summary.period_count}`",
        f"- 最新复盘期号：`{summary.latest_issue}`",
        f"- 直选命中：`{summary.direct_hits}/{summary.period_count}`（`{summary.direct_hit_rate:.2%}`）",
        f"- 组选命中：`{group_text}`",
        f"- 排序模式样本：`{modes}`",
        "",
        "说明：累计结果只统计开奖前已留痕的推荐，样本少时不代表稳定优势。",
        "",
    ]
    if summary.model_performance:
        lines.extend(
            [
                "## 逐模型表现",
                "",
                "| 模型 | 命中 | 样本 | 命中率 | 随机命中率 | p值 | 可调权 | 建议权重 |",
                "|---|---:|---:|---:|---:|---:|---|---:|",
            ]
        )
        for name, values in summary.model_performance.items():
            lines.append(
                f"| {name} | {values['hits']} | {values['sampleCount']} | "
                f"{float(values['hitRate']):.2%} | "
                f"{float(values['randomHitRate']):.2%} | "
                f"{float(values['pValue']):.6f} | "
                f"{'是' if values['significantAboveRandom'] else '否'} | "
                f"{summary.recommended_model_weights.get(name, 0.0):.4f} |"
            )
        lines.extend(
            [
                "",
                "建议权重按同候选空间随机基线做加一平滑；至少 100 期且单侧 p<0.01 才正向调权，单模型最多浮动 10%。",
                "",
            ]
        )
    return "\n".join(lines)


def write_digit_live_summary(
    summary: DigitLiveSummary,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写入累计表现 Markdown 与 JSON。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{summary.rule_code}_live_summary"
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"
    markdown_path.write_text(
        build_digit_live_summary_markdown(summary), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown_path, json_path


def process_digit_pick_evaluations(
    history: pd.DataFrame,
    rule: LotteryRule,
    picks_dir: str | Path,
    evaluations_dir: str | Path,
) -> tuple[list[DigitPickEvaluation], Path | None]:
    """自动复盘所有已开奖快照并刷新累计汇总。"""

    directory = Path(picks_dir)
    evaluations: list[DigitPickEvaluation] = []
    if directory.exists():
        for snapshot in sorted(directory.glob(f"{rule.code}_*.json")):
            evaluation = evaluate_digit_pick_snapshot(history, rule, snapshot)
            if evaluation is not None:
                evaluations.append(evaluation)
                write_digit_evaluation(evaluation, evaluations_dir)
    if not evaluations:
        return [], None
    summary = build_digit_live_summary(evaluations)
    summary_path, _ = write_digit_live_summary(summary, evaluations_dir)
    return evaluations, summary_path


__all__ = [
    "DigitLiveSummary",
    "DigitPickEvaluation",
    "build_digit_live_summary",
    "derive_live_ensemble_weights",
    "evaluate_digit_pick_snapshot",
    "process_digit_pick_evaluations",
    "save_digit_pick_snapshot",
    "write_digit_evaluation",
    "write_digit_live_summary",
]
