# -*- coding: utf-8 -*-
"""汇总锁定的彩票模型证据，禁止按局部命中率挑选赢家。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from scipy.stats import binom

_RANDOM_BASELINE = 0.05


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


_LOCKED_DIRECT_ROWS: tuple[dict[str, object], ...] = (
    {
        "modelFamily": "core_linear",
        "modelVersion": "learned_ranker_v4_historical_blocks",
        "lottery": "fc3d",
        "periods": 7000,
        "fixedBlocks": 14,
        "hits": 346,
        "randomPValue": 0.5945,
        "properScoreStatus": "joint_gate_failed",
        "stableBlocks": 0,
        "jointPassedBlocks": 0,
        "source": "README.md:132-138",
    },
    {
        "modelFamily": "core_linear",
        "modelVersion": "learned_ranker_v4_historical_blocks",
        "lottery": "pl3",
        "periods": 7000,
        "fixedBlocks": 14,
        "hits": 367,
        "randomPValue": 0.1824,
        "properScoreStatus": "joint_gate_failed",
        "stableBlocks": 0,
        "jointPassedBlocks": 0,
        "source": "README.md:132-138",
    },
    {
        "modelFamily": "nonlinear_tree",
        "modelVersion": "lightgbm_three_position",
        "lottery": "fc3d",
        "periods": 6000,
        "fixedBlocks": 12,
        "hits": 282,
        "randomPValue": 0.8639,
        "properScoreStatus": "worse_than_uniform",
        "stableBlocks": None,
        "jointPassedBlocks": 0,
        "source": "README.md:140-147",
    },
    {
        "modelFamily": "nonlinear_tree",
        "modelVersion": "lightgbm_three_position",
        "lottery": "pl3",
        "periods": 6000,
        "fixedBlocks": 12,
        "hits": 294,
        "randomPValue": 0.6471,
        "properScoreStatus": "worse_than_uniform",
        "stableBlocks": None,
        "jointPassedBlocks": 0,
        "source": "README.md:140-147",
    },
    {
        "modelFamily": "rank_ftrl",
        "modelVersion": "rank_ftrl_v4_1",
        "lottery": "fc3d",
        "periods": 7000,
        "fixedBlocks": 14,
        "hits": 359,
        "randomPValue": 0.3183,
        "properScoreStatus": "joint_gate_failed",
        "stableBlocks": 8,
        "jointPassedBlocks": 1,
        "source": "README.md:149-155",
    },
    {
        "modelFamily": "rank_ftrl",
        "modelVersion": "rank_ftrl_v4_1",
        "lottery": "pl3",
        "periods": 7000,
        "fixedBlocks": 14,
        "hits": 368,
        "randomPValue": 0.1684,
        "properScoreStatus": "joint_gate_failed",
        "stableBlocks": 9,
        "jointPassedBlocks": 0,
        "source": "README.md:149-155",
    },
    {
        "modelFamily": "behavioral",
        "modelVersion": "behavioral_context_v2",
        "lottery": "fc3d",
        "periods": 6500,
        "fixedBlocks": 13,
        "hits": 323,
        "randomPValue": None,
        "pairedPValue": 0.3266,
        "properScoreStatus": "worse_than_core",
        "stableBlocks": None,
        "jointPassedBlocks": 0,
        "source": "CHANGELOG.md:51",
    },
    {
        "modelFamily": "behavioral",
        "modelVersion": "behavioral_context_v2",
        "lottery": "pl3",
        "periods": 6500,
        "fixedBlocks": 13,
        "hits": 333,
        "randomPValue": None,
        "pairedPValue": 0.8289,
        "properScoreStatus": "worse_than_core",
        "stableBlocks": None,
        "jointPassedBlocks": 0,
        "source": "CHANGELOG.md:51",
    },
)

_INCOMPLETE_EVIDENCE: tuple[dict[str, object], ...] = (
    {
        "modelVersion": "behavioral_context_v1",
        "lottery": "fc3d",
        "periods": 500,
        "hits": 28,
        "hitRate": 0.056,
        "reasonNotComparable": "只覆盖单个复用开发块，未覆盖全部固定块",
        "decision": "closed",
        "source": "agent_report.md:46-48",
    },
    {
        "modelVersion": "behavioral_context_v1",
        "lottery": "pl3",
        "periods": 500,
        "hits": 32,
        "hitRate": 0.064,
        "reasonNotComparable": "只覆盖单个复用开发块，未覆盖全部固定块",
        "decision": "closed",
        "source": "agent_report.md:46-48",
    },
    {
        "modelVersion": "rank_ftrl_v4_1_drop_sum_distribution",
        "lottery": "fc3d",
        "periods": 7000,
        "hits": 379,
        "hitRate": 379 / 7000,
        "randomPValue": 0.0604,
        "reasonNotComparable": "事后归因形成的单彩种候选，pl3没有共享删除候选",
        "decision": "future_only_not_activated",
        "source": "README.md:157",
    },
)

_NONCOMPARABLE_DIAGNOSTICS: tuple[dict[str, object], ...] = (
    {
        "objective": "group_projection_top50",
        "lottery": "fc3d",
        "periods": 7000,
        "hits": 1349,
        "weightedBaselineRate": 0.1873,
        "pValue": 0.1242,
        "decision": "closed",
        "source": "README.md:161",
    },
    {
        "objective": "group_projection_top50",
        "lottery": "pl3",
        "periods": 7000,
        "hits": 1215,
        "weightedBaselineRate": 0.1735,
        "pValue": 0.4990,
        "decision": "closed",
        "source": "README.md:161",
    },
    {
        "objective": "independent_group_top10",
        "lottery": "fc3d",
        "periods": 7000,
        "hits": 445,
        "weightedBaselineRate": 0.06,
        "pValue": 0.1095,
        "decision": "closed",
        "source": "README.md:161",
    },
    {
        "objective": "independent_group_top10",
        "lottery": "pl3",
        "periods": 7000,
        "hits": 418,
        "weightedBaselineRate": 0.06,
        "pValue": 0.5472,
        "decision": "closed",
        "source": "README.md:161",
    },
)


def _normalized_locked_row(source: Mapping[str, object]) -> dict[str, object]:
    row = dict(source)
    periods = int(cast(int, row["periods"]))
    hits = int(cast(int, row["hits"]))
    if row.get("randomPValue") is None:
        row["randomPValue"] = float(binom.sf(hits - 1, periods, _RANDOM_BASELINE))
    row.update(
        {
            "objective": "direct_top50",
            "topK": 50,
            "hits": hits,
            "hitRate": hits / periods,
            "randomBaseline": _RANDOM_BASELINE,
            "evidenceStatus": "retrospective_reused_development",
            "decision": "closed",
        }
    )
    return row


def _behavior_row(report: Mapping[str, object]) -> dict[str, object]:
    if bool(report.get("frozenTestRead")):
        raise ValueError("行为总账不得读取Frozen")
    coverage = cast(Mapping[str, object], report.get("developmentCoverage", {}))
    if not bool(coverage.get("allCompleteFixedBlocksEvaluated")):
        raise ValueError("行为总账必须覆盖全部完整固定块")
    if bool(report.get("currentDailyModelReplaced")):
        raise ValueError("行为挑战器不得替换日常模型")

    version = str(report["modelVersion"])
    lottery = str(report["lottery"])
    groups = cast(Mapping[str, object], report["groups"])
    challenger = cast(Mapping[str, object], groups["C"])
    comparisons = cast(Mapping[str, object], report["comparisons"])
    comparison = cast(Mapping[str, object], comparisons["CvsA"])
    gate = cast(Mapping[str, object], report["gate"])
    periods = int(cast(int, challenger["periods"]))
    hits = int(cast(int, challenger["hits"]))
    blocks = cast(Sequence[float], challenger["fixed500BlockHitRates"])
    boundary = cast(Mapping[str, object], challenger["behavioralBoundaryContribution"])
    log_improvement = _as_float(comparison["meanLogLossImprovement"])
    brier_improvement = _as_float(comparison["meanBrierImprovement"])
    proper_status = (
        "improved_vs_core"
        if log_improvement > 0 and brier_improvement > 0
        else "worse_than_core"
    )
    return {
        "modelFamily": "behavioral",
        "modelVersion": version,
        "lottery": lottery,
        "objective": "direct_top50",
        "topK": 50,
        "periods": periods,
        "fixedBlocks": len(blocks),
        "hits": hits,
        "hitRate": hits / periods,
        "randomBaseline": _RANDOM_BASELINE,
        "randomPValue": _as_float(challenger["top50PValue"]),
        "pairedPValue": _as_float(comparison["pairedTop50PValue"]),
        "properScoreStatus": proper_status,
        "meanLogLossImprovementVsCore": log_improvement,
        "meanBrierImprovementVsCore": brier_improvement,
        "stableBlocks": sum(value >= _RANDOM_BASELINE for value in blocks),
        "jointPassedBlocks": 0,
        "abstainedPeriods": int(cast(int, challenger["abstainedPeriods"])),
        "behavioralBoundaryMean": _as_float(boundary["mean"]),
        "behavioralBoundaryAllBlocksPositive": bool(boundary["allFixedBlocksPositive"]),
        "evidenceStatus": "retrospective_reused_development",
        "decision": "closed" if not bool(gate["passed"]) else "review_required",
        "source": f"reports/development/{version}_{lottery}_all_blocks_20260721.json",
    }


def _holm_adjust(rows: list[dict[str, object]]) -> None:
    ordered = sorted(
        enumerate(rows), key=lambda item: _as_float(item[1]["randomPValue"])
    )
    count = len(ordered)
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * _as_float(row["randomPValue"]))
        running = max(running, adjusted)
        rows[index]["holmAdjustedRandomPValue"] = running


def build_model_scoreboard(
    behavioral_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """构建固定成本总账；所有未过联合闸门的模型一律关闭。"""

    rows = [_normalized_locked_row(item) for item in _LOCKED_DIRECT_ROWS]
    seen: set[tuple[str, str]] = set()
    for report in behavioral_reports:
        row = _behavior_row(report)
        key = (str(row["modelVersion"]), str(row["lottery"]))
        if key in seen:
            raise ValueError("行为报告版本和彩种不得重复")
        seen.add(key)
        rows.append(row)
    _holm_adjust(rows)
    rows.sort(key=lambda row: (str(row["modelVersion"]), str(row["lottery"])))

    return {
        "schemaVersion": "digit_model_scoreboard_v1",
        "comparisonPolicy": {
            "objective": "direct_top50",
            "fixedCost": 50,
            "randomBaseline": _RANDOM_BASELINE,
            "winnerSelectionAllowed": False,
            "multipleComparisonCorrection": "Holm",
            "minimumEvidencePeriods": 6000,
            "requiredJointMetrics": [
                "LogLoss",
                "Brier",
                "Top50",
                "timeBlockStability",
            ],
        },
        "directTop50Evidence": rows,
        "incompleteEvidence": [dict(item) for item in _INCOMPLETE_EVIDENCE],
        "nonComparableDiagnostics": [dict(item) for item in _NONCOMPARABLE_DIAGNOSTICS],
        "consumedFrozenEvidence": {
            "status": "completed_failed_joint_gate",
            "periodsPerLottery": 500,
            "rerunAllowed": False,
            "markers": [
                "state/learned_ranker_v4/frozen_fc3d_once.marker.json",
                "state/learned_ranker_v4/frozen_pl3_once.marker.json",
            ],
        },
        "decision": {
            "selectedModel": None,
            "productionMode": "uniform_abstain",
            "userVisibleCandidates": [],
            "behaviorFamilyRetired": True,
            "coreProspectiveMode": "passive_frozen_parameters",
            "coreProspectiveCheckpoints": [50, 100, 200],
            "coreProspectiveRule": "50期只允许提前淘汰；100期首次正式判断；200期仍无联合优势则终止",
            "newHistoricalTuningAllowed": False,
        },
    }


def render_model_scoreboard_markdown(scoreboard: Mapping[str, object]) -> str:
    """生成面向审计的Markdown总账，不按命中率宣布赢家。"""

    rows = cast(Sequence[Mapping[str, object]], scoreboard["directTop50Evidence"])
    incomplete = cast(Sequence[Mapping[str, object]], scoreboard["incompleteEvidence"])
    diagnostics = cast(
        Sequence[Mapping[str, object]], scoreboard["nonComparableDiagnostics"]
    )
    decision = cast(Mapping[str, object], scoreboard["decision"])
    lines = [
        "# 数字彩模型统一证据总账",
        "",
        "> 固定成本直选Top50；历史结果只允许淘汰，不允许按局部高点挑赢家。",
        "",
        "## 直选Top50可比证据",
        "",
        "| 目标 | 模型 | 彩种 | 期数 | 命中 | 命中率 | 原始p | Holm校正p | 概率质量 | 稳定块 | 联合通过块 | 决策 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        stable = row.get("stableBlocks")
        lines.append(
            "| direct_top50 | {model} | {lottery} | {periods} | {hits} | "
            "{rate:.3%} | {p:.4f} | {adjusted:.4f} | {proper} | {stable} | "
            "{joint} | {decision} |".format(
                model=row["modelVersion"],
                lottery=row["lottery"],
                periods=row["periods"],
                hits=row["hits"],
                rate=_as_float(row["hitRate"]),
                p=_as_float(row["randomPValue"]),
                adjusted=_as_float(row["holmAdjustedRandomPValue"]),
                proper=row["properScoreStatus"],
                stable="—" if stable is None else stable,
                joint=row["jointPassedBlocks"],
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## 覆盖不足或事后形成、不得进入可比排名的证据",
            "",
            "| 模型 | 彩种 | 期数 | 命中 | 原因 | 决策 |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in incomplete:
        lines.append(
            "| {model} | {lottery} | {periods} | {hits} | {reason} | {decision} |".format(
                model=row["modelVersion"],
                lottery=row["lottery"],
                periods=row["periods"],
                hits=row["hits"],
                reason=row["reasonNotComparable"],
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## 不同成本、不可与直选Top50混排的诊断",
            "",
            "| 目标 | 彩种 | 期数 | 命中 | 基线率 | p值 | 决策 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in diagnostics:
        lines.append(
            "| {objective} | {lottery} | {periods} | {hits} | {baseline:.3%} | "
            "{p:.4f} | {decision} |".format(
                objective=row["objective"],
                lottery=row["lottery"],
                periods=row["periods"],
                hits=row["hits"],
                baseline=_as_float(row["weightedBaselineRate"]),
                p=_as_float(row["pValue"]),
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## 最终决策",
            "",
            "- **无模型入选**：全部模型未共同通过LogLoss、Brier、Top50与时间稳定性。",
            f"- 生产模式：`{decision['productionMode']}`。",
            "- 行为v1～v4：封存为负证据，不再在同一历史调参。",
            "- 核心前瞻检查点：50/100/200期；50期只允许提前淘汰。",
            "- 用户可见候选保持为空。",
            "",
        ]
    )
    return "\n".join(lines)


def write_model_scoreboard(scoreboard: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(scoreboard)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    payload["scoreboardSha256"] = hashlib.sha256(canonical).hexdigest()
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


__all__ = [
    "build_model_scoreboard",
    "render_model_scoreboard_markdown",
    "write_model_scoreboard",
]
