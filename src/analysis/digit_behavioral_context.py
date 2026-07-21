# -*- coding: utf-8 -*-
"""behavioral_context_v1：近期行为特征的独立A/B挑战模型。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.analysis.digit_learned_features import (
    BEHAVIORAL_FEATURE_NAMES,
    FEATURE_NAMES,
    LearnedFeatureConfig,
)
from src.analysis.digit_online_gradient import (
    OnlineGradientConfig,
    OnlineGradientReport,
    run_online_gradient_research,
)
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class BehavioralContextConfig:
    online: OnlineGradientConfig
    feature_config: LearnedFeatureConfig = LearnedFeatureConfig()
    behavioral_l2_multiplier: float = 10.0
    paired_block_size: int = 10
    paired_permutations: int = 9999
    random_seed: int = 20260721
    minimum_periods: int = 500
    paired_p_threshold: float = 0.01
    minimum_random_lift: float = 1.25
    maximum_top50_triples: int = 1

    def __post_init__(self) -> None:
        if self.behavioral_l2_multiplier <= 0:
            raise ValueError("行为特征L2倍率必须为正")
        if self.paired_block_size <= 0 or self.paired_permutations <= 0:
            raise ValueError("配对块和置换次数必须为正")
        if self.minimum_periods <= 0 or not 0 < self.paired_p_threshold < 1:
            raise ValueError("验收样本和p值门槛无效")
        if self.minimum_random_lift <= 1 or self.maximum_top50_triples < 0:
            raise ValueError("提升和形态门槛无效")


def _paired_block_sign_pvalue(
    improvements: np.ndarray,
    *,
    block_size: int,
    permutations: int,
    seed: int,
) -> float:
    values = np.asarray(improvements, dtype=float)
    blocks = np.asarray(
        [
            values[start : start + block_size].mean()
            for start in range(0, len(values), block_size)
        ]
    )
    observed = float(blocks.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=len(blocks))
        exceedances += float(np.mean(blocks * signs)) >= observed
    return (exceedances + 1.0) / (permutations + 1.0)


def _summary(report: OnlineGradientReport) -> dict[str, object]:
    periods = report.periods
    hits = sum(item.research_direct_hit for item in periods)
    hit_rate = hits / len(periods)
    time_blocks = tuple(
        float(np.mean([item.research_direct_hit for item in block]))
        for block in np.array_split(np.asarray(periods, dtype=object), 3)
        if len(block)
    )
    shape_means = {
        name: float(np.mean([item.top50_shape_counts[name] for item in periods]))
        for name in ("组六", "组三", "豹子")
    }
    max_triples = max(item.top50_shape_counts["豹子"] for item in periods)
    fixed_500_block_rates = [
        float(
            np.mean([item.research_direct_hit for item in periods[start : start + 500]])
        )
        for start in range(0, len(periods), 500)
        if len(periods[start : start + 500]) == 500
    ]
    return {
        "periods": len(periods),
        "hits": hits,
        "hitRate": hit_rate,
        "randomBaseline": report.config.direct_top_k / 1000.0,
        "randomLift": hit_rate / (report.config.direct_top_k / 1000.0),
        "top50PValue": float(
            binom.sf(
                hits - 1,
                len(periods),
                report.config.direct_top_k / 1000.0,
            )
        ),
        "meanLogLoss": float(np.mean([item.research_log_loss for item in periods])),
        "meanBrier": float(np.mean([item.research_brier for item in periods])),
        "timeBlockHitRates": list(time_blocks),
        "fixed500BlockHitRates": fixed_500_block_rates,
        "allTimeBlocksAtOrAboveRandom": all(
            value >= report.config.direct_top_k / 1000.0 for value in time_blocks
        ),
        "meanTop50ShapeCounts": shape_means,
        "maximumTop50Triples": max_triples,
        "shapeHealthPassed": max_triples <= 1,
        "abstainedPeriods": sum(item.abstained for item in periods),
        "finalWeights": periods[-1].weights_after,
    }


def run_behavioral_context_challenge(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: BehavioralContextConfig,
) -> dict[str, object]:
    """在完全相同目标期上运行A（当前）与B（当前+近期行为）。"""
    base = replace(config.online, feature_names=FEATURE_NAMES)
    behavior_names = (*FEATURE_NAMES, *BEHAVIORAL_FEATURE_NAMES)
    behavior_multipliers = tuple(base.feature_l2_multipliers) + tuple(
        (name, config.behavioral_l2_multiplier) for name in BEHAVIORAL_FEATURE_NAMES
    )
    behavior = replace(
        config.online,
        feature_names=behavior_names,
        feature_l2_multipliers=behavior_multipliers,
    )
    report_a = run_online_gradient_research(
        history,
        rule,
        base,
        config.feature_config,
        frozen_test_read=False,
    )
    report_b = run_online_gradient_research(
        history,
        rule,
        behavior,
        config.feature_config,
        frozen_test_read=False,
    )
    if tuple(item.target_issue for item in report_a.periods) != tuple(
        item.target_issue for item in report_b.periods
    ):
        raise RuntimeError("A/B目标期不一致")

    log_improvements = np.asarray(
        [
            left.research_log_loss - right.research_log_loss
            for left, right in zip(report_a.periods, report_b.periods)
        ]
    )
    brier_improvements = np.asarray(
        [
            left.research_brier - right.research_brier
            for left, right in zip(report_a.periods, report_b.periods)
        ]
    )
    log_p = _paired_block_sign_pvalue(
        log_improvements,
        block_size=config.paired_block_size,
        permutations=config.paired_permutations,
        seed=config.random_seed,
    )
    brier_p = _paired_block_sign_pvalue(
        brier_improvements,
        block_size=config.paired_block_size,
        permutations=config.paired_permutations,
        seed=config.random_seed + 1,
    )
    summary_a = _summary(report_a)
    summary_b = _summary(report_b)
    gate_reasons: list[str] = []
    if cast(int, summary_b["periods"]) < config.minimum_periods:
        gate_reasons.append("有效样本不足")
    if float(np.mean(log_improvements)) <= 0 or log_p >= config.paired_p_threshold:
        gate_reasons.append("配对LogLoss未达到单侧p门槛")
    if float(np.mean(brier_improvements)) <= 0 or brier_p >= config.paired_p_threshold:
        gate_reasons.append("配对Brier未达到单侧p门槛")
    if cast(float, summary_b["top50PValue"]) >= config.paired_p_threshold:
        gate_reasons.append("Top50未达到单侧p门槛")
    if cast(float, summary_b["randomLift"]) < config.minimum_random_lift:
        gate_reasons.append("相对随机提升不足25%")
    if not bool(summary_b["allTimeBlocksAtOrAboveRandom"]):
        gate_reasons.append("存在低于随机的时间块")
    if not bool(summary_b["shapeHealthPassed"]):
        gate_reasons.append("Top50形态异常集中")

    target_issues = "\n".join(item.target_issue for item in report_b.periods)
    return {
        "modelVersion": "behavioral_context_v1",
        "evaluationKind": "development_paired_behavioral_challenge",
        "evidenceStatus": "exploratory_reused_development",
        "lottery": rule.code,
        "frozenTestRead": False,
        "currentDailyModelReplaced": False,
        "behavioralFeatures": list(BEHAVIORAL_FEATURE_NAMES),
        "behavioralFeatureInitialWeight": 0.0,
        "behavioralFeatureL2Multiplier": config.behavioral_l2_multiplier,
        "targetIssuesSha256": hashlib.sha256(target_issues.encode()).hexdigest(),
        "groups": {
            "A": summary_a,
            "B": summary_b,
            "C": {"status": "trial_data_unavailable"},
            "D": {"status": "trial_data_unavailable"},
        },
        "comparison": {
            "pairedPeriods": len(log_improvements),
            "meanLogLossImprovement": float(np.mean(log_improvements)),
            "pairedLogLossPValue": log_p,
            "meanBrierImprovement": float(np.mean(brier_improvements)),
            "pairedBrierPValue": brier_p,
            "pairedBlockSize": config.paired_block_size,
            "pairedPermutations": config.paired_permutations,
        },
        "gate": {
            "passed": not gate_reasons,
            "minimumPeriods": config.minimum_periods,
            "pairedPThreshold": config.paired_p_threshold,
            "minimumRandomLift": config.minimum_random_lift,
            "requiredTimeBlocks": 3,
            "maximumTop50Triples": config.maximum_top50_triples,
            "reasons": gate_reasons,
            "newShadowStateAllowed": not gate_reasons,
        },
    }


def write_behavioral_context_report(
    report: dict[str, object], path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["reportSha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


__all__ = [
    "BehavioralContextConfig",
    "run_behavioral_context_challenge",
    "write_behavioral_context_report",
]
