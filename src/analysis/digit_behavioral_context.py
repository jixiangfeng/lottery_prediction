# -*- coding: utf-8 -*-
"""behavioral_context_v2：标准化行为风险的固定A/B/C挑战模型。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.analysis.digit_data import (
    canonical_digit_data_sha256,
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    BEHAVIORAL_FEATURE_NAMES,
    BEHAVIORAL_FEATURE_SEMANTICS,
    BEHAVIORAL_RISK_HALF_LIFE,
    FEATURE_NAMES,
    SHAPE_PRIORS,
    LearnedFeatureConfig,
)
from src.analysis.digit_online_gradient import (
    OnlineGradientConfig,
    OnlineGradientReport,
    run_online_gradient_research,
)
from src.lotteries.base import LotteryRule


def behavioral_context_source_fingerprint() -> str:
    """计算行为挑战器及其直接依赖的源码指纹。"""

    directory = Path(__file__).resolve().parent
    paths = (
        directory / "digit_daily_policy.py",
        directory / "digit_learned_features.py",
        directory / "digit_online_gradient.py",
        directory / "digit_behavioral_context.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BehavioralContextConfig:
    online: OnlineGradientConfig
    feature_config: LearnedFeatureConfig = LearnedFeatureConfig()
    behavioral_feature_names: tuple[str, ...] = BEHAVIORAL_FEATURE_NAMES
    behavioral_l2_multiplier: float = 2.0
    paired_block_size: int = 10
    paired_permutations: int = 9999
    random_seed: int = 20260721
    minimum_periods: int = 500
    paired_p_threshold: float = 0.01
    minimum_random_lift: float = 1.25
    maximum_top50_triples: int = 1
    maximum_shape_total_variation: float = 0.10
    fixed_block_size: int = 500

    def __post_init__(self) -> None:
        if not self.behavioral_feature_names or len(
            set(self.behavioral_feature_names)
        ) != len(self.behavioral_feature_names):
            raise ValueError("行为特征子集必须非空且不得重复")
        if any(
            name not in BEHAVIORAL_FEATURE_NAMES
            for name in self.behavioral_feature_names
        ):
            raise ValueError("行为特征子集包含未知特征")
        if self.behavioral_l2_multiplier <= 0:
            raise ValueError("行为特征L2倍率必须为正")
        if self.paired_block_size <= 0 or self.paired_permutations <= 0:
            raise ValueError("配对块和置换次数必须为正")
        if self.minimum_periods <= 0 or not 0 < self.paired_p_threshold < 1:
            raise ValueError("验收样本和p值门槛无效")
        if self.minimum_random_lift <= 1 or self.maximum_top50_triples < 0:
            raise ValueError("提升和形态门槛无效")
        if not 0 <= self.maximum_shape_total_variation <= 1:
            raise ValueError("形态总变差门槛必须位于0到1")
        if self.fixed_block_size <= 0:
            raise ValueError("固定时间块大小必须为正")


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


def _summary(
    report: OnlineGradientReport,
    *,
    maximum_top50_triples: int,
    maximum_shape_total_variation: float,
    fixed_block_size: int,
) -> dict[str, object]:
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
    shape_total_variation = 0.5 * sum(
        abs(shape_means[name] / report.config.direct_top_k - SHAPE_PRIORS[name])
        for name in ("组六", "组三", "豹子")
    )
    feature_attribution = {
        name: {
            "meanBoundaryContribution": float(
                np.mean([item.boundary_contributions[name] for item in periods])
            ),
            "meanAbsoluteBoundaryContribution": float(
                np.mean([abs(item.boundary_contributions[name]) for item in periods])
            ),
            "negativeBoundaryPeriods": sum(
                item.boundary_contributions[name] < 0 for item in periods
            ),
            "positiveBoundaryPeriods": sum(
                item.boundary_contributions[name] > 0 for item in periods
            ),
        }
        for name in report.config.feature_names
    }
    behavioral_names = tuple(
        name for name in BEHAVIORAL_FEATURE_NAMES if name in report.config.feature_names
    )
    behavioral_boundary = np.asarray(
        [
            sum(item.boundary_contributions[name] for name in behavioral_names)
            for item in periods
        ],
        dtype=float,
    )
    behavioral_fixed_blocks = [
        float(behavioral_boundary[start : start + fixed_block_size].mean())
        for start in range(0, len(periods), fixed_block_size)
        if len(periods[start : start + fixed_block_size]) == fixed_block_size
    ]
    fixed_block_rates = [
        float(
            np.mean(
                [
                    item.research_direct_hit
                    for item in periods[start : start + fixed_block_size]
                ]
            )
        )
        for start in range(0, len(periods), fixed_block_size)
        if len(periods[start : start + fixed_block_size]) == fixed_block_size
    ]
    random_baseline = report.config.direct_top_k / 1000.0
    return {
        "periods": len(periods),
        "hits": hits,
        "hitRate": hit_rate,
        "randomBaseline": random_baseline,
        "randomLift": hit_rate / random_baseline,
        "top50PValue": float(
            binom.sf(
                hits - 1,
                len(periods),
                random_baseline,
            )
        ),
        "meanLogLoss": float(np.mean([item.research_log_loss for item in periods])),
        "meanBrier": float(np.mean([item.research_brier for item in periods])),
        "timeBlockHitRates": list(time_blocks),
        "fixedBlockSize": fixed_block_size,
        "fixedBlockHitRates": fixed_block_rates,
        "fixed500BlockHitRates": (fixed_block_rates if fixed_block_size == 500 else []),
        "completeFixedBlocks": len(fixed_block_rates),
        "periodsOutsideCompleteFixedBlocks": len(periods) % fixed_block_size,
        "allTimeBlocksAtOrAboveRandom": all(
            value >= random_baseline for value in time_blocks
        ),
        "allFixedBlocksAtOrAboveRandom": bool(fixed_block_rates)
        and all(value >= random_baseline for value in fixed_block_rates),
        "meanTop50ShapeCounts": shape_means,
        "maximumTop50Triples": max_triples,
        "maximumTop50TriplesAllowed": maximum_top50_triples,
        "shapeDistributionTotalVariation": shape_total_variation,
        "maximumShapeTotalVariationAllowed": maximum_shape_total_variation,
        "shapeHealthPassed": max_triples <= maximum_top50_triples
        and shape_total_variation <= maximum_shape_total_variation,
        "dailyCandidatePolicyApplied": report.config.daily_candidate_policy,
        "abstainedPeriods": sum(item.abstained for item in periods),
        "finalWeights": periods[-1].weights_after,
        "featureAttribution": feature_attribution,
        "behavioralBoundaryContribution": (
            {
                "mean": float(behavioral_boundary.mean()),
                "fixedBlockMeans": behavioral_fixed_blocks,
                "allFixedBlocksPositive": bool(behavioral_fixed_blocks)
                and all(value > 0 for value in behavioral_fixed_blocks),
            }
            if behavioral_names
            else None
        ),
    }


def _paired_comparison(
    baseline: OnlineGradientReport,
    challenger: OnlineGradientReport,
    config: BehavioralContextConfig,
    *,
    seed_offset: int,
) -> dict[str, object]:
    log_improvements = np.asarray(
        [
            left.research_log_loss - right.research_log_loss
            for left, right in zip(baseline.periods, challenger.periods)
        ]
    )
    brier_improvements = np.asarray(
        [
            left.research_brier - right.research_brier
            for left, right in zip(baseline.periods, challenger.periods)
        ]
    )
    period_pairs = tuple(zip(baseline.periods, challenger.periods))
    top50_gained = sum(
        not left.research_direct_hit and right.research_direct_hit
        for left, right in period_pairs
    )
    top50_lost = sum(
        left.research_direct_hit and not right.research_direct_hit
        for left, right in period_pairs
    )
    discordant_periods = top50_gained + top50_lost
    return {
        "pairedPeriods": len(log_improvements),
        "meanLogLossImprovement": float(np.mean(log_improvements)),
        "pairedLogLossPValue": _paired_block_sign_pvalue(
            log_improvements,
            block_size=config.paired_block_size,
            permutations=config.paired_permutations,
            seed=config.random_seed + seed_offset,
        ),
        "meanBrierImprovement": float(np.mean(brier_improvements)),
        "pairedBrierPValue": _paired_block_sign_pvalue(
            brier_improvements,
            block_size=config.paired_block_size,
            permutations=config.paired_permutations,
            seed=config.random_seed + seed_offset + 1,
        ),
        "pairedBlockSize": config.paired_block_size,
        "pairedPermutations": config.paired_permutations,
        "top50GainedPeriods": top50_gained,
        "top50LostPeriods": top50_lost,
        "top50DiscordantPeriods": discordant_periods,
        "pairedTop50PValue": (
            float(binom.sf(top50_gained - 1, discordant_periods, 0.5))
            if discordant_periods
            else 1.0
        ),
        "meanRawRankImprovement": float(
            np.mean(
                [
                    left.research_rank - right.research_rank
                    for left, right in period_pairs
                ]
            )
        ),
        "baselineActivePeriods": sum(not item.abstained for item in baseline.periods),
        "challengerActivePeriods": sum(
            not item.abstained for item in challenger.periods
        ),
        "commonActivePeriods": sum(
            not left.abstained and not right.abstained for left, right in period_pairs
        ),
    }


def run_behavioral_context_challenge(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: BehavioralContextConfig,
) -> dict[str, object]:
    """运行A核心、B自由行为和预注册C单调行为三个固定组。"""
    base = replace(
        config.online,
        feature_names=FEATURE_NAMES,
        daily_candidate_policy=True,
        maximum_top50_triples=config.maximum_top50_triples,
    )
    selected_behavior_names = config.behavioral_feature_names
    behavior_names = (*FEATURE_NAMES, *selected_behavior_names)
    behavior_multipliers = tuple(base.feature_l2_multipliers) + tuple(
        (name, config.behavioral_l2_multiplier) for name in selected_behavior_names
    )
    unconstrained = replace(
        base,
        feature_names=behavior_names,
        feature_l2_multipliers=behavior_multipliers,
    )
    monotonic = replace(
        unconstrained,
        nonpositive_features=selected_behavior_names,
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
        unconstrained,
        config.feature_config,
        frozen_test_read=False,
    )
    report_c = run_online_gradient_research(
        history,
        rule,
        monotonic,
        config.feature_config,
        frozen_test_read=False,
    )
    target_issues = tuple(item.target_issue for item in report_a.periods)
    if any(
        tuple(item.target_issue for item in report.periods) != target_issues
        for report in (report_b, report_c)
    ):
        raise RuntimeError("A/B/C目标期不一致")

    summary_a = _summary(
        report_a,
        maximum_top50_triples=config.maximum_top50_triples,
        maximum_shape_total_variation=config.maximum_shape_total_variation,
        fixed_block_size=config.fixed_block_size,
    )
    summary_b = _summary(
        report_b,
        maximum_top50_triples=config.maximum_top50_triples,
        maximum_shape_total_variation=config.maximum_shape_total_variation,
        fixed_block_size=config.fixed_block_size,
    )
    summary_c = _summary(
        report_c,
        maximum_top50_triples=config.maximum_top50_triples,
        maximum_shape_total_variation=config.maximum_shape_total_variation,
        fixed_block_size=config.fixed_block_size,
    )
    summary_a["variant"] = "core_v4"
    summary_b["variant"] = "standardized_unconstrained_behavior"
    summary_c["variant"] = "standardized_monotonic_behavior"
    comparison_b = _paired_comparison(report_a, report_b, config, seed_offset=0)
    comparison_c = _paired_comparison(report_a, report_c, config, seed_offset=10)
    first_eligible_index = (
        config.online.warmup_history
        + config.online.search_lookback
        + config.online.validation_lookback
    )
    available_outer_periods = config.online.development_end - first_eligible_index
    complete_development_periods = (
        available_outer_periods // config.fixed_block_size * config.fixed_block_size
    )
    all_development_blocks_evaluated = (
        complete_development_periods > 0
        and config.online.outer_periods == complete_development_periods
        and config.online.development_end - config.online.outer_periods
        == first_eligible_index
    )
    gate_reasons: list[str] = []
    if cast(int, summary_c["periods"]) < config.minimum_periods:
        gate_reasons.append("有效样本不足")
    if cast(int, summary_c["abstainedPeriods"]) > 0:
        gate_reasons.append("存在放弃期，不能用原始研究排序晋级")
    if (
        cast(float, comparison_c["meanLogLossImprovement"]) <= 0
        or cast(float, comparison_c["pairedLogLossPValue"]) >= config.paired_p_threshold
    ):
        gate_reasons.append("配对LogLoss未达到单侧p门槛")
    if (
        cast(float, comparison_c["meanBrierImprovement"]) <= 0
        or cast(float, comparison_c["pairedBrierPValue"]) >= config.paired_p_threshold
    ):
        gate_reasons.append("配对Brier未达到单侧p门槛")
    if cast(float, summary_c["top50PValue"]) >= config.paired_p_threshold:
        gate_reasons.append("Top50未达到单侧p门槛")
    if (
        cast(int, comparison_c["top50GainedPeriods"])
        <= cast(int, comparison_c["top50LostPeriods"])
        or cast(float, comparison_c["pairedTop50PValue"]) >= config.paired_p_threshold
    ):
        gate_reasons.append("配对Top50增量未达到单侧p门槛")
    if cast(float, summary_c["randomLift"]) < config.minimum_random_lift:
        gate_reasons.append("相对随机提升不足25%")
    if not bool(summary_c["allTimeBlocksAtOrAboveRandom"]):
        gate_reasons.append("存在低于随机的时间块")
    if not bool(summary_c["allFixedBlocksAtOrAboveRandom"]):
        gate_reasons.append("完整固定时间块未全部达到随机基线")
    if not all_development_blocks_evaluated:
        gate_reasons.append("未覆盖开发区全部完整固定时间块")
    if not bool(summary_c["shapeHealthPassed"]):
        gate_reasons.append("Top50形态异常集中")
    behavioral_boundary_payload = cast(
        dict[str, object], summary_c["behavioralBoundaryContribution"]
    )
    if cast(float, behavioral_boundary_payload["mean"]) <= 0 or not bool(
        behavioral_boundary_payload["allFixedBlocksPositive"]
    ):
        gate_reasons.append("行为特征Top50边界贡献为负或跨块不稳定")

    target_issue_text = "\n".join(target_issues)
    evaluation_history = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    ).iloc[: config.online.development_end]
    return {
        "modelVersion": "behavioral_context_v2",
        "evaluationKind": "development_fixed_abc_behavioral_challenge",
        "evidenceStatus": "exploratory_reused_development",
        "lottery": rule.code,
        "dataSha256": canonical_digit_data_sha256(evaluation_history, rule),
        "sourceFingerprint": behavioral_context_source_fingerprint(),
        "frozenTestRead": False,
        "currentDailyModelReplaced": False,
        "behavioralFeatures": list(selected_behavior_names),
        "behavioralFeatureSemantics": {
            name: BEHAVIORAL_FEATURE_SEMANTICS[name] for name in selected_behavior_names
        },
        "behavioralFeatureProfile": (
            "all" if selected_behavior_names == BEHAVIORAL_FEATURE_NAMES else "subset"
        ),
        "behavioralFeatureInitialWeight": 0.0,
        "behavioralFeatureL2Multiplier": config.behavioral_l2_multiplier,
        "behavioralFeatureNormalization": "per_query_standard_z",
        "behavioralRiskHalfLifePeriods": BEHAVIORAL_RISK_HALF_LIFE,
        "primaryChallengerGroup": "C",
        "winnerSelectionAllowed": False,
        "candidatePolicy": {
            "excludeLatestExact": True,
            "topK": config.online.direct_top_k,
            "maximumTop50Triples": config.maximum_top50_triples,
        },
        "developmentCoverage": {
            "availableOuterPeriods": available_outer_periods,
            "completeDevelopmentPeriods": complete_development_periods,
            "evaluatedOuterPeriods": config.online.outer_periods,
            "expectedFirstTargetIndex": first_eligible_index,
            "actualFirstTargetIndex": (
                config.online.development_end - config.online.outer_periods
            ),
            "allCompleteFixedBlocksEvaluated": all_development_blocks_evaluated,
        },
        "targetIssuesSha256": hashlib.sha256(target_issue_text.encode()).hexdigest(),
        "groups": {
            "A": summary_a,
            "B": summary_b,
            "C": summary_c,
            "D": {"status": "trial_data_unavailable"},
        },
        "comparisons": {
            "BvsA": comparison_b,
            "CvsA": comparison_c,
        },
        "comparison": comparison_c,
        "gate": {
            "passed": not gate_reasons,
            "minimumPeriods": config.minimum_periods,
            "pairedPThreshold": config.paired_p_threshold,
            "minimumRandomLift": config.minimum_random_lift,
            "requiredTimeBlocks": 3,
            "fixedBlockSize": config.fixed_block_size,
            "maximumTop50Triples": config.maximum_top50_triples,
            "maximumShapeTotalVariation": config.maximum_shape_total_variation,
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
    "behavioral_context_source_fingerprint",
    "run_behavioral_context_challenge",
    "write_behavioral_context_report",
]
