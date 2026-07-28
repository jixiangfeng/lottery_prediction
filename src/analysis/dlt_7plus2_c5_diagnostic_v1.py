# -*- coding: utf-8 -*-
"""大乐透C5在线Hedge的开发历史执行诊断；禁止读取v1 Frozen。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from src.analysis.dlt_7plus2_c5_hedge_v1 import (
    C5_BACK_EPSILON,
    C5_BACK_TAU,
    C5_FRONT_EPSILON,
    C5_FRONT_TAU,
    C5BlockPrediction,
    build_c5_output,
)
from src.analysis.dlt_7plus2_validation_v1 import (
    UNIFORM_BACK_BRIER,
    UNIFORM_BACK_SET_LL,
    UNIFORM_FRONT_BRIER,
    UNIFORM_FRONT_SET_LL,
    _control_utilities,
    builder_metrics,
    deterministic_random_controls,
)
from src.analysis.dlt_fixed_cardinality_v1 import FixedCardinalityDistribution
from src.analysis.dlt_ridge_candidates_v1 import BACK_SIZE, FRONT_SIZE, CandidateScores

V1_FROZEN_START = 2401


class DiagnosticDrawLike(Protocol):
    @property
    def issue(self) -> str: ...

    @property
    def front(self) -> Sequence[int]: ...

    @property
    def back(self) -> Sequence[int]: ...


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("诊断指标不能为空")
    return math.fsum(values) / len(values)


def _brier(marginals: Sequence[float], observed: Sequence[int]) -> float:
    selected = {value - 1 for value in observed}
    return math.fsum(
        (probability - float(index in selected)) ** 2
        for index, probability in enumerate(marginals)
    ) / len(marginals)


def _equal_logpool(
    components: tuple[CandidateScores, CandidateScores, CandidateScores],
) -> CandidateScores:
    return CandidateScores(
        tuple(
            math.fsum(component.front[index] for component in components) / 3.0
            for index in range(FRONT_SIZE)
        ),
        tuple(
            math.fsum(component.back[index] for component in components) / 3.0
            for index in range(BACK_SIZE)
        ),
    )


def _distributions(
    scores: CandidateScores,
) -> tuple[FixedCardinalityDistribution, FixedCardinalityDistribution]:
    return (
        FixedCardinalityDistribution(
            scores.front, 5, tau=C5_FRONT_TAU, epsilon=C5_FRONT_EPSILON
        ),
        FixedCardinalityDistribution(
            scores.back, 2, tau=C5_BACK_TAU, epsilon=C5_BACK_EPSILON
        ),
    )


def _self_hash(report: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in report.items() if key != "reportSha256"}
    payload = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_c5_diagnostic(
    draws: Sequence[DiagnosticDrawLike],
    predictions: Sequence[C5BlockPrediction],
    *,
    start_index: int,
    stop_index: int,
    protocol_sha256: str,
) -> dict[str, object]:
    """汇总已查看开发段；边界硬限制为v1 Frozen之前。"""

    if (
        start_index < 600
        or stop_index <= start_index
        or stop_index > V1_FROZEN_START
        or stop_index > len(draws)
        or len(predictions) != stop_index - start_index
        or len(protocol_sha256) != 64
    ):
        raise ValueError("C5诊断边界或协议摘要非法")
    expected_indices = tuple(range(start_index, stop_index))
    if tuple(item.target_index for item in predictions) != expected_indices:
        raise ValueError("C5诊断预测索引不连续")

    front_losses: list[float] = []
    back_losses: list[float] = []
    front_briers: list[float] = []
    back_briers: list[float] = []
    joint_improvements: list[float] = []
    c5_minus_c4_improvements: list[float] = []
    utilities: list[float] = []
    c4_utilities: list[float] = []
    random_utilities: list[float] = []
    front_hits: list[int] = []
    c4_front_hits: list[int] = []
    back_hits: list[int] = []
    c4_back_hits: list[int] = []
    weight_rows: list[tuple[float, float, float]] = []
    per_issue: list[dict[str, object]] = []

    uniform_joint = UNIFORM_FRONT_SET_LL + UNIFORM_BACK_SET_LL
    for prediction in predictions:
        draw = draws[prediction.target_index]
        observed_front = tuple(draw.front)
        observed_back = tuple(draw.back)
        front_distribution, back_distribution = _distributions(prediction.scores)
        front_loss = -front_distribution.log_probability(
            tuple(value - 1 for value in observed_front)
        )
        back_loss = -back_distribution.log_probability(
            tuple(value - 1 for value in observed_back)
        )
        joint_loss = front_loss + back_loss
        c4_scores = _equal_logpool(prediction.expert_scores)
        c4_front, c4_back = _distributions(c4_scores)
        c4_joint_loss = -c4_front.log_probability(
            tuple(value - 1 for value in observed_front)
        ) - c4_back.log_probability(tuple(value - 1 for value in observed_back))
        output = build_c5_output(prediction.scores)
        builder = builder_metrics(
            output.front, output.back, observed_front, observed_back
        )
        c4_output = build_c5_output(c4_scores)
        c4_builder = builder_metrics(
            c4_output.front, c4_output.back, observed_front, observed_back
        )
        controls = deterministic_random_controls(protocol_sha256, draw.issue)
        random_mean = float(
            _control_utilities(controls, observed_front, observed_back).mean()
        )

        front_losses.append(front_loss)
        back_losses.append(back_loss)
        front_briers.append(_brier(front_distribution.marginals, observed_front))
        back_briers.append(_brier(back_distribution.marginals, observed_back))
        joint_improvements.append(uniform_joint - joint_loss)
        c5_minus_c4_improvements.append(c4_joint_loss - joint_loss)
        utility = float(cast(float, builder["U"]))
        c4_utility = float(cast(float, c4_builder["U"]))
        utilities.append(utility)
        c4_utilities.append(c4_utility)
        random_utilities.append(random_mean)
        front_hits.append(int(cast(int, builder["HF"])))
        c4_front_hits.append(int(cast(int, c4_builder["HF"])))
        back_hits.append(int(cast(int, builder["HB"])))
        c4_back_hits.append(int(cast(int, c4_builder["HB"])))
        weight_rows.append(prediction.weights)
        per_issue.append(
            {
                "targetIndex": prediction.target_index,
                "fitCutoff": prediction.fit_cutoff,
                "issue": draw.issue,
                "weights": dict(zip(("C1", "C2", "C3"), prediction.weights)),
                "selectedFront": list(output.front),
                "selectedBack": list(output.back),
                "observedFront": list(observed_front),
                "observedBack": list(observed_back),
                "frontSetLogLoss": front_loss,
                "backSetLogLoss": back_loss,
                "jointLogLossImprovement": uniform_joint - joint_loss,
                "jointLogLossImprovementVsC4": c4_joint_loss - joint_loss,
                "builder21": builder,
                "C4": {
                    "selectedFront": list(c4_output.front),
                    "selectedBack": list(c4_output.back),
                    "builder21": c4_builder,
                },
                "random512MeanU": random_mean,
            }
        )

    complete_blocks = len(predictions) // 100
    joint_blocks = [
        _mean(joint_improvements[index * 100 : (index + 1) * 100])
        for index in range(complete_blocks)
    ]
    u_improvements = [
        value - random
        for value, random in zip(utilities, random_utilities, strict=True)
    ]
    u_blocks = [
        _mean(u_improvements[index * 100 : (index + 1) * 100])
        for index in range(complete_blocks)
    ]
    front_distribution_counts = Counter(front_hits)
    back_distribution_counts = Counter(back_hits)
    metrics: dict[str, object] = {
        "meanFrontSetLogLoss": _mean(front_losses),
        "meanBackSetLogLoss": _mean(back_losses),
        "meanJointLogLoss": _mean(front_losses) + _mean(back_losses),
        "meanJointLogLossImprovement": _mean(joint_improvements),
        "meanJointLogLossImprovementVsC4": _mean(c5_minus_c4_improvements),
        "meanFrontMarginalBrier": _mean(front_briers),
        "meanBackMarginalBrier": _mean(back_briers),
        "uniformFrontBrier": UNIFORM_FRONT_BRIER,
        "uniformBackBrier": UNIFORM_BACK_BRIER,
        "meanHF": _mean([float(value) for value in front_hits]),
        "meanHFImprovementVsC4": _mean(
            [
                float(value - control)
                for value, control in zip(front_hits, c4_front_hits, strict=True)
            ]
        ),
        "meanHB": _mean([float(value) for value in back_hits]),
        "meanHBImprovementVsC4": _mean(
            [
                float(value - control)
                for value, control in zip(back_hits, c4_back_hits, strict=True)
            ]
        ),
        "meanU": _mean(utilities),
        "meanUImprovementVsC4": _mean(
            [
                value - control
                for value, control in zip(utilities, c4_utilities, strict=True)
            ]
        ),
        "meanRandom512U": _mean(random_utilities),
        "meanUImprovementVsRandom512": _mean(u_improvements),
        "frontHitDistribution": {
            str(value): front_distribution_counts[value] for value in range(6)
        },
        "backHitDistribution": {
            str(value): back_distribution_counts[value] for value in range(3)
        },
        "jointBlockImprovements": joint_blocks,
        "uBlockImprovementsVsRandom512": u_blocks,
        "complete100Blocks": complete_blocks,
        "meanWeights": {
            f"C{index + 1}": _mean([row[index] for row in weight_rows])
            for index in range(3)
        },
        "minimumWeights": {
            f"C{index + 1}": min(row[index] for row in weight_rows)
            for index in range(3)
        },
        "maximumWeights": {
            f"C{index + 1}": max(row[index] for row in weight_rows)
            for index in range(3)
        },
    }
    portfolio_changed_count = sum(
        row["selectedFront"] != cast(Mapping[str, object], row["C4"])["selectedFront"]
        or row["selectedBack"] != cast(Mapping[str, object], row["C4"])["selectedBack"]
        for row in per_issue
    )
    report: dict[str, object] = {
        "schemaVersion": "dlt_7plus2_c5_diagnostic_v1",
        "protocolSha256": protocol_sha256,
        "evidenceStatus": "exploratory_reused_development",
        "independentEvidence": False,
        "formalGate": False,
        "formalOutput": "uniform_abstain",
        "autoPromotion": False,
        "portfolioChangedCountVsC4": portfolio_changed_count,
        "developmentDisposition": (
            "retired_no_portfolio_change"
            if portfolio_changed_count == 0
            else "future_only_pending_human_review"
        ),
        "boundary": {
            "startIndex": start_index,
            "stopIndexExclusive": stop_index,
            "count": len(predictions),
            "maximumConsumedIndex": stop_index - 1,
            "v1FrozenStartIndex": V1_FROZEN_START,
            "frozenRowsAccessed": 0,
        },
        "metrics": metrics,
        "perIssue": per_issue,
    }
    report["reportSha256"] = _self_hash(report)
    return report


__all__ = ["V1_FROZEN_START", "summarize_c5_diagnostic"]
