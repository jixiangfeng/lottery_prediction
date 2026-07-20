# -*- coding: utf-8 -*-
"""三层设计的实战准入、降级与淘汰状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StrategyStatus(str, Enum):
    RESEARCH = "research"
    OBSERVATION = "observation"
    ACTIVE = "active"
    DEMOTED = "demoted"
    RETIRED = "retired"


@dataclass(frozen=True)
class StrategyEvidence:
    search_lift: float
    validation_lift: float
    stable_validation_blocks: int
    validation_blocks: int
    mean_log_loss: float
    uniform_log_loss: float
    mean_brier_score: float
    uniform_brier_score: float
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.05
    frozen_test_evaluated: bool = False
    frozen_test_lift: float | None = None
    data_fingerprint: str = ""
    params_fingerprint: str = ""
    source_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.validation_blocks <= 0:
            raise ValueError("validation_blocks必须大于零")
        if not 0 <= self.stable_validation_blocks <= self.validation_blocks:
            raise ValueError("稳定块数量必须位于0到总块数")
        if self.frozen_test_evaluated != (self.frozen_test_lift is not None):
            raise ValueError("Frozen评估状态与lift必须一致")
        if not 0 <= self.expected_calibration_error <= 1:
            raise ValueError("ECE必须位于0-1")
        if not 0 <= self.maximum_calibration_error <= 1:
            raise ValueError("最大ECE阈值必须位于0-1")


@dataclass(frozen=True)
class StrategyDecision:
    status: StrategyStatus
    admitted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "admitted": self.admitted,
            "reasons": list(self.reasons),
        }


def _stable_enough(evidence: StrategyEvidence) -> bool:
    return evidence.stable_validation_blocks * 3 >= evidence.validation_blocks * 2


def decide_strategy_status(
    evidence: StrategyEvidence,
    *,
    previous_status: StrategyStatus = StrategyStatus.RESEARCH,
) -> StrategyDecision:
    """根据固定证据决定策略层级；Frozen未评估绝不进入active。"""

    reasons: list[str] = []
    if evidence.search_lift <= 1.0:
        reasons.append("Search未严格优于同成本随机")
    if evidence.validation_lift <= 1.0:
        reasons.append("Validation未严格优于同成本随机")
    if not _stable_enough(evidence):
        reasons.append("Validation稳定时间块不足2/3")
    if evidence.mean_log_loss > evidence.uniform_log_loss:
        reasons.append("logloss劣于均匀基线")
    if evidence.mean_brier_score > evidence.uniform_brier_score:
        reasons.append("Brier劣于均匀基线")
    if evidence.expected_calibration_error > evidence.maximum_calibration_error:
        reasons.append("ECE超过固定阈值")

    development_passed = not reasons
    if not evidence.frozen_test_evaluated:
        if previous_status is StrategyStatus.ACTIVE and not development_passed:
            return StrategyDecision(
                StrategyStatus.DEMOTED, False, tuple(reasons + ["活跃策略开发证据回落"])
            )
        if development_passed:
            return StrategyDecision(
                StrategyStatus.OBSERVATION,
                False,
                ("开发闸门通过，但Frozen Test尚未执行",),
            )
        status = (
            StrategyStatus.RETIRED
            if previous_status is StrategyStatus.DEMOTED
            else StrategyStatus.RESEARCH
        )
        return StrategyDecision(status, False, tuple(reasons))

    if evidence.frozen_test_lift is None or evidence.frozen_test_lift <= 1.0:
        reasons.append("Frozen Test未严格优于同成本随机")
    if not reasons:
        return StrategyDecision(StrategyStatus.ACTIVE, True, ("全部固定闸门通过",))
    if previous_status is StrategyStatus.ACTIVE:
        return StrategyDecision(StrategyStatus.DEMOTED, False, tuple(reasons))
    if previous_status is StrategyStatus.DEMOTED:
        return StrategyDecision(StrategyStatus.RETIRED, False, tuple(reasons))
    return StrategyDecision(StrategyStatus.RESEARCH, False, tuple(reasons))


__all__ = [
    "StrategyDecision",
    "StrategyEvidence",
    "StrategyStatus",
    "decide_strategy_status",
]
