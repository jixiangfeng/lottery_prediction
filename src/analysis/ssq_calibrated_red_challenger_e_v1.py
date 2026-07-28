# -*- coding: utf-8 -*-
# mypy: disable-error-code="arg-type,call-overload"
"""双色球 Challenger E1：固定校准红球在线模型与独立当前报告。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, cast

from src.analysis.ssq_ensemble_v1 import RedMarginalExpert, RedPairExpert
from src.analysis.ssq_history import SSQDraw, load_official_history_csv
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    protocol_sha256 as builder_protocol_sha256,
)

MODEL_NAME = "ssq_calibrated_red_challenger_e_v1"
SCHEMA_VERSION = "ssq_challenger_e_v1"
RED_COUNT = 33
FEATURE_COUNT = 5
WARMUP_DRAWS = 120
UNIFORM_RED_PROBABILITY = 6.0 / RED_COUNT
ADAGRAD_INITIAL_ACCUMULATOR = 1e-6
LEARNING_RATE = 0.05
L2_PENALTY = 0.01
EPSILON = 1e-8
INTERCEPT_LOWER = -30.0
INTERCEPT_UPPER = 30.0
INTERCEPT_ITERATIONS = 80
INTERCEPT_TOLERANCE = 1e-12
FUTURE_HORIZON = 500

FUTURE_PROTOCOL: dict[str, object] = {
    "horizon": FUTURE_HORIZON,
    "evaluationTiming": "only_at_500_completed_periods",
    "safety": {
        "redLogLoss": "E<D8",
        "redBrier": "E<=D8",
        "required": "both",
    },
    "efficacy": {
        "primary": "paired_mean_red8_overlap_E_minus_D8_greater_than_0",
        "test": "one_sided",
        "alpha": 0.025,
        "blocks": {
            "count": 5,
            "size": 100,
            "requireAtLeastNonnegative": 4,
            "minimumAnyBlock": -0.05,
        },
        "secondary": "red5_rate_E>=D8",
        "recordOnly": "red6",
    },
    "promotion": {
        "automatic": False,
        "humanReviewRequired": True,
        "failureStatus": "uniform_abstain",
    },
}

PROTOCOL: dict[str, object] = {
    "model": MODEL_NAME,
    "purpose": "independent_retrospective_and_current_research_only",
    "features": [
        "EWMA30 deviation clip((p-6/33)/0.05,-3,3)",
        "EWMA120 deviation clip((p-6/33)/0.05,-3,3)",
        "log1p(min(gap,60))/log(61)-0.5",
        "prior draw indicator-6/33",
        "mean prior RedPairExpert120 modifier versus prior red6",
    ],
    "parameters": {
        "fixedEffects": 33,
        "fixedEffectsConstraint": "zero_mean_after_each_update",
        "beta": FEATURE_COUNT,
        "optimizer": "AdaGrad",
        "initialAccumulator": ADAGRAD_INITIAL_ACCUMULATOR,
        "eta": LEARNING_RATE,
        "l2": L2_PENALTY,
        "epsilon": EPSILON,
        "warmup": WARMUP_DRAWS,
    },
    "calibration": {
        "score": "a_i+beta*x_i",
        "intercept": "bisection",
        "bounds": [INTERCEPT_LOWER, INTERCEPT_UPPER],
        "iterations": INTERCEPT_ITERATIONS,
        "tolerance": INTERCEPT_TOLERANCE,
        "constraint": "sum(sigmoid(intercept+score_i))=6",
    },
    "onlineOrder": (
        "lock prior-only prediction/features; mean gradients and AdaGrad; "
        "project fixed effects; update feature states"
    ),
    "fixed": {"cliOverrides": False, "gridSearch": False, "retries": 0},
    "futureProtocol": FUTURE_PROTOCOL,
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(payload: object) -> str:
    """返回 JSON 规范化 SHA-256。"""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def protocol_sha256() -> str:
    """返回 E1 固定协议摘要。"""

    return sha256_payload(PROTOCOL)


def future_protocol_sha256() -> str:
    """返回未来 500 期协议摘要；本模块不实现未来链。"""

    return sha256_payload(FUTURE_PROTOCOL)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _calibrated_probabilities(scores: Sequence[float]) -> tuple[float, list[float]]:
    lower = INTERCEPT_LOWER
    upper = INTERCEPT_UPPER
    intercept = 0.0
    for _ in range(INTERCEPT_ITERATIONS):
        intercept = (lower + upper) / 2.0
        probabilities = [_sigmoid(intercept + score) for score in scores]
        difference = sum(probabilities) - 6.0
        if abs(difference) <= INTERCEPT_TOLERANCE:
            return intercept, probabilities
        if difference > 0.0:
            upper = intercept
        else:
            lower = intercept
    intercept = (lower + upper) / 2.0
    return intercept, [_sigmoid(intercept + score) for score in scores]


@dataclass(frozen=True)
class LockedPrediction:
    """一次严格前序预测锁，必须先消费后才能再次预测。"""

    probabilities: tuple[float, ...]
    features: tuple[tuple[float, ...], ...]
    intercept: float
    state_fingerprint: str
    prediction_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        """转换为可序列化字典。"""

        return {
            "probabilities": list(self.probabilities),
            "features": [list(row) for row in self.features],
            "intercept": self.intercept,
            "stateFingerprint": self.state_fingerprint,
            "predictionFingerprint": self.prediction_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LockedPrediction:
        """从序列化字典恢复预测锁。"""

        probabilities = tuple(
            float(value) for value in cast(list[object], payload["probabilities"])
        )
        features = tuple(
            tuple(float(value) for value in cast(list[object], row))
            for row in cast(list[object], payload["features"])
        )
        return cls(
            probabilities=probabilities,
            features=features,
            intercept=float(payload["intercept"]),
            state_fingerprint=str(payload["stateFingerprint"]),
            prediction_fingerprint=str(payload["predictionFingerprint"]),
        )


@dataclass
class CalibratedRedChallengerEState:
    """E1 完整在线状态；预测特征只允许来自前序开奖。"""

    fixed_effects: list[float] = field(default_factory=lambda: [0.0] * RED_COUNT)
    beta: list[float] = field(default_factory=lambda: [0.0] * FEATURE_COUNT)
    fixed_accumulators: list[float] = field(
        default_factory=lambda: [ADAGRAD_INITIAL_ACCUMULATOR] * RED_COUNT
    )
    beta_accumulators: list[float] = field(
        default_factory=lambda: [ADAGRAD_INITIAL_ACCUMULATOR] * FEATURE_COUNT
    )
    ewma30: RedMarginalExpert = field(default_factory=lambda: RedMarginalExpert(30))
    ewma120: RedMarginalExpert = field(default_factory=lambda: RedMarginalExpert(120))
    pair120: RedPairExpert = field(default_factory=RedPairExpert)
    last_seen_period: list[int] = field(default_factory=lambda: [-1] * RED_COUNT)
    prior_red: tuple[int, ...] = ()
    periods_seen: int = 0
    pending_prediction: LockedPrediction | None = None

    def _feature_matrix(self) -> tuple[tuple[float, ...], ...]:
        ewma30 = self.ewma30.probabilities()
        ewma120 = self.ewma120.probabilities()
        pair_modifiers = self.pair120.modifiers()
        prior = set(self.prior_red)
        rows: list[tuple[float, ...]] = []
        for ball in range(1, RED_COUNT + 1):
            last_seen = self.last_seen_period[ball - 1]
            gap = self.periods_seen if last_seen < 0 else self.periods_seen - last_seen
            pair_values = [
                pair_modifiers[
                    _pair_index(min(ball, prior_ball), max(ball, prior_ball))
                ]
                for prior_ball in self.prior_red
                if prior_ball != ball
            ]
            rows.append(
                (
                    max(
                        -3.0,
                        min(
                            3.0,
                            (ewma30[ball - 1] - UNIFORM_RED_PROBABILITY) / 0.05,
                        ),
                    ),
                    max(
                        -3.0,
                        min(
                            3.0,
                            (ewma120[ball - 1] - UNIFORM_RED_PROBABILITY) / 0.05,
                        ),
                    ),
                    math.log1p(min(gap, 60)) / math.log(61.0) - 0.5,
                    (1.0 if ball in prior else 0.0) - UNIFORM_RED_PROBABILITY,
                    sum(pair_values) / len(pair_values) if pair_values else 0.0,
                )
            )
        return tuple(rows)

    def state_payload(self, include_pending: bool = True) -> dict[str, object]:
        """返回完整稳定序列化状态。"""

        payload: dict[str, object] = {
            "model": MODEL_NAME,
            "fixedEffects": self.fixed_effects,
            "beta": self.beta,
            "fixedAccumulators": self.fixed_accumulators,
            "betaAccumulators": self.beta_accumulators,
            "ewma30": {
                "counts": self.ewma30.counts,
                "effectiveDraws": self.ewma30.effective_draws,
            },
            "ewma120": {
                "counts": self.ewma120.counts,
                "effectiveDraws": self.ewma120.effective_draws,
            },
            "pair120": {
                "counts": self.pair120.counts,
                "effectiveDraws": self.pair120.effective_draws,
            },
            "lastSeenPeriod": self.last_seen_period,
            "priorRed": list(self.prior_red),
            "periodsSeen": self.periods_seen,
        }
        if include_pending:
            payload["pendingPrediction"] = (
                None
                if self.pending_prediction is None
                else self.pending_prediction.to_dict()
            )
        return payload

    def fingerprint(self) -> str:
        """返回不含临时预测锁的状态指纹。"""

        return sha256_payload(self.state_payload(include_pending=False))

    def predict(self) -> LockedPrediction:
        """锁定一次预测及特征；未更新前禁止重复预测。"""

        if self.pending_prediction is not None:
            raise RuntimeError("E1存在未消费的预测锁，必须先按开奖号更新")
        features = self._feature_matrix()
        scores = [
            self.fixed_effects[index]
            + sum(weight * value for weight, value in zip(self.beta, features[index]))
            for index in range(RED_COUNT)
        ]
        intercept, probabilities = _calibrated_probabilities(scores)
        state_fingerprint = self.fingerprint()
        prediction_payload = {
            "stateFingerprint": state_fingerprint,
            "features": features,
            "intercept": intercept,
            "probabilities": probabilities,
        }
        locked = LockedPrediction(
            probabilities=tuple(probabilities),
            features=features,
            intercept=intercept,
            state_fingerprint=state_fingerprint,
            prediction_fingerprint=sha256_payload(prediction_payload),
        )
        self.pending_prediction = locked
        return locked

    def score_then_update(self, draw: SSQDraw) -> None:
        """消费预测锁，按逐球平均梯度更新并推进特征状态。"""

        locked = self.pending_prediction
        if locked is None:
            raise RuntimeError("E1必须先predict锁定严格前序特征，再执行更新")
        if locked.state_fingerprint != self.fingerprint():
            raise RuntimeError("E1预测锁对应状态已变化，拒绝更新")
        selected = set(draw.red)
        residuals = [
            probability - (1.0 if ball in selected else 0.0)
            for ball, probability in enumerate(locked.probabilities, start=1)
        ]
        fixed_gradients = [
            residual / RED_COUNT + L2_PENALTY * value
            for residual, value in zip(residuals, self.fixed_effects)
        ]
        beta_gradients = [
            sum(
                residuals[index] * locked.features[index][feature]
                for index in range(RED_COUNT)
            )
            / RED_COUNT
            + L2_PENALTY * self.beta[feature]
            for feature in range(FEATURE_COUNT)
        ]
        for index, gradient in enumerate(fixed_gradients):
            self.fixed_accumulators[index] += gradient * gradient
            self.fixed_effects[index] -= (
                LEARNING_RATE
                * gradient
                / (math.sqrt(self.fixed_accumulators[index]) + EPSILON)
            )
        mean_effect = sum(self.fixed_effects) / RED_COUNT
        self.fixed_effects = [value - mean_effect for value in self.fixed_effects]
        for index, gradient in enumerate(beta_gradients):
            self.beta_accumulators[index] += gradient * gradient
            self.beta[index] -= (
                LEARNING_RATE
                * gradient
                / (math.sqrt(self.beta_accumulators[index]) + EPSILON)
            )
        self.pending_prediction = None
        self.ewma30.update(draw.red)
        self.ewma120.update(draw.red)
        self.pair120.update(draw.red)
        for ball in draw.red:
            self.last_seen_period[ball - 1] = self.periods_seen
        self.prior_red = tuple(draw.red)
        self.periods_seen += 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CalibratedRedChallengerEState:
        """从完整状态字典恢复 E1。"""

        state = cls(
            fixed_effects=[
                float(value) for value in cast(list[object], payload["fixedEffects"])
            ],
            beta=[float(value) for value in cast(list[object], payload["beta"])],
            fixed_accumulators=[
                float(value)
                for value in cast(list[object], payload["fixedAccumulators"])
            ],
            beta_accumulators=[
                float(value)
                for value in cast(list[object], payload["betaAccumulators"])
            ],
            last_seen_period=[
                int(value) for value in cast(list[object], payload["lastSeenPeriod"])
            ],
            prior_red=tuple(
                int(value) for value in cast(list[object], payload["priorRed"])
            ),
            periods_seen=int(payload["periodsSeen"]),
        )
        for name, expert in (("ewma30", state.ewma30), ("ewma120", state.ewma120)):
            raw = cast(Mapping[str, object], payload[name])
            expert.counts = [
                float(value) for value in cast(list[object], raw["counts"])
            ]
            expert.effective_draws = float(raw["effectiveDraws"])
        pair_raw = cast(Mapping[str, object], payload["pair120"])
        state.pair120.counts = [
            float(value) for value in cast(list[object], pair_raw["counts"])
        ]
        state.pair120.effective_draws = float(pair_raw["effectiveDraws"])
        pending = payload.get("pendingPrediction")
        state.pending_prediction = (
            None
            if pending is None
            else LockedPrediction.from_dict(cast(Mapping[str, object], pending))
        )
        state._validate()
        return state

    def _validate(self) -> None:
        if (
            len(self.fixed_effects) != RED_COUNT
            or len(self.fixed_accumulators) != RED_COUNT
        ):
            raise ValueError("E1固定效应状态长度非法")
        if (
            len(self.beta) != FEATURE_COUNT
            or len(self.beta_accumulators) != FEATURE_COUNT
        ):
            raise ValueError("E1 beta状态长度非法")
        if len(self.last_seen_period) != RED_COUNT:
            raise ValueError("E1间隔状态长度非法")
        if (
            len(self.ewma30.counts) != RED_COUNT
            or len(self.ewma120.counts) != RED_COUNT
        ):
            raise ValueError("E1边际专家状态长度非法")
        if len(self.pair120.counts) != RED_COUNT * (RED_COUNT - 1) // 2:
            raise ValueError("E1红球对专家状态长度非法")
        if self.periods_seen < 0:
            raise ValueError("E1已处理期数非法")
        if any(
            value <= 0.0 or not math.isfinite(value)
            for value in self.fixed_accumulators
        ):
            raise ValueError("E1固定效应AdaGrad累加器非法")
        if any(
            value <= 0.0 or not math.isfinite(value) for value in self.beta_accumulators
        ):
            raise ValueError("E1 beta AdaGrad累加器非法")
        if any(not math.isfinite(value) for value in (*self.fixed_effects, *self.beta)):
            raise ValueError("E1参数包含非有限值")
        if tuple(sorted(self.prior_red)) != self.prior_red or len(
            set(self.prior_red)
        ) != len(self.prior_red):
            raise ValueError("E1前一期红球状态非法")
        if self.prior_red and len(self.prior_red) != 6:
            raise ValueError("E1前一期红球必须为空或恰好6个")
        if any(ball < 1 or ball > RED_COUNT for ball in self.prior_red):
            raise ValueError("E1前一期红球越界")
        if any(
            period < -1 or period >= self.periods_seen
            for period in self.last_seen_period
        ):
            raise ValueError("E1最近出现期状态非法")
        if abs(sum(self.fixed_effects)) > 1e-10:
            raise ValueError("E1固定效应不满足零均值约束")
        if self.pending_prediction is not None:
            expected_features = self._feature_matrix()
            scores = [
                self.fixed_effects[index]
                + sum(
                    weight * value
                    for weight, value in zip(self.beta, expected_features[index])
                )
                for index in range(RED_COUNT)
            ]
            expected_intercept, expected_probabilities = _calibrated_probabilities(
                scores
            )
            expected_state_fingerprint = self.fingerprint()
            expected_prediction_fingerprint = sha256_payload(
                {
                    "stateFingerprint": expected_state_fingerprint,
                    "features": expected_features,
                    "intercept": expected_intercept,
                    "probabilities": expected_probabilities,
                }
            )
            pending = self.pending_prediction
            if (
                pending.features != expected_features
                or pending.intercept != expected_intercept
                or pending.probabilities != tuple(expected_probabilities)
                or pending.state_fingerprint != expected_state_fingerprint
                or pending.prediction_fingerprint != expected_prediction_fingerprint
            ):
                raise ValueError("E1序列化预测锁与当前状态不一致")


def _pair_index(left: int, right: int) -> int:
    if left >= right:
        raise ValueError("红球对必须严格递增")
    return (left - 1) * (66 - left) // 2 + right - left - 1


def train_state(draws: Sequence[SSQDraw]) -> CalibratedRedChallengerEState:
    """按期号顺序严格先预测后更新训练固定 E1。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("E1输入包含重复期号")
    state = CalibratedRedChallengerEState()
    for draw in ordered:
        state.predict()
        state.score_then_update(draw)
    return state


def walk_forward_prediction_fingerprints(
    draws: Sequence[SSQDraw], stop: int | None = None
) -> list[str]:
    """返回严格走步预测指纹，用于前缀不变性测试。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    limit = len(ordered) if stop is None else min(stop, len(ordered))
    state = CalibratedRedChallengerEState()
    fingerprints: list[str] = []
    for draw in ordered[:limit]:
        fingerprints.append(state.predict().prediction_fingerprint)
        state.score_then_update(draw)
    return fingerprints


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_current_report(
    csv_path: str | Path, ensemble_report_path: str | Path
) -> dict[str, object]:
    """只读历史与 ensemble 报告，构建独立 E 当前报告。"""

    csv = Path(csv_path)
    ensemble_path = Path(ensemble_report_path)
    before_bytes = ensemble_path.read_bytes()
    before_file_sha = hashlib.sha256(before_bytes).hexdigest()
    ensemble = cast(dict[str, object], json.loads(before_bytes.decode("utf-8")))
    embedded_report_sha = str(ensemble.get("reportSha256", ""))
    if not embedded_report_sha:
        raise ValueError("既有ensemble报告缺少reportSha256")
    draws = load_official_history_csv(csv)
    state = train_state(draws)
    locked = state.predict()
    audit = cast(Mapping[str, object], ensemble["auditMetadata"])
    final_probabilities = cast(Mapping[str, object], audit["finalNextProbabilities"])
    blue_probabilities = [
        float(value) for value in cast(list[object], final_probabilities["blue"])
    ]
    b_document = cast(Mapping[str, object], ensemble["diversifiedPortfolioV2"])
    candidate = build_small_compound_8red1blue_v1(
        locked.probabilities, blue_probabilities, b_document
    )
    raw_top8 = sorted(
        range(1, 34),
        key=lambda ball: (-locked.probabilities[ball - 1], ball),
    )[:8]
    e_red = set(cast(list[int], candidate["red"]))
    incumbent = cast(Mapping[str, object], ensemble["smallCompound8Red1BlueV1"])
    d8_red = set(int(value) for value in cast(list[object], incumbent["red"]))
    b_red_union = {
        int(value)
        for group in cast(list[Mapping[str, object]], b_document["groups"])
        for value in cast(list[object], group["red"])
    }
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedDeterministically": True,
        "researchOnly": True,
        "retrospective": False,
        "formalGate": False,
        "selection": None,
        "formalRecommendationStatus": "uniform_abstain",
        "model": MODEL_NAME,
        "protocol": PROTOCOL,
        "protocolSha256": protocol_sha256(),
        "futureProtocolSha256": future_protocol_sha256(),
        "builderProtocolSha256": builder_protocol_sha256(),
        "input": {
            "csvPath": str(csv),
            "csvSha256": _file_sha256(csv),
            "periods": len(draws),
            "latestIssue": draws[-1].issue,
            "latestDate": draws[-1].draw_date,
            "ensembleReportPath": str(ensemble_path),
            "ensembleFileSha256": before_file_sha,
            "ensembleReportSha256": embedded_report_sha,
        },
        "fingerprints": {
            "stateSha256": locked.state_fingerprint,
            "predictionSha256": locked.prediction_fingerprint,
            "redProbabilitiesSha256": sha256_payload(list(locked.probabilities)),
            "blueProbabilitiesSha256": sha256_payload(blue_probabilities),
            "bDocumentSha256": sha256_payload(b_document),
            "inputSha256": sha256_payload(
                {
                    "csv": _file_sha256(csv),
                    "ensembleFile": before_file_sha,
                    "ensembleReport": embedded_report_sha,
                }
            ),
        },
        "currentTargetGroup": {
            "rawTop8": raw_top8,
            "red": candidate["red"],
            "blue": candidate["blue"],
            "selectedCandidateRank": candidate["selectedCandidateRank"],
            "expandedTickets": candidate["expandedTickets"],
            "audit": candidate["audit"],
            "overlaps": {
                "EWithD8Red": len(e_red & d8_red),
                "EWithBRedUnion": len(e_red & b_red_union),
                "EWithBTickets": cast(Mapping[str, object], candidate["audit"])[
                    "overlapWithB"
                ],
            },
        },
        "probabilities": {
            "red": list(locked.probabilities),
            "blueFromEnsemble": blue_probabilities,
        },
        "state": state.state_payload(include_pending=False),
        "ensembleIntegrity": {
            "bytesUnchanged": ensemble_path.read_bytes() == before_bytes,
            "fileSha256Unchanged": _file_sha256(ensemble_path) == before_file_sha,
            "reportSha256Unchanged": str(
                json.loads(ensemble_path.read_text(encoding="utf-8"))["reportSha256"]
            )
            == embedded_report_sha,
        },
    }
    if not all(cast(Mapping[str, bool], report["ensembleIntegrity"]).values()):
        raise RuntimeError("E1构建期间既有ensemble报告发生变化")
    report["reportSha256"] = sha256_payload(report)
    return report


def write_report(report: Mapping[str, object], output_path: str | Path) -> Path:
    """原子写入 E1 JSON 报告。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                report,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "CalibratedRedChallengerEState",
    "FUTURE_PROTOCOL",
    "LockedPrediction",
    "PROTOCOL",
    "build_current_report",
    "future_protocol_sha256",
    "protocol_sha256",
    "sha256_payload",
    "train_state",
    "walk_forward_prediction_fingerprints",
    "write_report",
]
