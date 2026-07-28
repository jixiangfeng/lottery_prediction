# -*- coding: utf-8 -*-
# mypy: disable-error-code="arg-type,call-overload"
"""双色球 Challenger E2：冻结八候选所共用的严格前序在线核心。"""

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

MODEL_NAME = "ssq_challenger_e2_v1"
SCHEMA_VERSION = "ssq_challenger_e2_v1"
RED_COUNT = 33
UNIFORM_RED_PROBABILITY = 6.0 / RED_COUNT
ADAGRAD_INITIAL_ACCUMULATOR = 1e-6
LEARNING_RATE = 0.05
EPSILON = 1e-8
INTERCEPT_LOWER = -30.0
INTERCEPT_UPPER = 30.0
INTERCEPT_ITERATIONS = 80
INTERCEPT_TOLERANCE = 1e-12
FEATURE_NAMES = (
    "EWMA30",
    "EWMA120",
    "gap",
    "prior-repeat",
    "pair-affinity",
    "trend",
)
FEATURE_MASKS: dict[str, tuple[str, ...]] = {
    "F0": FEATURE_NAMES[:2],
    "F1": FEATURE_NAMES[:4],
    "F2": FEATURE_NAMES[:5],
    "F3": FEATURE_NAMES,
}


@dataclass(frozen=True)
class CandidateSpec:
    """一个冻结候选；候选间只允许特征掩码、beta 维度和 L2 不同。"""

    candidate_id: str
    feature_set: str
    feature_names: tuple[str, ...]
    l2: float


CANDIDATE_SPECS = tuple(
    CandidateSpec(
        candidate_id=f"{feature_set}_L{l2_id}",
        feature_set=feature_set,
        feature_names=FEATURE_MASKS[feature_set],
        l2=l2,
    )
    for feature_set in ("F0", "F1", "F2", "F3")
    for l2_id, l2 in (("001", 0.01), ("010", 0.10))
)
CANDIDATE_BY_ID = {spec.candidate_id: spec for spec in CANDIDATE_SPECS}

PROTOCOL: dict[str, object] = {
    "model": MODEL_NAME,
    "purpose": "finite_selection_retrospective_research_only",
    "candidateCount": 8,
    "candidates": [
        {
            "candidateId": spec.candidate_id,
            "featureSet": spec.feature_set,
            "featureMask": list(spec.feature_names),
            "betaDimension": len(spec.feature_names),
            "l2": spec.l2,
        }
        for spec in CANDIDATE_SPECS
    ],
    "featureEquations": {
        "EWMA30": "clip((p_EWMA30-6/33)/0.05,-3,3)",
        "EWMA120": "clip((p_EWMA120-6/33)/0.05,-3,3)",
        "gap": "log1p(min(gap,60))/log(61)-0.5",
        "prior-repeat": "I(ball in prior red6)-6/33",
        "pair-affinity": "mean prior RedPairExpert120 modifier versus prior red6",
        "trend": "x_EWMA30-x_EWMA120",
    },
    "parameters": {
        "fixedEffects": 33,
        "fixedEffectsConstraint": "zero_mean_after_each_update",
        "optimizer": "AdaGrad",
        "eta": LEARNING_RATE,
        "initialAccumulator": ADAGRAD_INITIAL_ACCUMULATOR,
        "epsilon": EPSILON,
    },
    "calibration": {
        "constraint": "sum(sigmoid(intercept+score_i))=6",
        "method": "fixed_bisection",
        "bounds": [INTERCEPT_LOWER, INTERCEPT_UPPER],
        "iterations": INTERCEPT_ITERATIONS,
        "tolerance": INTERCEPT_TOLERANCE,
    },
    "onlineOrder": "predict-lock-score-update",
    "tieBreak": "probability_descending_then_ball_ascending",
    "partitions": {
        "warmup": 120,
        "search": 923,
        "validation": 500,
        "diagnostic": 500,
        "total": 2043,
    },
    "selection": {
        "proper": "candidate LogLoss < uniform AND candidate Brier <= uniform",
        "paired": "mean candidate-minus-D8 red8 actual overlap > 0",
        "blocks": "five consecutive 100; at least four nonnegative; none below -0.05",
        "secondaryOnly": "red5 rate",
        "recordOnly": "red6",
        "order": "LogLoss asc, Brier asc, mean red8 delta desc, candidateId asc",
        "noneEligible": "rejected",
        "posthocRelaxation": False,
    },
    "diagnostic": "selected candidate only; latest500; not promotion evidence",
    "current": "selected candidate replayed from beginning through all2043",
    "fixed": {"cliOverrides": False, "retries": 0, "automaticPromotion": False},
}


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def protocol_sha256() -> str:
    """返回导入时即固定的 E2 协议摘要。"""

    return sha256_payload(PROTOCOL)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _calibrated_probabilities(scores: Sequence[float]) -> tuple[float, list[float]]:
    lower, upper = INTERCEPT_LOWER, INTERCEPT_UPPER
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


def _pair_index(left: int, right: int) -> int:
    if left >= right:
        raise ValueError("红球对必须严格递增")
    return (left - 1) * (66 - left) // 2 + right - left - 1


@dataclass(frozen=True)
class LockedPrediction:
    probabilities: tuple[float, ...]
    features: tuple[tuple[float, ...], ...]
    intercept: float
    state_fingerprint: str
    prediction_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "probabilities": list(self.probabilities),
            "features": [list(row) for row in self.features],
            "intercept": self.intercept,
            "stateFingerprint": self.state_fingerprint,
            "predictionFingerprint": self.prediction_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LockedPrediction:
        return cls(
            probabilities=tuple(
                float(value) for value in cast(list[object], payload["probabilities"])
            ),
            features=tuple(
                tuple(float(value) for value in cast(list[object], row))
                for row in cast(list[object], payload["features"])
            ),
            intercept=float(payload["intercept"]),
            state_fingerprint=str(payload["stateFingerprint"]),
            prediction_fingerprint=str(payload["predictionFingerprint"]),
        )


@dataclass
class ChallengerE2State:
    """E2 单候选在线状态，所有特征均只读取严格前序状态。"""

    spec: CandidateSpec
    fixed_effects: list[float] = field(default_factory=lambda: [0.0] * RED_COUNT)
    beta: list[float] = field(default_factory=list)
    fixed_accumulators: list[float] = field(
        default_factory=lambda: [ADAGRAD_INITIAL_ACCUMULATOR] * RED_COUNT
    )
    beta_accumulators: list[float] = field(default_factory=list)
    ewma30: RedMarginalExpert = field(default_factory=lambda: RedMarginalExpert(30))
    ewma120: RedMarginalExpert = field(default_factory=lambda: RedMarginalExpert(120))
    pair120: RedPairExpert = field(default_factory=RedPairExpert)
    last_seen_period: list[int] = field(default_factory=lambda: [-1] * RED_COUNT)
    prior_red: tuple[int, ...] = ()
    periods_seen: int = 0
    pending_prediction: LockedPrediction | None = None

    def __post_init__(self) -> None:
        dimension = len(self.spec.feature_names)
        if not self.beta:
            self.beta = [0.0] * dimension
        if not self.beta_accumulators:
            self.beta_accumulators = [ADAGRAD_INITIAL_ACCUMULATOR] * dimension

    def _full_feature_matrix(self) -> tuple[tuple[float, ...], ...]:
        ewma30 = self.ewma30.probabilities()
        ewma120 = self.ewma120.probabilities()
        pair_modifiers = self.pair120.modifiers()
        prior = set(self.prior_red)
        rows: list[tuple[float, ...]] = []
        for ball in range(1, RED_COUNT + 1):
            x30 = max(
                -3.0,
                min(3.0, (ewma30[ball - 1] - UNIFORM_RED_PROBABILITY) / 0.05),
            )
            x120 = max(
                -3.0,
                min(3.0, (ewma120[ball - 1] - UNIFORM_RED_PROBABILITY) / 0.05),
            )
            last_seen = self.last_seen_period[ball - 1]
            gap = self.periods_seen if last_seen < 0 else self.periods_seen - last_seen
            pair_values = [
                pair_modifiers[_pair_index(min(ball, other), max(ball, other))]
                for other in self.prior_red
                if other != ball
            ]
            rows.append(
                (
                    x30,
                    x120,
                    math.log1p(min(gap, 60)) / math.log(61.0) - 0.5,
                    (1.0 if ball in prior else 0.0) - UNIFORM_RED_PROBABILITY,
                    sum(pair_values) / len(pair_values) if pair_values else 0.0,
                    x30 - x120,
                )
            )
        return tuple(rows)

    def _feature_matrix(self) -> tuple[tuple[float, ...], ...]:
        dimension = len(self.spec.feature_names)
        return tuple(row[:dimension] for row in self._full_feature_matrix())

    def state_payload(self, include_pending: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": MODEL_NAME,
            "candidateId": self.spec.candidate_id,
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
        return sha256_payload(self.state_payload(include_pending=False))

    def predict(self) -> LockedPrediction:
        if self.pending_prediction is not None:
            raise RuntimeError("E2存在未消费预测锁，必须先更新")
        features = self._feature_matrix()
        scores = [
            self.fixed_effects[index]
            + sum(weight * value for weight, value in zip(self.beta, features[index]))
            for index in range(RED_COUNT)
        ]
        intercept, probabilities = _calibrated_probabilities(scores)
        state_fingerprint = self.fingerprint()
        prediction_fingerprint = sha256_payload(
            {
                "candidateId": self.spec.candidate_id,
                "stateFingerprint": state_fingerprint,
                "features": features,
                "intercept": intercept,
                "probabilities": probabilities,
            }
        )
        locked = LockedPrediction(
            tuple(probabilities),
            features,
            intercept,
            state_fingerprint,
            prediction_fingerprint,
        )
        self.pending_prediction = locked
        return locked

    def score_then_update(self, draw: SSQDraw) -> None:
        locked = self.pending_prediction
        if locked is None:
            raise RuntimeError("E2必须先predict锁定严格前序特征")
        if locked.state_fingerprint != self.fingerprint():
            raise RuntimeError("E2预测锁状态已变化")
        selected = set(draw.red)
        residuals = [
            probability - (1.0 if ball in selected else 0.0)
            for ball, probability in enumerate(locked.probabilities, start=1)
        ]
        fixed_gradients = [
            residual / RED_COUNT + self.spec.l2 * value
            for residual, value in zip(residuals, self.fixed_effects)
        ]
        beta_gradients = [
            sum(
                residuals[index] * locked.features[index][feature]
                for index in range(RED_COUNT)
            )
            / RED_COUNT
            + self.spec.l2 * self.beta[feature]
            for feature in range(len(self.beta))
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
    def from_dict(cls, payload: Mapping[str, object]) -> ChallengerE2State:
        candidate_id = str(payload["candidateId"])
        if candidate_id not in CANDIDATE_BY_ID:
            raise ValueError("E2候选ID非法")
        state = cls(
            spec=CANDIDATE_BY_ID[candidate_id],
            fixed_effects=[
                float(v) for v in cast(list[object], payload["fixedEffects"])
            ],
            beta=[float(v) for v in cast(list[object], payload["beta"])],
            fixed_accumulators=[
                float(v) for v in cast(list[object], payload["fixedAccumulators"])
            ],
            beta_accumulators=[
                float(v) for v in cast(list[object], payload["betaAccumulators"])
            ],
            last_seen_period=[
                int(v) for v in cast(list[object], payload["lastSeenPeriod"])
            ],
            prior_red=tuple(int(v) for v in cast(list[object], payload["priorRed"])),
            periods_seen=int(payload["periodsSeen"]),
        )
        for name, expert in (("ewma30", state.ewma30), ("ewma120", state.ewma120)):
            raw = cast(Mapping[str, object], payload[name])
            expert.counts = [float(v) for v in cast(list[object], raw["counts"])]
            expert.effective_draws = float(raw["effectiveDraws"])
        pair_raw = cast(Mapping[str, object], payload["pair120"])
        state.pair120.counts = [
            float(v) for v in cast(list[object], pair_raw["counts"])
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
        dimension = len(self.spec.feature_names)
        if (
            len(self.fixed_effects) != RED_COUNT
            or len(self.fixed_accumulators) != RED_COUNT
        ):
            raise ValueError("E2固定效应状态长度非法")
        if len(self.beta) != dimension or len(self.beta_accumulators) != dimension:
            raise ValueError("E2 beta状态长度非法")
        if abs(sum(self.fixed_effects)) > 1e-10:
            raise ValueError("E2固定效应不满足零均值")
        if any(
            value <= 0 or not math.isfinite(value) for value in self.fixed_accumulators
        ):
            raise ValueError("E2固定累加器非法")
        if any(
            value <= 0 or not math.isfinite(value) for value in self.beta_accumulators
        ):
            raise ValueError("E2 beta累加器非法")
        if self.pending_prediction is not None:
            pending = self.pending_prediction
            self.pending_prediction = None
            expected = self.predict()
            self.pending_prediction = pending
            if expected != pending:
                raise ValueError("E2序列化预测锁与当前状态不一致")


def train_state(draws: Sequence[SSQDraw], spec: CandidateSpec) -> ChallengerE2State:
    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("E2输入包含重复期号")
    state = ChallengerE2State(spec)
    for draw in ordered:
        state.predict()
        state.score_then_update(draw)
    return state


def walk_forward_prediction_fingerprints(
    draws: Sequence[SSQDraw], spec: CandidateSpec, stop: int | None = None
) -> list[str]:
    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    limit = len(ordered) if stop is None else min(stop, len(ordered))
    state = ChallengerE2State(spec)
    fingerprints: list[str] = []
    for draw in ordered[:limit]:
        fingerprints.append(state.predict().prediction_fingerprint)
        state.score_then_update(draw)
    return fingerprints


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_current_report(
    csv_path: str | Path,
    ensemble_report_path: str | Path,
    selection_report: Mapping[str, object],
    draws: Sequence[SSQDraw] | None = None,
) -> dict[str, object]:
    """按已固定 Validation 选择构建独立当前报告；拒绝时目标组为 null。"""

    csv = Path(csv_path)
    ensemble_path = Path(ensemble_report_path)
    before = ensemble_path.read_bytes()
    ensemble_file_sha = hashlib.sha256(before).hexdigest()
    ensemble = cast(dict[str, object], json.loads(before))
    embedded_sha = str(ensemble.get("reportSha256", ""))
    status = str(selection_report["selectionStatus"])
    selected_id_raw = selection_report.get("selectedCandidateId")
    selected_id = None if selected_id_raw is None else str(selected_id_raw)
    if selection_report.get("protocolSha256") != protocol_sha256():
        raise ValueError("E2选择报告协议摘要不匹配")
    if status not in {"selected", "rejected"}:
        raise ValueError("E2选择状态非法")
    if status == "selected" and selected_id not in CANDIDATE_BY_ID:
        raise ValueError("E2已选候选ID非法")
    current_target: dict[str, object] | None = None
    probabilities: dict[str, object] | None = None
    state_payload: dict[str, object] | None = None
    state_sha: str | None = None
    prediction_sha: str | None = None
    input_draws = list(draws) if draws is not None else None
    if status == "selected":
        if input_draws is None:
            input_draws = load_official_history_csv(csv)
        if len(input_draws) != 2043:
            raise ValueError("E2当前重放要求恰好2043期")
        state = train_state(input_draws, CANDIDATE_BY_ID[cast(str, selected_id)])
        locked = state.predict()
        audit = cast(Mapping[str, object], ensemble["auditMetadata"])
        final = cast(Mapping[str, object], audit["finalNextProbabilities"])
        blue = [float(v) for v in cast(list[object], final["blue"])]
        b_document = cast(Mapping[str, object], ensemble["diversifiedPortfolioV2"])
        candidate = build_small_compound_8red1blue_v1(
            locked.probabilities, blue, b_document
        )
        current_target = {
            "candidateId": selected_id,
            "red": candidate["red"],
            "blue": candidate["blue"],
            "expandedTickets": candidate["expandedTickets"],
            "selectedCandidateRank": candidate["selectedCandidateRank"],
            "audit": candidate["audit"],
        }
        probabilities = {"red": list(locked.probabilities), "blueFromIncumbent": blue}
        state_payload = state.state_payload(include_pending=False)
        state_sha = locked.state_fingerprint
        prediction_sha = locked.prediction_fingerprint
    if ensemble_path.read_bytes() != before:
        raise RuntimeError("E2构建期间ensemble字节发生变化")
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedDeterministically": True,
        "researchOnly": True,
        "retrospective": False,
        "formalGate": False,
        "formalRecommendationStatus": "uniform_abstain",
        "automaticPromotion": False,
        "selectionStatus": status,
        "selectedCandidateId": selected_id,
        "protocol": PROTOCOL,
        "protocolSha256": protocol_sha256(),
        "selectionReportSha256": selection_report.get("reportSha256"),
        "input": {
            "csvPath": str(csv),
            "csvSha256": _file_sha256(csv) if csv.exists() else None,
            "ensembleReportPath": str(ensemble_path),
            "ensembleFileSha256": ensemble_file_sha,
            "ensembleReportSha256": embedded_sha,
        },
        "currentTargetGroup": current_target,
        "probabilities": probabilities,
        "state": state_payload,
        "fingerprints": {
            "stateSha256": state_sha,
            "predictionSha256": prediction_sha,
            "inputSha256": sha256_payload(
                {
                    "csv": _file_sha256(csv) if csv.exists() else None,
                    "ensembleFile": ensemble_file_sha,
                    "selection": selection_report.get("reportSha256"),
                }
            ),
        },
        "ensembleIntegrity": {
            "bytesUnchanged": ensemble_path.read_bytes() == before,
            "fileSha256Unchanged": _file_sha256(ensemble_path) == ensemble_file_sha,
            "reportSha256Unchanged": str(
                json.loads(ensemble_path.read_bytes())["reportSha256"]
            )
            == embedded_sha,
        },
        "prospectiveArtifactsCreated": False,
    }
    report["reportSha256"] = sha256_payload(report)
    return report


def write_report(report: Mapping[str, object], output_path: str | Path) -> Path:
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
    "CANDIDATE_SPECS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "CandidateSpec",
    "ChallengerE2State",
    "LockedPrediction",
    "build_current_report",
    "canonical_bytes",
    "protocol_sha256",
    "sha256_payload",
    "train_state",
    "walk_forward_prediction_fingerprints",
    "write_report",
]
