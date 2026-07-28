# -*- coding: utf-8 -*-
# mypy: disable-error-code="arg-type,call-overload"
"""双色球 Challenger E2 固定历史选择与选后 Diagnostic 评估。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, cast

from src.analysis.ssq_8red1blue_v1_history import (
    CONTROL_COUNT,
    build_matched_random_control,
)
from src.analysis.ssq_challenger_e2_v1 import (
    CANDIDATE_BY_ID,
    CANDIDATE_SPECS,
    PROTOCOL,
    ChallengerE2State,
    protocol_sha256,
    sha256_payload,
)
from src.analysis.ssq_diversified_portfolio_v2 import build_diversified_portfolio_v2
from src.analysis.ssq_ensemble_v1 import FixedEnsembleState
from src.analysis.ssq_history import SSQDraw, load_official_history_csv
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)

SCHEMA_VERSION = "ssq_challenger_e2_selection_v1"
WARMUP_DRAWS = 120
SEARCH_DRAWS = 923
VALIDATION_DRAWS = 500
DIAGNOSTIC_DRAWS = 500
TOTAL_DRAWS = WARMUP_DRAWS + SEARCH_DRAWS + VALIDATION_DRAWS + DIAGNOSTIC_DRAWS
FROZEN_LAST_ISSUE = "2026085"
VALIDATION_START = WARMUP_DRAWS + SEARCH_DRAWS
DIAGNOSTIC_START = VALIDATION_START + VALIDATION_DRAWS
BLOCK_SIZE = 100
RED_COUNT = 33
UNIFORM_RED_PROBABILITY = 6.0 / RED_COUNT


def _frozen_research_prefix(draws: Sequence[SSQDraw]) -> list[SSQDraw]:
    """从增长中的规范历史提取E2预注册的固定2043期前缀。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("E2输入包含重复期号")
    frozen = [draw for draw in ordered if int(draw.issue) <= int(FROZEN_LAST_ISSUE)]
    if len(frozen) != TOTAL_DRAWS or frozen[-1].issue != FROZEN_LAST_ISSUE:
        raise ValueError("E2冻结研究前缀必须恰好截至2026085并包含2043期")
    return frozen


def _frozen_data_sha256(draws: Sequence[SSQDraw]) -> str:
    return sha256_payload(
        [
            {
                "issue": draw.issue,
                "date": draw.draw_date,
                "red": list(draw.red),
                "blue": draw.blue,
                "sourceUrl": draw.source_url,
                "rawHash": draw.raw_hash,
            }
            for draw in draws
        ]
    )


def _ordered(draws: Sequence[SSQDraw]) -> list[SSQDraw]:
    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) != TOTAL_DRAWS:
        raise ValueError("E2固定研究要求官方历史恰好2043期")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("E2输入包含重复期号")
    return ordered


def fixed_split_boundaries(draws: Sequence[SSQDraw]) -> dict[str, dict[str, object]]:
    """返回四段固定索引和实际期号边界。"""

    ordered = _ordered(draws)
    definitions = (
        ("warmup", 0, WARMUP_DRAWS),
        ("search", WARMUP_DRAWS, VALIDATION_START),
        ("validation", VALIDATION_START, DIAGNOSTIC_START),
        ("diagnostic", DIAGNOSTIC_START, TOTAL_DRAWS),
    )
    return {
        name: {
            "startIndex": start,
            "endIndexExclusive": end,
            "periods": end - start,
            "firstIssue": ordered[start].issue,
            "lastIssue": ordered[end - 1].issue,
            "firstDate": ordered[start].draw_date,
            "lastDate": ordered[end - 1].draw_date,
        }
        for name, start, end in definitions
    }


def _red_scores(
    probabilities: Sequence[float], red: Sequence[int]
) -> tuple[float, float]:
    selected = set(red)
    log_loss = 0.0
    brier = 0.0
    for ball, probability in enumerate(probabilities, start=1):
        if not 0.0 < probability < 1.0 or not math.isfinite(probability):
            raise ValueError("E2红球概率必须为有限开区间值")
        outcome = 1.0 if ball in selected else 0.0
        log_loss -= outcome * math.log(probability) + (1.0 - outcome) * math.log(
            1.0 - probability
        )
        brier += (probability - outcome) ** 2
    return log_loss / RED_COUNT, brier / RED_COUNT


def _red8(document: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in cast(list[object], document["red"]))


def _overlap(red8: Sequence[int], draw: SSQDraw) -> int:
    return len(set(red8).intersection(draw.red))


@dataclass
class _CandidateAccumulator:
    candidate_id: str
    log_loss_total: float = 0.0
    brier_total: float = 0.0
    overlap_total: int = 0
    d8_overlap_total: int = 0
    red5_hits: int = 0
    d8_red5_hits: int = 0
    red6_hits: int = 0
    d8_red6_hits: int = 0
    block_deltas: list[float] = field(default_factory=lambda: [0.0] * 5)
    observations: int = 0

    def add(
        self, log_loss: float, brier: float, candidate_overlap: int, d8_overlap: int
    ) -> None:
        block = self.observations // BLOCK_SIZE
        delta = candidate_overlap - d8_overlap
        self.log_loss_total += log_loss
        self.brier_total += brier
        self.overlap_total += candidate_overlap
        self.d8_overlap_total += d8_overlap
        self.red5_hits += int(candidate_overlap >= 5)
        self.d8_red5_hits += int(d8_overlap >= 5)
        self.red6_hits += int(candidate_overlap >= 6)
        self.d8_red6_hits += int(d8_overlap >= 6)
        self.block_deltas[block] += delta
        self.observations += 1

    def report(self, uniform_scores: Mapping[str, float]) -> dict[str, object]:
        if self.observations != VALIDATION_DRAWS:
            raise ValueError("E2 Validation候选必须恰好500个观测")
        report: dict[str, object] = {
            "candidateId": self.candidate_id,
            "properScores": {
                "candidate": {
                    "observations": self.observations,
                    "redLogLossPerBall": self.log_loss_total / self.observations,
                    "redBrierPerBall": self.brier_total / self.observations,
                },
                "uniform": dict(uniform_scores),
            },
            "paired": {
                "meanRed8OverlapCandidate": self.overlap_total / self.observations,
                "meanRed8OverlapD8": self.d8_overlap_total / self.observations,
                "meanRed8OverlapCandidateMinusD8": (
                    self.overlap_total - self.d8_overlap_total
                )
                / self.observations,
                "red5RateCandidate": self.red5_hits / self.observations,
                "red5RateD8": self.d8_red5_hits / self.observations,
                "red5RateCandidateMinusD8": (self.red5_hits - self.d8_red5_hits)
                / self.observations,
                "red6CountCandidate": self.red6_hits,
                "red6CountD8": self.d8_red6_hits,
                "red6CountCandidateMinusD8": self.red6_hits - self.d8_red6_hits,
                "blocks": [
                    {
                        "blockIndex": index,
                        "startValidationOffset": index * BLOCK_SIZE,
                        "endValidationOffsetExclusive": (index + 1) * BLOCK_SIZE,
                        "periods": BLOCK_SIZE,
                        "meanRed8OverlapCandidateMinusD8": total / BLOCK_SIZE,
                    }
                    for index, total in enumerate(self.block_deltas)
                ],
            },
        }
        report["eligibility"] = assess_eligibility(report)
        return report


def assess_eligibility(candidate_metrics: Mapping[str, object]) -> dict[str, object]:
    """执行不可放宽的 Validation 硬准入。"""

    proper = cast(Mapping[str, object], candidate_metrics["properScores"])
    candidate = cast(Mapping[str, object], proper["candidate"])
    uniform = cast(Mapping[str, object], proper["uniform"])
    paired = cast(Mapping[str, object], candidate_metrics["paired"])
    blocks = cast(list[Mapping[str, object]], paired["blocks"])
    block_values = [float(block["meanRed8OverlapCandidateMinusD8"]) for block in blocks]
    if len(block_values) != 5:
        raise ValueError("E2准入要求恰好5个固定100期块")
    checks = {
        "logLossStrictlyBetterThanUniform": float(candidate["redLogLossPerBall"])
        < float(uniform["redLogLossPerBall"]),
        "brierNoWorseThanUniform": float(candidate["redBrierPerBall"])
        <= float(uniform["redBrierPerBall"]),
        "meanRed8OverlapCandidateMinusD8Positive": float(
            paired["meanRed8OverlapCandidateMinusD8"]
        )
        > 0.0,
        "atLeastFourNonnegativeBlocks": sum(value >= 0.0 for value in block_values)
        >= 4,
        "noBlockBelowMinus0_05": min(block_values) >= -0.05,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "secondaryRed5UsedForSelection": False,
        "recordOnlyRed6UsedForSelection": False,
    }


def select_candidate(
    candidate_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """按冻结字典序选择；没有合格候选时拒绝。"""

    if [str(item["candidateId"]) for item in candidate_metrics] != [
        spec.candidate_id for spec in CANDIDATE_SPECS
    ]:
        raise ValueError("E2选择输入必须按冻结顺序包含恰好8个候选")
    eligible: list[Mapping[str, object]] = []
    for item in candidate_metrics:
        eligibility = assess_eligibility(item)
        if bool(eligibility["eligible"]):
            eligible.append(item)
    if not eligible:
        return {"selectionStatus": "rejected", "selectedCandidateId": None}

    def key(item: Mapping[str, object]) -> tuple[float, float, float, str]:
        proper = cast(Mapping[str, object], item["properScores"])
        candidate = cast(Mapping[str, object], proper["candidate"])
        paired = cast(Mapping[str, object], item["paired"])
        return (
            float(candidate["redLogLossPerBall"]),
            float(candidate["redBrierPerBall"]),
            -float(paired["meanRed8OverlapCandidateMinusD8"]),
            str(item["candidateId"]),
        )

    selected = min(eligible, key=key)
    return {
        "selectionStatus": "selected",
        "selectedCandidateId": str(selected["candidateId"]),
    }


def _uniform_scores() -> dict[str, float | int]:
    log_loss, brier = _red_scores(
        [UNIFORM_RED_PROBABILITY] * RED_COUNT, (1, 2, 3, 4, 5, 6)
    )
    return {
        "observations": VALIDATION_DRAWS,
        "redLogLossPerBall": log_loss,
        "redBrierPerBall": brier,
    }


def _data_payload(draws: Sequence[SSQDraw]) -> list[dict[str, object]]:
    return [
        {
            "issue": draw.issue,
            "date": draw.draw_date,
            "red": list(draw.red),
            "blue": draw.blue,
            "sourceUrl": draw.source_url,
            "rawHash": draw.raw_hash,
        }
        for draw in draws
    ]


def _nested_float(
    item: Mapping[str, object], section: str, label: str, metric: str | None = None
) -> float:
    section_payload = cast(Mapping[str, object], item[section])
    value = section_payload[label]
    if metric is not None:
        value = cast(Mapping[str, object], value)[metric]
    return float(cast(float | int, value))


def _diagnostic_report(
    candidate_id: str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(records)
    if count != DIAGNOSTIC_DRAWS:
        raise ValueError("E2 Diagnostic必须恰好500期")
    proper: dict[str, dict[str, float | int]] = {}
    for label in ("candidate", "D8", "uniform"):
        log_losses = [
            _nested_float(item, "properScores", label, "redLogLossPerBall")
            for item in records
        ]
        briers = [
            _nested_float(item, "properScores", label, "redBrierPerBall")
            for item in records
        ]
        proper[label] = {
            "observations": count,
            "redLogLossPerBall": sum(log_losses) / count,
            "redBrierPerBall": sum(briers) / count,
        }
    overlap_labels = ("candidate", "D8", "C32IssueMean")
    overlaps = {
        label: sum(
            [_nested_float(item, "red8ActualOverlap", label) for item in records]
        )
        / count
        for label in overlap_labels
    }
    return {
        "candidateId": candidate_id,
        "periods": count,
        "explicitlyDiagnostic": True,
        "promotionEvidence": False,
        "automaticPromotion": False,
        "properScores": proper,
        "red8ActualOverlapMean": overlaps,
        "differences": {
            "candidateMinusD8": {
                "redLogLossPerBall": proper["candidate"]["redLogLossPerBall"]
                - proper["D8"]["redLogLossPerBall"],
                "redBrierPerBall": proper["candidate"]["redBrierPerBall"]
                - proper["D8"]["redBrierPerBall"],
                "red8ActualOverlap": overlaps["candidate"] - overlaps["D8"],
            },
            "candidateMinusUniform": {
                "redLogLossPerBall": proper["candidate"]["redLogLossPerBall"]
                - proper["uniform"]["redLogLossPerBall"],
                "redBrierPerBall": proper["candidate"]["redBrierPerBall"]
                - proper["uniform"]["redBrierPerBall"],
            },
            "candidateMinusC32IssueMean": {
                "red8ActualOverlap": overlaps["candidate"] - overlaps["C32IssueMean"]
            },
        },
        "auditSha256": sha256_payload(records),
    }


def evaluate_selection(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """一次运行冻结八候选 Validation 选择，再做选后 Diagnostic。"""

    ordered = _ordered(draws)
    states = {spec.candidate_id: ChallengerE2State(spec) for spec in CANDIDATE_SPECS}
    incumbent = FixedEnsembleState()
    accumulators = {
        spec.candidate_id: _CandidateAccumulator(spec.candidate_id)
        for spec in CANDIDATE_SPECS
    }
    validation_audit: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    selected_id: str | None = None
    selection_result: dict[str, object] | None = None
    uniform_vector = [UNIFORM_RED_PROBABILITY] * RED_COUNT
    uniform_validation = _uniform_scores()

    for index, draw in enumerate(ordered):
        locked = {
            candidate_id: state.predict() for candidate_id, state in states.items()
        }
        incumbent_red, incumbent_blue, _ = incumbent.predict()
        if any(abs(sum(item.probabilities) - 6.0) > 1e-10 for item in locked.values()):
            raise RuntimeError("E2逐期候选概率和不等于6")

        if VALIDATION_START <= index < DIAGNOSTIC_START:
            b_document = build_diversified_portfolio_v2(incumbent_red, incumbent_blue)
            d8_document = build_small_compound_8red1blue_v1(
                incumbent_red, incumbent_blue, b_document
            )
            d8_overlap = _overlap(_red8(d8_document), draw)
            candidate_audit: dict[str, object] = {}
            for spec in CANDIDATE_SPECS:
                prediction = locked[spec.candidate_id]
                document = build_small_compound_8red1blue_v1(
                    prediction.probabilities, incumbent_blue, b_document
                )
                overlap = _overlap(_red8(document), draw)
                log_loss, brier = _red_scores(prediction.probabilities, draw.red)
                accumulators[spec.candidate_id].add(
                    log_loss, brier, overlap, d8_overlap
                )
                candidate_audit[spec.candidate_id] = {
                    "predictionSha256": prediction.prediction_fingerprint,
                    "documentSha256": sha256_payload(document),
                    "red8ActualOverlap": overlap,
                    "redLogLossPerBall": log_loss,
                    "redBrierPerBall": brier,
                }
            validation_audit.append(
                {
                    "issue": draw.issue,
                    "rawHash": draw.raw_hash,
                    "D8DocumentSha256": sha256_payload(d8_document),
                    "D8Red8ActualOverlap": d8_overlap,
                    "candidates": candidate_audit,
                }
            )

        if index == DIAGNOSTIC_START:
            validation_metrics = [
                accumulators[spec.candidate_id].report(uniform_validation)
                for spec in CANDIDATE_SPECS
            ]
            selection_result = select_candidate(validation_metrics)
            selected_raw = selection_result["selectedCandidateId"]
            selected_id = None if selected_raw is None else str(selected_raw)

        if index >= DIAGNOSTIC_START and selected_id is not None:
            b_document = build_diversified_portfolio_v2(incumbent_red, incumbent_blue)
            d8_document = build_small_compound_8red1blue_v1(
                incumbent_red, incumbent_blue, b_document
            )
            candidate_prediction = locked[selected_id]
            candidate_document = build_small_compound_8red1blue_v1(
                candidate_prediction.probabilities, incumbent_blue, b_document
            )
            candidate_scores = _red_scores(candidate_prediction.probabilities, draw.red)
            d8_scores = _red_scores(incumbent_red, draw.red)
            uniform_scores = _red_scores(uniform_vector, draw.red)
            candidate_overlap = _overlap(_red8(candidate_document), draw)
            d8_overlap = _overlap(_red8(d8_document), draw)
            random_overlaps = [
                _overlap(
                    build_matched_random_control(draw.issue, control_index).red8, draw
                )
                for control_index in range(CONTROL_COUNT)
            ]
            diagnostic_records.append(
                {
                    "issue": draw.issue,
                    "rawHash": draw.raw_hash,
                    "properScores": {
                        "candidate": {
                            "redLogLossPerBall": candidate_scores[0],
                            "redBrierPerBall": candidate_scores[1],
                        },
                        "D8": {
                            "redLogLossPerBall": d8_scores[0],
                            "redBrierPerBall": d8_scores[1],
                        },
                        "uniform": {
                            "redLogLossPerBall": uniform_scores[0],
                            "redBrierPerBall": uniform_scores[1],
                        },
                    },
                    "red8ActualOverlap": {
                        "candidate": candidate_overlap,
                        "D8": d8_overlap,
                        "C32IssueMean": sum(random_overlaps) / CONTROL_COUNT,
                    },
                    "hashes": {
                        "candidatePredictionSha256": candidate_prediction.prediction_fingerprint,
                        "candidateDocumentSha256": sha256_payload(candidate_document),
                        "D8DocumentSha256": sha256_payload(d8_document),
                    },
                }
            )

        for candidate_id, state in states.items():
            state.score_then_update(draw)
        incumbent.score_then_update(draw)

    validation_metrics = [
        accumulators[spec.candidate_id].report(uniform_validation)
        for spec in CANDIDATE_SPECS
    ]
    if selection_result is None:
        selection_result = select_candidate(validation_metrics)
        selected_raw = selection_result["selectedCandidateId"]
        selected_id = None if selected_raw is None else str(selected_raw)
    diagnostic = (
        None
        if selected_id is None
        else _diagnostic_report(selected_id, diagnostic_records)
    )
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedDeterministically": True,
        "researchOnly": True,
        "retrospective": True,
        "predictionClaim": False,
        "formalGate": False,
        "formalRecommendationStatus": "uniform_abstain",
        "automaticPromotion": False,
        "historicalPromotion": False,
        "futureChainRequired": True,
        "historicalEvidenceAlreadyViewed": True,
        "protocol": PROTOCOL,
        "protocolSha256": protocol_sha256(),
        "splits": fixed_split_boundaries(ordered),
        "selectionStatus": selection_result["selectionStatus"],
        "selectedCandidateId": selected_id,
        "validation": {
            "periods": VALIDATION_DRAWS,
            "candidateCount": len(validation_metrics),
            "candidates": validation_metrics,
            "auditSha256": sha256_payload(validation_audit),
        },
        "diagnostic": diagnostic,
        "dataSha256": sha256_payload(_data_payload(ordered)),
        "inputSha256": sha256_payload(
            {
                "data": sha256_payload(_data_payload(ordered)),
                "protocol": protocol_sha256(),
            }
        ),
        "auditFingerprints": {
            "validationPerIssueSha256": sha256_payload(validation_audit),
            "diagnosticPerIssueSha256": (
                None if selected_id is None else sha256_payload(diagnostic_records)
            ),
            "validationFirstIssueSha256": sha256_payload(validation_audit[0]),
            "validationLastIssueSha256": sha256_payload(validation_audit[-1]),
        },
    }
    report["reportSha256"] = sha256_payload(report)
    validate_selection_report(report)
    return report


def validate_selection_report(report: Mapping[str, object]) -> None:
    """验证 E2 选择报告的安全边界与八候选完整性。"""

    required = {
        "schemaVersion": SCHEMA_VERSION,
        "researchOnly": True,
        "retrospective": True,
        "formalGate": False,
        "formalRecommendationStatus": "uniform_abstain",
        "automaticPromotion": False,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"E2选择报告字段不匹配：{key}")
    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("E2选择报告缺少Validation")
    candidates = validation.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 8:
        raise ValueError("E2 Validation必须包含恰好8个候选")
    ids = [str(cast(Mapping[str, object], item)["candidateId"]) for item in candidates]
    if ids != [spec.candidate_id for spec in CANDIDATE_SPECS]:
        raise ValueError("E2 Validation候选ID或顺序不匹配")
    status = report.get("selectionStatus")
    selected = report.get("selectedCandidateId")
    diagnostic = report.get("diagnostic")
    if status == "selected":
        if selected not in CANDIDATE_BY_ID or not isinstance(diagnostic, Mapping):
            raise ValueError("E2已选择报告必须包含唯一候选Diagnostic")
        if (
            diagnostic.get("candidateId") != selected
            or diagnostic.get("promotionEvidence") is not False
        ):
            raise ValueError("E2 Diagnostic边界非法")
    elif status == "rejected":
        if selected is not None or diagnostic is not None:
            raise ValueError("E2拒绝报告不得包含候选或Diagnostic")
    else:
        raise ValueError("E2选择状态非法")


def build_selection_report(
    csv_path: str | Path, ensemble_report_path: str | Path
) -> dict[str, object]:
    """读取官方历史并在 ensemble 完整性保护下运行 E2。"""

    csv = Path(csv_path)
    ensemble_path = Path(ensemble_report_path)
    before = ensemble_path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    embedded_sha = str(json.loads(before)["reportSha256"])
    visible_draws = load_official_history_csv(csv)
    draws = _frozen_research_prefix(visible_draws)
    report = evaluate_selection(draws)
    report["input"] = {
        "csvPath": str(csv),
        "frozenPrefixSha256": _frozen_data_sha256(draws),
        "frozenThroughIssue": FROZEN_LAST_ISSUE,
        "periods": len(draws),
        "latestIssue": draws[-1].issue,
        "ensembleReportPath": str(ensemble_path),
        "ensembleFileSha256": before_sha,
        "ensembleReportSha256": embedded_sha,
    }
    report["ensembleIntegrity"] = {
        "bytesUnchanged": ensemble_path.read_bytes() == before,
        "fileSha256Unchanged": hashlib.sha256(ensemble_path.read_bytes()).hexdigest()
        == before_sha,
        "reportSha256Unchanged": str(
            json.loads(ensemble_path.read_bytes())["reportSha256"]
        )
        == embedded_sha,
    }
    if not all(cast(Mapping[str, bool], report["ensembleIntegrity"]).values()):
        raise RuntimeError("E2运行期间ensemble报告发生变化")
    report.pop("reportSha256")
    report["reportSha256"] = sha256_payload(report)
    return report


__all__ = [
    "DIAGNOSTIC_DRAWS",
    "SEARCH_DRAWS",
    "VALIDATION_DRAWS",
    "WARMUP_DRAWS",
    "assess_eligibility",
    "build_selection_report",
    "evaluate_selection",
    "fixed_split_boundaries",
    "select_candidate",
    "validate_selection_report",
]
