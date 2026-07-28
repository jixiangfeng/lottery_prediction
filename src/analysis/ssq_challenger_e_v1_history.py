# -*- coding: utf-8 -*-
# mypy: disable-error-code="arg-type,call-overload"
"""双色球 Challenger E1 全历史严格前序独立诊断。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import cast

from src.analysis.ssq_8red1blue_v1_history import (
    CONTROL_COUNT,
    Compound8,
    build_matched_random_control,
)
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    WARMUP_DRAWS,
    CalibratedRedChallengerEState,
)
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    protocol_sha256 as challenger_protocol_sha256,
)
from src.analysis.ssq_diversified_portfolio_v2 import (
    build_diversified_portfolio_v2,
)
from src.analysis.ssq_ensemble_v1 import FixedEnsembleState
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    protocol_sha256 as builder_protocol_sha256,
)

SCHEMA_VERSION = "ssq_challenger_e_v1_full_history"
RED_COUNT = 33
UNIFORM_RED_PROBABILITY = 6.0 / RED_COUNT
THRESHOLDS = (3, 4, 5, 6)
COMPOUND_METRICS = (
    "red8Overlap",
    *(f"red8AtLeast{threshold}" for threshold in THRESHOLDS),
    "blueHit",
    "exact6PlusBlue",
    "exact6PlusNoBlue",
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "strict_retrospective_diagnostic_only_no_historical_promotion",
    "walkForward": {
        "warmupDraws": WARMUP_DRAWS,
        "sequence": "obtain_E_and_incumbent_predictions_build_score_then_update",
        "evaluatedHistory": "all_periods_after_fixed_warmup",
        "futureLeakage": False,
    },
    "construction": {
        "B": "incumbent red and blue probabilities",
        "E": "E red probabilities plus incumbent blue probabilities plus same B",
        "D8": "incumbent red and blue probabilities plus same B",
        "controls": {
            "count": CONTROL_COUNT,
            "source": "existing D8 deterministic matched random 8-red plus 1-blue",
            "resultIndependent": True,
            "modelIndependent": True,
        },
    },
    "scores": {
        "red": "per-ball average Bernoulli LogLoss and Brier",
        "models": ["E", "incumbent", "uniform"],
        "ECalibration": "sum probabilities equals 6 every issue",
    },
    "diagnostics": {
        "compoundMetrics": list(COMPOUND_METRICS),
        "comparisons": ["E-D8", "E-C32_issue_mean"],
        "EAgainstD8Overlap": ["red8 intersection", "full-ticket intersection"],
        "officialPrizeClaims": False,
    },
    "claims": {
        "retrospective": True,
        "formalGate": False,
        "selection": None,
        "formalRecommendationStatus": "uniform_abstain",
        "historicalPromotion": False,
    },
    "fixed": {"cliTuning": False, "controlCount": CONTROL_COUNT},
}


def protocol_sha256() -> str:
    """返回 E1 历史诊断固定协议摘要。"""

    return _sha256_payload(PROTOCOL)


@dataclass
class _ProperAccumulator:
    observations: int = 0
    log_loss_total: float = 0.0
    brier_total: float = 0.0

    def add(self, log_loss: float, brier: float) -> None:
        self.observations += 1
        self.log_loss_total += log_loss
        self.brier_total += brier

    def report(self) -> dict[str, float | int]:
        if self.observations <= 0:
            raise ValueError("空 proper score 不能生成汇总")
        return {
            "observations": self.observations,
            "redLogLossPerBall": self.log_loss_total / self.observations,
            "redBrierPerBall": self.brier_total / self.observations,
        }


@dataclass
class _CompoundAccumulator:
    observations: int = 0
    totals: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in COMPOUND_METRICS}
    )
    red_overlap_distribution: list[int] = field(default_factory=lambda: [0] * 7)

    def add(self, metrics: Mapping[str, float]) -> None:
        self.observations += 1
        for name in COMPOUND_METRICS:
            self.totals[name] += metrics[name]
        self.red_overlap_distribution[int(metrics["red8Overlap"])] += 1

    def report(self) -> dict[str, object]:
        if self.observations <= 0:
            raise ValueError("空组合诊断不能生成汇总")
        return {
            "observations": self.observations,
            "metricsPerIssue": {
                name: self.totals[name] / self.observations for name in COMPOUND_METRICS
            },
            "red8OverlapDistribution": {
                str(index): count
                for index, count in enumerate(self.red_overlap_distribution)
            },
        }


def _ordered_draws(draws: Sequence[SSQDraw]) -> list[SSQDraw]:
    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) <= WARMUP_DRAWS:
        raise ValueError("双色球历史不足120期预热加至少1期诊断")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("双色球历史包含重复期号")
    return ordered


def _red_scores(
    probabilities: Sequence[float], red: Sequence[int]
) -> tuple[float, float]:
    if len(probabilities) != RED_COUNT:
        raise ValueError("红球概率必须恰好包含33项")
    selected = set(red)
    log_loss = 0.0
    brier = 0.0
    for ball, probability in enumerate(probabilities, start=1):
        if not 0.0 < probability < 1.0 or not math.isfinite(probability):
            raise ValueError("红球概率必须为有限开区间值")
        outcome = 1.0 if ball in selected else 0.0
        log_loss -= outcome * math.log(probability) + (1.0 - outcome) * math.log(
            1.0 - probability
        )
        brier += (probability - outcome) ** 2
    return log_loss / RED_COUNT, brier / RED_COUNT


def _compound_from_document(document: Mapping[str, object]) -> Compound8:
    red8 = tuple(int(value) for value in cast(list[object], document["red"]))
    blue = int(document["blue"])
    tickets = tuple((tuple(red6), blue) for red6 in combinations(red8, 6))
    return Compound8(red8=red8, blue=blue, tickets=tickets)


def _compound_metrics(compound: Compound8, draw: SSQDraw) -> dict[str, float]:
    overlap = len(set(compound.red8).intersection(draw.red))
    blue_hit = compound.blue == draw.blue
    metrics = {
        "red8Overlap": float(overlap),
        "blueHit": float(blue_hit),
        "exact6PlusBlue": float(overlap == 6 and blue_hit),
        "exact6PlusNoBlue": float(overlap == 6 and not blue_hit),
    }
    for threshold in THRESHOLDS:
        metrics[f"red8AtLeast{threshold}"] = float(overlap >= threshold)
    return metrics


def _mean_metrics(scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(scores) != CONTROL_COUNT:
        raise ValueError("每期必须恰好包含32个随机对照")
    return {
        name: sum(score[name] for score in scores) / CONTROL_COUNT
        for name in COMPOUND_METRICS
    }


def _b_tickets(document: Mapping[str, object]) -> set[tuple[tuple[int, ...], int]]:
    return {
        (
            tuple(int(value) for value in cast(list[object], ticket["red"])),
            int(ticket["blue"]),
        )
        for group in cast(list[Mapping[str, object]], document["groups"])
        for ticket in cast(list[Mapping[str, object]], group["expandedTickets"])
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


def _prediction_payload(
    issue: str,
    e_prediction_fingerprint: str,
    incumbent_red: Sequence[float],
    incumbent_blue: Sequence[float],
    b_document: Mapping[str, object],
    e_document: Mapping[str, object],
    d8_document: Mapping[str, object],
) -> dict[str, object]:
    return {
        "issue": issue,
        "E": e_prediction_fingerprint,
        "incumbentRed": _sha256_payload(list(incumbent_red)),
        "incumbentBlue": _sha256_payload(list(incumbent_blue)),
        "B": _sha256_payload(b_document),
        "E8": _sha256_payload(e_document),
        "D8": _sha256_payload(d8_document),
    }


def walk_forward_history_fingerprints(
    draws: Sequence[SSQDraw], stop: int | None = None
) -> list[str]:
    """返回逐期预测前缀指纹，验证未来追加不改变既有前缀。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("双色球历史包含重复期号")
    limit = len(ordered) if stop is None else min(stop, len(ordered))
    e_state = CalibratedRedChallengerEState()
    incumbent_state = FixedEnsembleState()
    prefix: list[dict[str, object]] = []
    fingerprints: list[str] = []
    for draw in ordered[:limit]:
        e_locked = e_state.predict()
        incumbent_red, incumbent_blue, _ = incumbent_state.predict()
        b_document = build_diversified_portfolio_v2(incumbent_red, incumbent_blue)
        e_document = build_small_compound_8red1blue_v1(
            e_locked.probabilities, incumbent_blue, b_document
        )
        d8_document = build_small_compound_8red1blue_v1(
            incumbent_red, incumbent_blue, b_document
        )
        prefix.append(
            _prediction_payload(
                draw.issue,
                e_locked.prediction_fingerprint,
                incumbent_red,
                incumbent_blue,
                b_document,
                e_document,
                d8_document,
            )
        )
        fingerprints.append(_sha256_payload(prefix))
        e_state.score_then_update(draw)
        incumbent_state.score_then_update(draw)
    return fingerprints


def evaluate_full_history(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """执行固定120期预热及其后全部时期的严格历史诊断。"""

    ordered = _ordered_draws(draws)
    e_state = CalibratedRedChallengerEState()
    incumbent_state = FixedEnsembleState()
    proper = {label: _ProperAccumulator() for label in ("E", "incumbent", "uniform")}
    compounds = {label: _CompoundAccumulator() for label in ("E", "D8", "C32")}
    e_minus_d8 = {name: 0.0 for name in COMPOUND_METRICS}
    e_minus_c32 = {name: 0.0 for name in COMPOUND_METRICS}
    e_d8_red_overlap_distribution = [0] * 9
    e_d8_ticket_overlap_distribution: dict[str, int] = {}
    prediction_prefix: list[dict[str, object]] = []
    scored_prefix: list[dict[str, object]] = []
    per_issue: list[dict[str, object]] = []

    for index, draw in enumerate(ordered):
        e_locked = e_state.predict()
        incumbent_red, incumbent_blue, _ = incumbent_state.predict()
        if abs(sum(e_locked.probabilities) - 6.0) > 1e-10:
            raise RuntimeError("E1逐期校准概率和不等于6")
        b_document = build_diversified_portfolio_v2(incumbent_red, incumbent_blue)
        e_document = build_small_compound_8red1blue_v1(
            e_locked.probabilities, incumbent_blue, b_document
        )
        d8_document = build_small_compound_8red1blue_v1(
            incumbent_red, incumbent_blue, b_document
        )
        prediction_record = _prediction_payload(
            draw.issue,
            e_locked.prediction_fingerprint,
            incumbent_red,
            incumbent_blue,
            b_document,
            e_document,
            d8_document,
        )
        prediction_prefix.append(prediction_record)
        prediction_prefix_sha = _sha256_payload(prediction_prefix)

        if index >= WARMUP_DRAWS:
            e_log_loss, e_brier = _red_scores(e_locked.probabilities, draw.red)
            incumbent_log_loss, incumbent_brier = _red_scores(incumbent_red, draw.red)
            uniform_log_loss, uniform_brier = _red_scores(
                [UNIFORM_RED_PROBABILITY] * RED_COUNT, draw.red
            )
            proper["E"].add(e_log_loss, e_brier)
            proper["incumbent"].add(incumbent_log_loss, incumbent_brier)
            proper["uniform"].add(uniform_log_loss, uniform_brier)

            e_compound = _compound_from_document(e_document)
            d8_compound = _compound_from_document(d8_document)
            e_metrics = _compound_metrics(e_compound, draw)
            d8_metrics = _compound_metrics(d8_compound, draw)
            control_scores = [
                _compound_metrics(
                    build_matched_random_control(draw.issue, control_index), draw
                )
                for control_index in range(CONTROL_COUNT)
            ]
            control_mean = _mean_metrics(control_scores)
            compounds["E"].add(e_metrics)
            compounds["D8"].add(d8_metrics)
            for control_score in control_scores:
                compounds["C32"].add(control_score)
            for name in COMPOUND_METRICS:
                e_minus_d8[name] += e_metrics[name] - d8_metrics[name]
                e_minus_c32[name] += e_metrics[name] - control_mean[name]

            e_ticket_set = set(e_compound.tickets)
            d8_ticket_set = set(d8_compound.tickets)
            b_ticket_set = _b_tickets(b_document)
            e_b_overlap = len(e_ticket_set.intersection(b_ticket_set))
            d8_b_overlap = len(d8_ticket_set.intersection(b_ticket_set))
            if e_b_overlap != 0 or d8_b_overlap != 0:
                raise RuntimeError("E或D8与同期B完整票发生重叠")
            e_d8_red_overlap = len(set(e_compound.red8).intersection(d8_compound.red8))
            e_d8_ticket_overlap = len(e_ticket_set.intersection(d8_ticket_set))
            e_d8_red_overlap_distribution[e_d8_red_overlap] += 1
            ticket_overlap_key = str(e_d8_ticket_overlap)
            e_d8_ticket_overlap_distribution[ticket_overlap_key] = (
                e_d8_ticket_overlap_distribution.get(ticket_overlap_key, 0) + 1
            )
            scored_record = {
                **prediction_record,
                "draw": {
                    "red": list(draw.red),
                    "blue": draw.blue,
                    "rawHash": draw.raw_hash,
                },
            }
            scored_prefix.append(scored_record)
            per_issue.append(
                {
                    "issue": draw.issue,
                    "date": draw.draw_date,
                    "prefixFingerprints": {
                        "predictionSha256": prediction_prefix_sha,
                        "scoredEvaluationSha256": _sha256_payload(scored_prefix),
                    },
                    "properScores": {
                        "E": {
                            "redLogLossPerBall": e_log_loss,
                            "redBrierPerBall": e_brier,
                            "sumRedProbabilities": sum(e_locked.probabilities),
                        },
                        "incumbent": {
                            "redLogLossPerBall": incumbent_log_loss,
                            "redBrierPerBall": incumbent_brier,
                            "sumRedProbabilities": sum(incumbent_red),
                        },
                        "uniform": {
                            "redLogLossPerBall": uniform_log_loss,
                            "redBrierPerBall": uniform_brier,
                            "sumRedProbabilities": 6.0,
                        },
                    },
                    "E": {
                        "red": list(e_compound.red8),
                        "blue": e_compound.blue,
                        "selectedCandidateRank": e_document["selectedCandidateRank"],
                        "metrics": e_metrics,
                        "documentSha256": _sha256_payload(e_document),
                    },
                    "D8": {
                        "red": list(d8_compound.red8),
                        "blue": d8_compound.blue,
                        "selectedCandidateRank": d8_document["selectedCandidateRank"],
                        "metrics": d8_metrics,
                        "documentSha256": _sha256_payload(d8_document),
                    },
                    "C32Mean": control_mean,
                    "overlaps": {
                        "EWithD8Red": e_d8_red_overlap,
                        "EWithD8Tickets": e_d8_ticket_overlap,
                        "EWithBTickets": e_b_overlap,
                        "D8WithBTickets": d8_b_overlap,
                    },
                    "hashes": {
                        "EStateSha256": e_locked.state_fingerprint,
                        "EPredictionSha256": e_locked.prediction_fingerprint,
                        "incumbentRedProbabilitiesSha256": _sha256_payload(
                            incumbent_red
                        ),
                        "incumbentBlueProbabilitiesSha256": _sha256_payload(
                            incumbent_blue
                        ),
                        "BDocumentSha256": _sha256_payload(b_document),
                    },
                }
            )

        e_state.score_then_update(draw)
        incumbent_state.score_then_update(draw)

    evaluated_periods = len(ordered) - WARMUP_DRAWS
    proper_summary = {
        label: accumulator.report() for label, accumulator in proper.items()
    }
    compound_summary = {
        label: accumulator.report() for label, accumulator in compounds.items()
    }
    e_proper = cast(dict[str, float], proper_summary["E"])
    incumbent_proper = cast(dict[str, float], proper_summary["incumbent"])
    uniform_proper = cast(dict[str, float], proper_summary["uniform"])
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedDeterministically": True,
        "researchOnly": True,
        "retrospective": True,
        "predictionClaim": False,
        "formalGate": False,
        "selection": None,
        "formalRecommendationStatus": "uniform_abstain",
        "historicalPromotion": False,
        "officialPrizeClaims": False,
        "protocol": PROTOCOL,
        "protocolSha256": protocol_sha256(),
        "challengerProtocolSha256": challenger_protocol_sha256(),
        "builderProtocolSha256": builder_protocol_sha256(),
        "dataSha256": _sha256_payload(_data_payload(ordered)),
        "history": {
            "totalPeriods": len(ordered),
            "warmupPeriods": WARMUP_DRAWS,
            "evaluatedPeriods": evaluated_periods,
            "firstEvaluatedIssue": ordered[WARMUP_DRAWS].issue,
            "firstEvaluatedDate": ordered[WARMUP_DRAWS].draw_date,
            "lastEvaluatedIssue": ordered[-1].issue,
            "lastEvaluatedDate": ordered[-1].draw_date,
        },
        "properScores": proper_summary,
        "properScoreDifferences": {
            "EMinusIncumbent": {
                "redLogLossPerBall": e_proper["redLogLossPerBall"]
                - incumbent_proper["redLogLossPerBall"],
                "redBrierPerBall": e_proper["redBrierPerBall"]
                - incumbent_proper["redBrierPerBall"],
            },
            "EMinusUniform": {
                "redLogLossPerBall": e_proper["redLogLossPerBall"]
                - uniform_proper["redLogLossPerBall"],
                "redBrierPerBall": e_proper["redBrierPerBall"]
                - uniform_proper["redBrierPerBall"],
            },
        },
        "compoundSummary": compound_summary,
        "compoundDifferences": {
            "EMinusD8Mean": {
                name: e_minus_d8[name] / evaluated_periods for name in COMPOUND_METRICS
            },
            "EMinusC32IssueMean": {
                name: e_minus_c32[name] / evaluated_periods for name in COMPOUND_METRICS
            },
        },
        "EAgainstD8Overlap": {
            "red8IntersectionDistribution": {
                str(index): count
                for index, count in enumerate(e_d8_red_overlap_distribution)
            },
            "meanRed8Intersection": sum(
                index * count
                for index, count in enumerate(e_d8_red_overlap_distribution)
            )
            / evaluated_periods,
            "meanFullTicketIntersection": sum(
                int(overlap) * count
                for overlap, count in e_d8_ticket_overlap_distribution.items()
            )
            / evaluated_periods,
            "fullTicketIntersectionDistribution": dict(
                sorted(
                    e_d8_ticket_overlap_distribution.items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "prefixFingerprints": {
            "allPredictionsSha256": _sha256_payload(prediction_prefix),
            "evaluatedScoredSha256": _sha256_payload(scored_prefix),
        },
        "perIssue": per_issue,
    }
    report["reportSha256"] = _sha256_payload(report)
    return report


def write_report(report: Mapping[str, object], output_path: str | Path) -> Path:
    """原子写入 E1 全历史诊断 JSON。"""

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
    "COMPOUND_METRICS",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "evaluate_full_history",
    "protocol_sha256",
    "walk_forward_history_fingerprints",
    "write_report",
]
