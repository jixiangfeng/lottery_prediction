# -*- coding: utf-8 -*-
# mypy: disable-error-code="arg-type,call-overload"
"""双色球 Challenger E1 独立仅未来 HMAC 评估链。"""

from __future__ import annotations

import hmac
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import cast

from src.analysis import ssq_8red1blue_v1_prospective as d8_chain
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    CalibratedRedChallengerEState,
    build_current_report,
    future_protocol_sha256,
)
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    protocol_sha256 as e_protocol_sha256,
)
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    sha256_payload as e_payload_sha256,
)
from src.analysis.ssq_calibrated_red_challenger_e_v1 import (
    train_state,
)
from src.analysis.ssq_diversified_portfolio_v2_prospective import (
    _atomic_replace_artifact,
    _draw_deadline,
    _fsync_directory,
    _next_target,
    _write_artifact,
    build_bound_ensemble_report,
    canonical_data_sha256,
    load_and_verify_artifact,
    payload_sha256,
)
from src.analysis.ssq_history import SSQDraw, load_official_history_csv
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    protocol_sha256 as builder_protocol_sha256,
)

SCHEMA_VERSION = "ssq_challenger_e_v1_prospective"
HORIZON = 500
BLOCK_SIZE = 100
BLOCK_COUNT = 5
DEFAULT_CANONICAL_CSV = Path("data/ssq/official_history.csv")
DEFAULT_E_REPORT = Path("reports/research/ssq_challenger_e_v1.json")
DEFAULT_ENSEMBLE_REPORT = Path("reports/research/ssq_ensemble_v1.json")
DEFAULT_STATE_DIR = Path("state/ssq_challenger_e_v1")
DEFAULT_D8_STATE_DIR = Path("state/ssq_8red1blue_v1")
FORMAL_RECOMMENDATION_STATUS = "uniform_abstain"

FIXED_METRICS: dict[str, object] = {
    "properScores": "per_ball_mean_bernoulli_logloss_and_brier_E_and_D8",
    "compound": [
        "red8Overlap",
        "atLeast3",
        "atLeast4",
        "atLeast5",
        "atLeast6",
        "blueHit",
        "exact6PlusBlue",
        "exact6PlusNoBlue",
    ],
    "paired": "E_minus_D8_per_issue",
    "blocks": {"count": BLOCK_COUNT, "size": BLOCK_SIZE, "fixed": True},
}

FIXED_GATES: dict[str, object] = {
    "evaluationTiming": "only_at_exactly_500_completed_issues",
    "safety": {"redLogLoss": "E<D8", "redBrier": "E<=D8", "required": "both"},
    "efficacy": {
        "pairedMeanRed8Overlap": ">0",
        "oneSidedAlpha": 0.025,
        "fixed100BlocksAtLeastNonnegative": 4,
        "minimumAnyBlock": -0.05,
        "red5Rate": "E>=D8",
        "red6": "record_only",
    },
    "promotion": {
        "automatic": False,
        "humanReviewRequired": True,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
    },
}

PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "independent_future_only_paired_E1_vs_D8_research_chain",
    "horizon": HORIZON,
    "settlement": "exactly_one_matching_issue_and_date_no_catchup_replay_or_skip",
    "predictionOrder": "lock_prediction_then_observe_score_update_and_generate_next",
    "integrity": {
        "artifacts": "canonical_json_sha256_plus_independent_hmac_sha256",
        "chain": "append_only_versions_and_previous_status_artifact_sha256",
        "pointer": "atomic_signed_current",
        "D8": "verified_existing_chain_read_only_or_deterministic_prefix_rebuild",
    },
    "metrics": FIXED_METRICS,
    "gates": FIXED_GATES,
    "claims": {
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "humanReviewRequired": True,
        "autoPromotion": False,
    },
}


def protocol_sha256() -> str:
    """返回 E1 prospective 固定协议摘要。"""

    return payload_sha256(PROTOCOL)


def _file_sha256(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unsigned(document: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in document.items()
        if key not in {"artifactSha256", "artifactHmacSha256"}
    }


def _load_json_object(path: str | Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须为JSON对象")
    return cast(dict[str, object], value)


def _verify_self_hashed_report(report: Mapping[str, object], label: str) -> None:
    claimed = report.get("reportSha256")
    unsigned = {key: value for key, value in report.items() if key != "reportSha256"}
    if not isinstance(claimed, str) or not hmac.compare_digest(
        claimed, payload_sha256(unsigned)
    ):
        raise ValueError(f"{label}自身SHA校验失败")


def _load_exact_reports(
    canonical_csv: str | Path,
    e_report_path: str | Path,
    ensemble_report_path: str | Path,
    draws: Sequence[SSQDraw],
) -> tuple[dict[str, object], dict[str, object]]:
    ensemble_before = Path(ensemble_report_path).read_bytes()
    e_before = Path(e_report_path).read_bytes()
    ensemble_report = _load_json_object(ensemble_report_path, "ssq_ensemble_v1报告")
    e_report = _load_json_object(e_report_path, "E1当前报告")
    _verify_self_hashed_report(ensemble_report, "ssq_ensemble_v1报告")
    _verify_self_hashed_report(e_report, "E1当前报告")
    if ensemble_report != build_bound_ensemble_report(draws):
        raise ValueError("ssq_ensemble_v1报告不是canonical边界的精确重算结果")
    if e_report != build_current_report(canonical_csv, ensemble_report_path):
        raise ValueError("E1当前报告不是canonical边界的精确重算结果")
    if Path(ensemble_report_path).read_bytes() != ensemble_before:
        raise RuntimeError("验证期间ssq_ensemble_v1报告字节发生变化")
    if Path(e_report_path).read_bytes() != e_before:
        raise RuntimeError("验证期间E1当前报告字节发生变化")
    return e_report, ensemble_report


def _ensemble_inputs(
    report: Mapping[str, object],
) -> tuple[list[float], list[float], Mapping[str, object]]:
    audit = report.get("auditMetadata")
    b_document = report.get("diversifiedPortfolioV2")
    if not isinstance(audit, Mapping) or not isinstance(b_document, Mapping):
        raise ValueError("ensemble报告缺少概率或B组合")
    probabilities = audit.get("finalNextProbabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError("ensemble报告缺少最终前序概率")
    red = probabilities.get("red")
    blue = probabilities.get("blue")
    if not isinstance(red, list) or not isinstance(blue, list):
        raise ValueError("ensemble概率字段非法")
    red_values = [float(value) for value in red]
    blue_values = [float(value) for value in blue]
    if len(red_values) != 33 or len(blue_values) != 16:
        raise ValueError("ensemble概率维度非法")
    return red_values, blue_values, b_document


def _ticket_set(document: Mapping[str, object]) -> set[tuple[tuple[int, ...], int]]:
    tickets = document.get("expandedTickets")
    if not isinstance(tickets, list):
        raise ValueError("8+1组合缺少完整28注")
    normalized: set[tuple[tuple[int, ...], int]] = set()
    for ticket in tickets:
        if not isinstance(ticket, Mapping):
            raise ValueError("8+1票必须为对象")
        red = ticket.get("red")
        blue = ticket.get("blue")
        if not isinstance(red, list) or not isinstance(blue, int):
            raise ValueError("8+1票字段非法")
        normalized.add((tuple(cast(list[int], red)), blue))
    if len(tickets) != 28 or len(normalized) != 28:
        raise ValueError("8+1组合必须恰好包含28张唯一票")
    return normalized


def _b_tickets(document: Mapping[str, object]) -> set[tuple[tuple[int, ...], int]]:
    groups = document.get("groups")
    if not isinstance(groups, list) or len(groups) != 5:
        raise ValueError("B组合必须包含5个分组")
    result: set[tuple[tuple[int, ...], int]] = set()
    nominal_count = 0
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("B分组必须为对象")
        tickets = group.get("expandedTickets")
        if not isinstance(tickets, list) or len(tickets) != 7:
            raise ValueError("B每个分组必须包含7张展开票")
        nominal_count += len(tickets)
        for ticket in tickets:
            if not isinstance(ticket, Mapping):
                raise ValueError("B票必须为对象")
            red = ticket.get("red")
            blue = ticket.get("blue")
            if not isinstance(red, list) or not isinstance(blue, int):
                raise ValueError("B票字段非法")
            result.add((tuple(cast(list[int], red)), blue))
    if nominal_count != 35 or len(result) != 35:
        raise ValueError("B组合必须恰好展开35张唯一票")
    return result


def _compound_payload(
    document: Mapping[str, object], b_document: Mapping[str, object]
) -> dict[str, object]:
    red = document.get("red")
    blue = document.get("blue")
    tickets = document.get("expandedTickets")
    audit = document.get("audit")
    if (
        not isinstance(red, list)
        or not isinstance(blue, int)
        or not isinstance(tickets, list)
        or not isinstance(audit, Mapping)
    ):
        raise ValueError("8+1组合结构非法")
    compound_tickets = _ticket_set(document)
    overlap = len(compound_tickets & _b_tickets(b_document))
    if len(red) != 8 or overlap != 0 or audit.get("overlapWithB") != 0:
        raise ValueError("8+1组合必须为8红、28注且与B零重叠")
    return {
        "red": list(cast(list[int], red)),
        "blue": blue,
        "tickets": tickets,
        "fixedCostMultiplier": 28,
        "selectedCandidateRank": document.get("selectedCandidateRank"),
        "portfolioSha256": payload_sha256(document),
        "ticketsSha256": payload_sha256(tickets),
        "BOverlapAudit": {
            "overlapWithB": overlap,
            "BPortfolioSha256": payload_sha256(b_document),
            "BExpandedTicketCount": len(_b_tickets(b_document)),
        },
    }


def _verified_d8_source(
    state_dir: str | Path, key: bytes
) -> tuple[dict[str, object], dict[str, object]]:
    manifest, snapshots, _ = d8_chain._verify_chain(Path(state_dir), key)
    return manifest, snapshots[-1]


def _locked_prediction_payload(
    draws: Sequence[SSQDraw],
    *,
    d8_manifest: Mapping[str, object],
    d8_snapshot: Mapping[str, object],
) -> dict[str, object]:
    ensemble_report = build_bound_ensemble_report(draws)
    d8_red_probabilities, blue_probabilities, b_document = _ensemble_inputs(
        ensemble_report
    )
    state = train_state(draws)
    locked = state.predict()
    e_document = build_small_compound_8red1blue_v1(
        locked.probabilities, blue_probabilities, b_document
    )
    d8_document = cast(
        Mapping[str, object], ensemble_report["smallCompound8Red1BlueV1"]
    )
    target_issue, target_date = _next_target(draws)
    d8_source_matches = (
        d8_snapshot.get("targetIssue") == target_issue
        and d8_snapshot.get("targetDate") == target_date
        and d8_snapshot.get("trainedThroughIssue") == draws[-1].issue
        and d8_snapshot.get("trainedThroughDate") == draws[-1].draw_date
    )
    if d8_source_matches:
        source_d8 = d8_snapshot.get("D8")
        if not isinstance(source_d8, Mapping):
            raise ValueError("D8活动快照缺少锁定组合")
        if source_d8.get("portfolioSha256") != payload_sha256(d8_document):
            raise ValueError("D8活动快照组合与canonical前缀重建不一致")
        source = "verified_current_D8_snapshot"
        source_snapshot_sha: str | None = cast(str, d8_snapshot["artifactSha256"])
    else:
        source = "deterministic_prefix_rebuild_after_verified_D8_chain"
        source_snapshot_sha = None
    e_payload = _compound_payload(e_document, b_document)
    d8_payload = _compound_payload(d8_document, b_document)
    e_red = set(cast(list[int], e_payload["red"]))
    d8_red = set(cast(list[int], d8_payload["red"]))
    return {
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "targetIssue": target_issue,
        "targetDate": target_date,
        "dataSha256": canonical_data_sha256(draws),
        "ensembleReportSha256": ensemble_report["reportSha256"],
        "ensembleProtocolSha256": ensemble_report["protocolSha256"],
        "BPortfolioSha256": payload_sha256(b_document),
        "E": {
            "redProbabilities": list(locked.probabilities),
            "redProbabilitiesSha256": e_payload_sha256(list(locked.probabilities)),
            "blueProbabilities": blue_probabilities,
            "blueProbabilitiesSha256": e_payload_sha256(blue_probabilities),
            "modelState": state.state_payload(include_pending=True),
            "modelStateSha256": locked.state_fingerprint,
            "featureFingerprint": payload_sha256(
                [list(row) for row in locked.features]
            ),
            "predictionFingerprint": locked.prediction_fingerprint,
            "lockedPrediction": locked.to_dict(),
            "portfolio": e_payload,
        },
        "D8": {
            "redProbabilities": d8_red_probabilities,
            "redProbabilitiesSha256": payload_sha256(d8_red_probabilities),
            "blueProbabilities": blue_probabilities,
            "blueProbabilitiesSha256": payload_sha256(blue_probabilities),
            "portfolio": d8_payload,
            "source": source,
            "sourceManifestArtifactSha256": d8_manifest["artifactSha256"],
            "sourceSnapshotArtifactSha256": source_snapshot_sha,
        },
        "pairedPortfolioAudit": {
            "sameFixedCost": e_payload["fixedCostMultiplier"]
            == d8_payload["fixedCostMultiplier"]
            == 28,
            "EWithD8RedOverlap": len(e_red & d8_red),
            "EWithD8TicketOverlap": len(
                _ticket_set(e_document) & _ticket_set(d8_document)
            ),
            "EWithBOverlap": cast(Mapping[str, object], e_payload["BOverlapAudit"])[
                "overlapWithB"
            ],
            "D8WithBOverlap": cast(Mapping[str, object], d8_payload["BOverlapAudit"])[
                "overlapWithB"
            ],
        },
    }


def _snapshot_payload(
    *,
    version: int,
    draws: Sequence[SSQDraw],
    manifest_sha256: str,
    previous_status_sha256: str | None,
    created_at: datetime,
    d8_manifest: Mapping[str, object],
    d8_snapshot: Mapping[str, object],
) -> dict[str, object]:
    locked = _locked_prediction_payload(
        draws, d8_manifest=d8_manifest, d8_snapshot=d8_snapshot
    )
    if created_at >= _draw_deadline(cast(str, locked["targetDate"])):
        raise ValueError("快照必须在锁定目标期开奖截止前生成")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "active_snapshot",
        "version": version,
        "createdAt": created_at.isoformat(timespec="seconds"),
        "completedBeforeSnapshot": version,
        "horizon": HORIZON,
        **locked,
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "humanReviewRequired": True,
        "autoPromotion": False,
    }


def _completed_snapshot_payload(
    *,
    version: int,
    draws: Sequence[SSQDraw],
    manifest_sha256: str,
    previous_status_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "completed_snapshot",
        "version": version,
        "createdAt": created_at.isoformat(timespec="seconds"),
        "completedBeforeSnapshot": version,
        "horizon": HORIZON,
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "targetIssue": None,
        "targetDate": None,
        "dataSha256": canonical_data_sha256(draws),
        "E": None,
        "D8": None,
        "pairedPortfolioAudit": None,
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "humanReviewRequired": True,
        "autoPromotion": False,
    }


def _draw_payload(draw: SSQDraw) -> dict[str, object]:
    return {
        "issue": draw.issue,
        "date": draw.draw_date,
        "red": list(draw.red),
        "blue": draw.blue,
        "sourceUrl": draw.source_url,
        "rawHash": draw.raw_hash,
    }


def _proper_scores(probabilities: Sequence[float], draw: SSQDraw) -> dict[str, float]:
    selected = set(draw.red)
    log_loss = 0.0
    brier = 0.0
    for ball, probability in enumerate(probabilities, start=1):
        clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
        outcome = 1.0 if ball in selected else 0.0
        log_loss -= outcome * math.log(clipped) + (1.0 - outcome) * math.log(
            1.0 - clipped
        )
        brier += (clipped - outcome) ** 2
    return {"redLogLossPerBall": log_loss / 33.0, "redBrierPerBall": brier / 33.0}


def _compound_scores(
    portfolio: Mapping[str, object], draw: SSQDraw
) -> dict[str, object]:
    red = portfolio.get("red")
    blue = portfolio.get("blue")
    if not isinstance(red, list) or not isinstance(blue, int):
        raise ValueError("锁定组合字段非法")
    overlap = len(set(cast(list[int], red)) & set(draw.red))
    blue_hit = blue == draw.blue
    return {
        "red8Overlap": overlap,
        "atLeast3": overlap >= 3,
        "atLeast4": overlap >= 4,
        "atLeast5": overlap >= 5,
        "atLeast6": overlap >= 6,
        "blueHit": blue_hit,
        "exact6PlusBlue": overlap == 6 and blue_hit,
        "exact6PlusNoBlue": overlap == 6 and not blue_hit,
    }


def _observation_payload(
    version: int, snapshot: Mapping[str, object] | None, draw: SSQDraw | None
) -> dict[str, object]:
    if version == 0:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "genesis_no_result",
            "version": 0,
            "settlesSnapshotVersion": None,
            "snapshotArtifactSha256": None,
            "officialResult": None,
            "scores": None,
        }
    if snapshot is None or draw is None:
        raise ValueError("结算观测缺少锁定快照或官方结果")
    e_payload = snapshot.get("E")
    d8_payload = snapshot.get("D8")
    if not isinstance(e_payload, Mapping) or not isinstance(d8_payload, Mapping):
        raise ValueError("活动快照缺少E/D8锁定内容")
    e_probabilities = cast(list[float], e_payload["redProbabilities"])
    d8_probabilities = cast(list[float], d8_payload["redProbabilities"])
    e_scores = {
        **_proper_scores(e_probabilities, draw),
        **_compound_scores(cast(Mapping[str, object], e_payload["portfolio"]), draw),
    }
    d8_scores = {
        **_proper_scores(d8_probabilities, draw),
        **_compound_scores(cast(Mapping[str, object], d8_payload["portfolio"]), draw),
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "settled_exactly_one_issue",
        "version": version,
        "settlesSnapshotVersion": version - 1,
        "snapshotArtifactSha256": snapshot["artifactSha256"],
        "officialResult": _draw_payload(draw),
        "scores": {
            "E": e_scores,
            "D8": d8_scores,
            "pairedEMinusD8": {
                "redLogLossPerBall": cast(float, e_scores["redLogLossPerBall"])
                - cast(float, d8_scores["redLogLossPerBall"]),
                "redBrierPerBall": cast(float, e_scores["redBrierPerBall"])
                - cast(float, d8_scores["redBrierPerBall"]),
                "red8Overlap": cast(int, e_scores["red8Overlap"])
                - cast(int, d8_scores["red8Overlap"]),
            },
        },
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _side_metrics(
    observations: Sequence[Mapping[str, object]], label: str
) -> dict[str, object]:
    scores = [
        cast(Mapping[str, object], cast(Mapping[str, object], item["scores"])[label])
        for item in observations
    ]
    return {
        "redLogLossPerBall": _mean(
            [float(score["redLogLossPerBall"]) for score in scores]
        ),
        "redBrierPerBall": _mean([float(score["redBrierPerBall"]) for score in scores]),
        "meanRed8Overlap": _mean([float(score["red8Overlap"]) for score in scores]),
        "atLeast3Rate": _mean([float(bool(score["atLeast3"])) for score in scores]),
        "atLeast4Rate": _mean([float(bool(score["atLeast4"])) for score in scores]),
        "atLeast5Rate": _mean([float(bool(score["atLeast5"])) for score in scores]),
        "atLeast6Rate": _mean([float(bool(score["atLeast6"])) for score in scores]),
        "blueHitRate": _mean([float(bool(score["blueHit"])) for score in scores]),
        "exact6PlusBlueRate": _mean(
            [float(bool(score["exact6PlusBlue"])) for score in scores]
        ),
        "exact6PlusNoBlueRate": _mean(
            [float(bool(score["exact6PlusNoBlue"])) for score in scores]
        ),
    }


def _paired_metrics(observations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paired = [
        cast(
            Mapping[str, object],
            cast(Mapping[str, object], item["scores"])["pairedEMinusD8"],
        )
        for item in observations
    ]
    overlaps = [float(item["red8Overlap"]) for item in paired]
    block_means = [
        _mean(overlaps[index : index + BLOCK_SIZE])
        for index in range(0, len(overlaps), BLOCK_SIZE)
    ]
    return {
        "meanRedLogLossPerBall": _mean(
            [float(item["redLogLossPerBall"]) for item in paired]
        ),
        "meanRedBrierPerBall": _mean(
            [float(item["redBrierPerBall"]) for item in paired]
        ),
        "meanRed8Overlap": _mean(overlaps),
        "fixed100BlockMeans": block_means,
    }


def _gate_evaluation(
    completed: int,
    observations: Sequence[Mapping[str, object]],
    e_metrics: Mapping[str, object],
    d8_metrics: Mapping[str, object],
    paired: Mapping[str, object],
) -> dict[str, object] | None:
    if completed != HORIZON:
        return None
    differences = [
        float(
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], item["scores"])["pairedEMinusD8"],
            )["red8Overlap"]
        )
        for item in observations
    ]
    mean_difference = _mean(differences)
    variance = sum((value - mean_difference) ** 2 for value in differences) / (
        len(differences) - 1
    )
    standard_error = math.sqrt(variance / len(differences))
    if standard_error == 0.0:
        p_value = 0.0 if mean_difference > 0.0 else 1.0
    else:
        p_value = 1.0 - NormalDist().cdf(mean_difference / standard_error)
    blocks = cast(list[float], paired["fixed100BlockMeans"])
    checks = {
        "safetyRedLogLoss": float(e_metrics["redLogLossPerBall"])
        < float(d8_metrics["redLogLossPerBall"]),
        "safetyRedBrier": float(e_metrics["redBrierPerBall"])
        <= float(d8_metrics["redBrierPerBall"]),
        "pairedMeanPositive": mean_difference > 0.0,
        "oneSidedPValueBelowAlpha": p_value < 0.025,
        "fourOfFiveBlocksNonnegative": sum(value >= 0.0 for value in blocks) >= 4,
        "minimumBlock": min(blocks) >= -0.05,
        "red5Rate": float(e_metrics["atLeast5Rate"])
        >= float(d8_metrics["atLeast5Rate"]),
    }
    return {
        "evaluated": True,
        "fixedGates": FIXED_GATES,
        "oneSidedNormalApproximationPValue": p_value,
        "checks": checks,
        "allResearchGatesPassed": all(checks.values()),
        "autoPromotion": False,
        "humanReviewRequired": True,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
    }


def _status_payload(
    *,
    version: int,
    snapshot: Mapping[str, object],
    observation: Mapping[str, object],
    settled_observations: Sequence[Mapping[str, object]],
    manifest_sha256: str,
    previous_status_sha256: str | None,
) -> dict[str, object]:
    e_metrics = _side_metrics(settled_observations, "E")
    d8_metrics = _side_metrics(settled_observations, "D8")
    paired = _paired_metrics(settled_observations)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "completed": version,
        "horizon": HORIZON,
        "remaining": HORIZON - version,
        "snapshotArtifactSha256": snapshot["artifactSha256"],
        "observationArtifactSha256": observation["artifactSha256"],
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "targetIssue": snapshot.get("targetIssue"),
        "targetDate": snapshot.get("targetDate"),
        "metrics": {"E": e_metrics, "D8": d8_metrics, "pairedEMinusD8": paired},
        "gates": _gate_evaluation(
            version, settled_observations, e_metrics, d8_metrics, paired
        ),
        "researchOnly": True,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "humanReviewRequired": True,
        "autoPromotion": False,
    }


def _publish_version(
    root: Path,
    *,
    version: int,
    snapshot_payload: Mapping[str, object],
    observation_payload: Mapping[str, object],
    settled_observations: Sequence[Mapping[str, object]],
    manifest_sha256: str,
    previous_status_sha256: str | None,
    key: bytes,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    versions = root / "versions"
    final_dir = versions / f"{version:04d}"
    staging = versions / f".{version:04d}.tmp"
    if final_dir.exists() or staging.exists():
        raise FileExistsError("E1 prospective目标版本已存在或存在待审计临时目录")
    staging.mkdir(mode=0o700)
    try:
        snapshot = _write_artifact(staging / "snapshot.json", snapshot_payload, key)
        observation = _write_artifact(
            staging / "observation.json", observation_payload, key
        )
        status = _write_artifact(
            staging / "status.json",
            _status_payload(
                version=version,
                snapshot=snapshot,
                observation=observation,
                settled_observations=settled_observations,
                manifest_sha256=manifest_sha256,
                previous_status_sha256=previous_status_sha256,
            ),
            key,
        )
        _fsync_directory(staging)
        os.replace(staging, final_dir)
        _fsync_directory(versions)
        try:
            _atomic_replace_artifact(
                root / "current.json",
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "version": version,
                    "statusArtifactSha256": status["artifactSha256"],
                },
                key,
            )
        except BaseException:
            shutil.rmtree(final_dir)
            _fsync_directory(versions)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return snapshot, observation, status


def register_prospective(
    canonical_csv: str | Path,
    e_report: str | Path,
    ensemble_report: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    d8_state_dir: str | Path = DEFAULT_D8_STATE_DIR,
    hmac_key: bytes,
    d8_hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """登记唯一边界；不创建真实首快照。"""

    root = Path(state_dir)
    if root.exists():
        raise FileExistsError("E1 prospective状态目录已存在，register只允许一次")
    draws = load_official_history_csv(canonical_csv)
    e_document, ensemble_document = _load_exact_reports(
        canonical_csv, e_report, ensemble_report, draws
    )
    d8_manifest, d8_snapshot = _verified_d8_source(d8_state_dir, d8_hmac_key)
    locked = _locked_prediction_payload(
        draws, d8_manifest=d8_manifest, d8_snapshot=d8_snapshot
    )
    created_at = d8_chain._now(now)
    if created_at >= _draw_deadline(cast(str, locked["targetDate"])):
        raise ValueError("登记时间已越过目标期开奖截止")
    staging = root.with_name(f".{root.name}.register.tmp")
    if staging.exists():
        raise FileExistsError("存在E1登记临时目录，需人工审计")
    staging.mkdir(parents=True, mode=0o700)
    try:
        protocol = _write_artifact(
            staging / "protocol.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "prospectiveProtocolSha256": protocol_sha256(),
                "ECoreProtocolSha256": e_protocol_sha256(),
                "EFutureProtocolSha256": future_protocol_sha256(),
                "builderProtocolSha256": builder_protocol_sha256(),
                "ensembleProtocolSha256": ensemble_document["protocolSha256"],
                "D8ProspectiveProtocolSha256": d8_chain.protocol_sha256(),
            },
            hmac_key,
        )
        _write_artifact(
            staging / "manifest.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "registeredAt": created_at.isoformat(timespec="seconds"),
                "registeredLatestIssue": draws[-1].issue,
                "registeredLatestDate": draws[-1].draw_date,
                "initialHistoryRows": len(draws),
                "initialTargetIssue": locked["targetIssue"],
                "initialTargetDate": locked["targetDate"],
                "canonicalDataSha256": canonical_data_sha256(draws),
                "canonicalFileSha256": _file_sha256(canonical_csv),
                "verifiedEReportSha256": e_document["reportSha256"],
                "verifiedEReportFileSha256": _file_sha256(e_report),
                "verifiedEnsembleReportSha256": ensemble_document["reportSha256"],
                "verifiedEnsembleReportFileSha256": _file_sha256(ensemble_report),
                "initialEPortfolioSha256": cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], locked["E"])["portfolio"],
                )["portfolioSha256"],
                "initialD8PortfolioSha256": cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], locked["D8"])["portfolio"],
                )["portfolioSha256"],
                "initialD8ManifestArtifactSha256": d8_manifest["artifactSha256"],
                "initialD8SnapshotArtifactSha256": d8_snapshot["artifactSha256"],
                "horizon": HORIZON,
                "fixedMetrics": FIXED_METRICS,
                "fixedGates": FIXED_GATES,
                "ECoreProtocolSha256": e_protocol_sha256(),
                "EFutureProtocolSha256": future_protocol_sha256(),
                "protocolArtifactSha256": protocol["artifactSha256"],
                "researchOnly": True,
                "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
                "humanReviewRequired": True,
                "autoPromotion": False,
            },
            hmac_key,
        )
        (staging / "versions").mkdir(mode=0o700)
        _atomic_replace_artifact(
            staging / "current.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "version": None,
                "statusArtifactSha256": None,
            },
            hmac_key,
        )
        _fsync_directory(staging)
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, root)
        _fsync_directory(root.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "action": "registered",
        "stateChanged": True,
        "registeredLatestIssue": draws[-1].issue,
        "registeredLatestDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "targetIssue": locked["targetIssue"],
        "targetDate": locked["targetDate"],
        "snapshotCreated": False,
        "autoPromotion": False,
    }


def _verify_registration(
    root: Path, key: bytes
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = load_and_verify_artifact(root / "protocol.json", key)
    manifest = load_and_verify_artifact(root / "manifest.json", key)
    if protocol.get("prospectiveProtocolSha256") != protocol_sha256():
        raise ValueError("E1 prospective协议摘要不一致")
    if protocol.get("ECoreProtocolSha256") != e_protocol_sha256():
        raise ValueError("E1核心协议摘要不一致")
    if protocol.get("EFutureProtocolSha256") != future_protocol_sha256():
        raise ValueError("E1未来协议摘要不一致")
    if manifest.get("protocolArtifactSha256") != protocol.get("artifactSha256"):
        raise ValueError("E1 manifest未绑定protocol")
    if (
        manifest.get("horizon") != HORIZON
        or manifest.get("fixedMetrics") != FIXED_METRICS
        or manifest.get("fixedGates") != FIXED_GATES
    ):
        raise ValueError("E1 manifest冻结指标或门槛不一致")
    return protocol, manifest


def create_snapshot(
    canonical_csv: str | Path,
    e_report: str | Path,
    ensemble_report: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    d8_state_dir: str | Path = DEFAULT_D8_STATE_DIR,
    hmac_key: bytes,
    d8_hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """生成唯一 0000 预开奖快照。"""

    root = Path(state_dir)
    _, manifest = _verify_registration(root, hmac_key)
    if any((root / "versions").iterdir()):
        raise FileExistsError("E1首快照已存在")
    current = load_and_verify_artifact(root / "current.json", hmac_key)
    if current.get("version") is not None:
        raise ValueError("E1 current指针与空版本目录不一致")
    draws = load_official_history_csv(canonical_csv)
    if (
        len(draws) != manifest["initialHistoryRows"]
        or canonical_data_sha256(draws) != manifest["canonicalDataSha256"]
    ):
        raise ValueError("snapshot要求canonical仍停留在登记边界")
    e_document, ensemble_document = _load_exact_reports(
        canonical_csv, e_report, ensemble_report, draws
    )
    if e_document["reportSha256"] != manifest["verifiedEReportSha256"]:
        raise ValueError("snapshot E1报告与登记报告不一致")
    if ensemble_document["reportSha256"] != manifest["verifiedEnsembleReportSha256"]:
        raise ValueError("snapshot ensemble报告与登记报告不一致")
    d8_manifest, d8_snapshot = _verified_d8_source(d8_state_dir, d8_hmac_key)
    snapshot_payload = _snapshot_payload(
        version=0,
        draws=draws,
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=None,
        created_at=d8_chain._now(now),
        d8_manifest=d8_manifest,
        d8_snapshot=d8_snapshot,
    )
    snapshot, _, _ = _publish_version(
        root,
        version=0,
        snapshot_payload=snapshot_payload,
        observation_payload=_observation_payload(0, None, None),
        settled_observations=[],
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=None,
        key=hmac_key,
    )
    return {
        "action": "snapshot_created",
        "stateChanged": True,
        "version": 0,
        "completed": 0,
        "horizon": HORIZON,
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "E": snapshot["E"],
        "D8": snapshot["D8"],
        "autoPromotion": False,
    }


def _versions(root: Path) -> list[int]:
    values: list[int] = []
    for path in (root / "versions").iterdir():
        if not path.is_dir() or not path.name.isdigit() or len(path.name) != 4:
            raise ValueError("E1版本目录存在非冻结条目")
        values.append(int(path.name))
    values.sort()
    if values != list(range(len(values))):
        raise ValueError("E1版本号不连续")
    return values


def _verify_snapshot_prefix(
    snapshot: Mapping[str, object], draws: Sequence[SSQDraw]
) -> None:
    if snapshot.get("kind") != "active_snapshot":
        raise ValueError("待结算版本不是活动快照")
    e_payload = snapshot.get("E")
    d8_payload = snapshot.get("D8")
    if not isinstance(e_payload, Mapping) or not isinstance(d8_payload, Mapping):
        raise ValueError("活动快照缺少E/D8")
    ensemble_report = build_bound_ensemble_report(draws)
    d8_red, blue, b_document = _ensemble_inputs(ensemble_report)
    state = train_state(draws)
    locked = state.predict()
    e_document = build_small_compound_8red1blue_v1(
        locked.probabilities, blue, b_document
    )
    d8_document = cast(
        Mapping[str, object], ensemble_report["smallCompound8Red1BlueV1"]
    )
    expected_pairs = {
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "dataSha256": canonical_data_sha256(draws),
        "targetIssue": _next_target(draws)[0],
        "targetDate": _next_target(draws)[1],
    }
    for key, expected in expected_pairs.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"锁定快照{key}与canonical前缀不一致")
    expected_e = {
        "redProbabilities": list(locked.probabilities),
        "redProbabilitiesSha256": e_payload_sha256(list(locked.probabilities)),
        "blueProbabilities": blue,
        "blueProbabilitiesSha256": e_payload_sha256(blue),
        "modelState": state.state_payload(include_pending=True),
        "modelStateSha256": locked.state_fingerprint,
        "featureFingerprint": payload_sha256([list(row) for row in locked.features]),
        "predictionFingerprint": locked.prediction_fingerprint,
        "lockedPrediction": locked.to_dict(),
        "portfolio": _compound_payload(e_document, b_document),
    }
    for key, expected in expected_e.items():
        if e_payload.get(key) != expected:
            raise ValueError(f"E锁定快照{key}重建不一致")
    expected_d8 = {
        "redProbabilities": d8_red,
        "redProbabilitiesSha256": payload_sha256(d8_red),
        "blueProbabilities": blue,
        "blueProbabilitiesSha256": payload_sha256(blue),
        "portfolio": _compound_payload(d8_document, b_document),
    }
    for key, expected in expected_d8.items():
        if d8_payload.get(key) != expected:
            raise ValueError(f"D8锁定快照{key}重建不一致")
    restored = CalibratedRedChallengerEState.from_dict(
        cast(Mapping[str, object], e_payload["modelState"])
    )
    if restored.fingerprint() != locked.state_fingerprint:
        raise ValueError("E完整状态往返指纹不一致")


def _verify_chain(root: Path, key: bytes, draws: Sequence[SSQDraw]) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    _, manifest = _verify_registration(root, key)
    versions = _versions(root)
    if not versions:
        raise ValueError("E1尚未创建首快照")
    initial_rows = cast(int, manifest["initialHistoryRows"])
    completed = len(versions) - 1
    minimum_rows = initial_rows + completed
    if len(draws) not in {minimum_rows, minimum_rows + 1}:
        raise ValueError("canonical长度与E1链边界不相容；禁止catchup/replay/skip")
    snapshots: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    previous_status_sha: str | None = None
    for version in versions:
        directory = root / "versions" / f"{version:04d}"
        if {path.name for path in directory.iterdir()} != {
            "snapshot.json",
            "observation.json",
            "status.json",
        }:
            raise ValueError(f"E1版本{version:04d}文件集合不符合冻结布局")
        snapshot = load_and_verify_artifact(directory / "snapshot.json", key)
        observation = load_and_verify_artifact(directory / "observation.json", key)
        status = load_and_verify_artifact(directory / "status.json", key)
        if any(
            item.get("version") != version for item in (snapshot, observation, status)
        ):
            raise ValueError(f"E1版本{version:04d}内部版本号不一致")
        if snapshot.get("previousStatusArtifactSha256") != previous_status_sha:
            raise ValueError(f"E1版本{version:04d}快照链断裂")
        if status.get("previousStatusArtifactSha256") != previous_status_sha:
            raise ValueError(f"E1版本{version:04d}状态链断裂")
        if status.get("snapshotArtifactSha256") != snapshot.get("artifactSha256"):
            raise ValueError(f"E1版本{version:04d}状态未绑定快照")
        if status.get("observationArtifactSha256") != observation.get("artifactSha256"):
            raise ValueError(f"E1版本{version:04d}状态未绑定观测")
        if version == 0:
            if observation.get("kind") != "genesis_no_result":
                raise ValueError("E1 0000 observation不得包含结果")
        else:
            previous_snapshot = snapshots[-1]
            expected_draw = draws[initial_rows + version - 1]
            if observation.get("settlesSnapshotVersion") != version - 1:
                raise ValueError("E1 observation未结算紧邻前一快照")
            if observation.get("snapshotArtifactSha256") != previous_snapshot.get(
                "artifactSha256"
            ):
                raise ValueError("E1 observation未绑定被结算快照")
            expected_observation = _observation_payload(
                version, previous_snapshot, expected_draw
            )
            if _unsigned(observation) != expected_observation:
                raise ValueError("E1 observation锁定评分复算不一致")
        if snapshot.get("kind") == "active_snapshot":
            prefix_rows = initial_rows + version
            _verify_snapshot_prefix(snapshot, draws[:prefix_rows])
        elif version != HORIZON or snapshot.get("kind") != "completed_snapshot":
            raise ValueError("E1完成快照仅允许出现在500期")
        snapshots.append(snapshot)
        observations.append(observation)
        settled = observations[1:]
        expected_status = _status_payload(
            version=version,
            snapshot=snapshot,
            observation=observation,
            settled_observations=settled,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=previous_status_sha,
        )
        if _unsigned(status) != expected_status:
            raise ValueError(f"E1版本{version:04d}累计状态复算不一致")
        previous_status_sha = cast(str, status["artifactSha256"])
        statuses.append(status)
    current = load_and_verify_artifact(root / "current.json", key)
    if current.get("version") != versions[-1]:
        raise ValueError("E1 current版本指针不匹配最新提交")
    if current.get("statusArtifactSha256") != statuses[-1].get("artifactSha256"):
        raise ValueError("E1 current未绑定最新status")
    return manifest, snapshots[-1], observations, statuses


def update_prospective(
    canonical_csv: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    d8_state_dir: str | Path = DEFAULT_D8_STATE_DIR,
    hmac_key: bytes,
    d8_hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """恰好结算一个锁定目标，并由前缀生成下一 E/D8 配对快照。"""

    root = Path(state_dir)
    draws = load_official_history_csv(canonical_csv)
    manifest, current_snapshot, observations, statuses = _verify_chain(
        root, hmac_key, draws
    )
    completed = len(observations) - 1
    if completed >= HORIZON:
        raise ValueError("E1 prospective链已完成500期")
    expected_rows = cast(int, manifest["initialHistoryRows"]) + completed + 1
    if len(draws) != expected_rows:
        raise ValueError("update要求canonical相对链尾恰好新增一期")
    draw = draws[-1]
    if draw.issue != current_snapshot.get(
        "targetIssue"
    ) or draw.draw_date != current_snapshot.get("targetDate"):
        raise ValueError("新增开奖与锁定targetIssue/targetDate不一致")
    created_at = d8_chain._now(now)
    if created_at < _draw_deadline(draw.draw_date):
        raise ValueError("目标期开奖截止前禁止结算")
    _verify_snapshot_prefix(current_snapshot, draws[:-1])
    version = completed + 1
    previous_status_sha = cast(str, statuses[-1]["artifactSha256"])
    if version == HORIZON:
        snapshot_payload = _completed_snapshot_payload(
            version=version,
            draws=draws,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=previous_status_sha,
            created_at=created_at,
        )
    else:
        d8_manifest, d8_snapshot = _verified_d8_source(d8_state_dir, d8_hmac_key)
        snapshot_payload = _snapshot_payload(
            version=version,
            draws=draws,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=previous_status_sha,
            created_at=created_at,
            d8_manifest=d8_manifest,
            d8_snapshot=d8_snapshot,
        )
    observation_payload = _observation_payload(version, current_snapshot, draw)
    snapshot, observation, status = _publish_version(
        root,
        version=version,
        snapshot_payload=snapshot_payload,
        observation_payload=observation_payload,
        settled_observations=[*observations[1:], observation_payload],
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=previous_status_sha,
        key=hmac_key,
    )
    return {
        "action": "updated",
        "stateChanged": True,
        "version": version,
        "completed": version,
        "horizon": HORIZON,
        "settledIssue": draw.issue,
        "scores": observation["scores"],
        "targetIssue": snapshot.get("targetIssue"),
        "targetDate": snapshot.get("targetDate"),
        "metrics": status["metrics"],
        "gates": status["gates"],
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "humanReviewRequired": True,
        "autoPromotion": False,
    }


def prospective_status(
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    canonical_csv: str | Path,
    hmac_key: bytes,
) -> dict[str, object]:
    """验签完整 E1 链并返回只读配对指标。"""

    draws = load_official_history_csv(canonical_csv)
    _, snapshot, observations, statuses = _verify_chain(
        Path(state_dir), hmac_key, draws
    )
    completed = len(observations) - 1
    expected_rows = (
        cast(
            int,
            load_and_verify_artifact(Path(state_dir) / "manifest.json", hmac_key)[
                "initialHistoryRows"
            ],
        )
        + completed
    )
    if len(draws) not in {expected_rows, expected_rows + 1}:
        raise ValueError("canonical与E1链边界不相容")
    latest_status = statuses[-1]
    return {
        "action": "status",
        "stateChanged": False,
        "completed": completed,
        "horizon": HORIZON,
        "remaining": HORIZON - completed,
        "targetIssue": snapshot.get("targetIssue"),
        "targetDate": snapshot.get("targetDate"),
        "pendingExactOneIssueUpdate": len(draws) == expected_rows + 1,
        "metrics": latest_status["metrics"],
        "fixed100Blocks": cast(Mapping[str, object], latest_status["metrics"])[
            "pairedEMinusD8"
        ],
        "gates": latest_status["gates"],
        "gatesEvaluated": completed == HORIZON,
        "researchOnly": True,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "humanReviewRequired": True,
        "autoPromotion": False,
    }


__all__ = [
    "DEFAULT_CANONICAL_CSV",
    "DEFAULT_D8_STATE_DIR",
    "DEFAULT_ENSEMBLE_REPORT",
    "DEFAULT_E_REPORT",
    "DEFAULT_STATE_DIR",
    "HORIZON",
    "create_snapshot",
    "prospective_status",
    "protocol_sha256",
    "register_prospective",
    "update_prospective",
]
