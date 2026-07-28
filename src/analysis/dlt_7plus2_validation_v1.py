# mypy: disable-error-code="arg-type,attr-defined,index,call-overload"
"""DLT 7+2 v1 Validation-only controller（零基索引 1901:2401）。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

from src.analysis.dlt_7plus2_v1 import FROZEN_PROTOCOL_SHA256
from src.analysis.dlt_fixed_cardinality_v1 import FixedCardinalityDistribution
from src.analysis.dlt_ridge_candidates_v1 import (
    BACK_SIZE,
    BLOCK_SIZE,
    CANDIDATE_IDS,
    FRONT_SIZE,
    CandidateScores,
    DLTDrawLike,
    FeatureSnapshotCache,
    fit_candidate_scores_from_cache,
)

VALIDATION_START = 1901
VALIDATION_STOP = 2401
VALIDATION_COUNT = 500
BOOTSTRAP_BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATES = 50_000
RANDOM_CONTROL_COUNT = 512
EXPECTED_SEARCH_REPORT_SHA256 = (
    "ef76a1fa1c35d38f2b92c9ded8d7d56180961b3580fd0bbe2ee5b651a7b0cfa7"
)
UNIFORM_FRONT_SET_LL = math.log(math.comb(35, 5))
UNIFORM_BACK_SET_LL = math.log(math.comb(12, 2))
UNIFORM_JOINT_LL = UNIFORM_FRONT_SET_LL + UNIFORM_BACK_SET_LL
UNIFORM_FRONT_BRIER = 6.0 / 49.0
UNIFORM_BACK_BRIER = 5.0 / 36.0
THRESHOLDS = tuple((front, back) for front in (3, 4, 5) for back in (0, 1, 2))


class ValidationDraw:
    def __init__(
        self, issue: str, front: tuple[int, ...], back: tuple[int, ...]
    ) -> None:
        self.issue = issue
        self.front = front
        self.back = back


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mean(values: Sequence[float] | np.ndarray) -> float:
    return math.fsum(float(value) for value in values) / len(values)


def global_validation_cutoffs() -> tuple[int, ...]:
    """返回覆盖 Validation 的严格全局 25 期拟合边界。"""

    return tuple(range(1900, VALIDATION_STOP, BLOCK_SIZE))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down 调整，保持输入候选顺序。"""

    count = len(p_values)
    if count == 0 or any(not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("Holm p 值必须是非空的 [0,1] 序列")
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def moving_block_bootstrap_p(
    improvements: Sequence[float] | np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
) -> float:
    """非循环移动块 bootstrap；返回 ``P*(mean<=0)`` 的加一一侧 p 值。"""

    values = np.asarray(improvements, dtype=np.float64)
    if values.ndim != 1 or len(values) < block_length or replicates < 1:
        raise ValueError("bootstrap 参数非法")
    starts_count = len(values) - block_length + 1
    blocks_needed = math.ceil(len(values) / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    nonpositive = 0
    remaining = replicates
    while remaining:
        batch = min(2000, remaining)
        starts = rng.integers(
            0, starts_count, size=(batch, blocks_needed), endpoint=False
        )
        indices = (starts[..., None] + offsets).reshape(batch, -1)[:, : len(values)]
        means = values[indices].mean(axis=1)
        nonpositive += int(np.count_nonzero(means <= 0.0))
        remaining -= batch
    return (nonpositive + 1.0) / (replicates + 1.0)


def _derived_seed(*parts: str) -> int:
    digest = hashlib.sha256("".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest, "big", signed=False)


def deterministic_random_controls(
    protocol_hash: str, target_issue: str
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """精确生成 r=0..511 的独立同成本 7+2 评估控制。"""

    controls = []
    for r in range(RANDOM_CONTROL_COUNT):
        seed = _derived_seed(protocol_hash, str(target_issue), str(r))
        rng = np.random.Generator(np.random.PCG64(seed))
        front = tuple(
            sorted(int(value) for value in rng.choice(35, 7, replace=False) + 1)
        )
        back = tuple(
            sorted(int(value) for value in rng.choice(12, 2, replace=False) + 1)
        )
        controls.append((front, back))
    return tuple(controls)


def builder_metrics(
    selected_front: Sequence[int],
    selected_back: Sequence[int],
    observed_front: Sequence[int],
    observed_back: Sequence[int],
) -> dict[str, object]:
    """计算固定 Top7/Top2、21 注命中矩阵及所有预注册描述指标。"""

    front = tuple(sorted(selected_front))
    back = tuple(sorted(selected_back))
    observed_front_set = set(observed_front)
    observed_back_set = set(observed_back)
    if len(front) != 7 or len(set(front)) != 7 or len(back) != 2 or len(set(back)) != 2:
        raise ValueError("构造必须恰为唯一 7+2")
    tickets = tuple(combinations(front, 5))
    if len(tickets) != 21 or len(set(tickets)) != 21:
        raise AssertionError("7+2 必须完整展开为 21 张唯一基本票")
    back_hits = len(set(back) & observed_back_set)
    matrix = [[0] * 3 for _ in range(6)]
    best_front = 0
    for ticket in tickets:
        hits = len(set(ticket) & observed_front_set)
        matrix[hits][back_hits] += 1
        best_front = max(best_front, hits)
    hf = len(set(front) & observed_front_set)
    threshold_indicators = {
        f"atLeast{front_hits}Plus{back_required}": bool(
            any(
                matrix[a][b]
                for a in range(front_hits, 6)
                for b in range(back_required, 3)
            )
        )
        for front_hits, back_required in THRESHOLDS
    }
    return {
        "HF": hf,
        "HB": back_hits,
        "U": hf / 5.0 + back_hits / 2.0,
        "ticketHitMatrix": matrix,
        "bestTicket": {"frontHits": best_front, "backHits": back_hits},
        "thresholdIndicators": threshold_indicators,
        "frontFiveFullyCovered": hf == 5,
        "backTwoFullyHit": back_hits == 2,
        "exact5Plus2Exists": best_front == 5 and back_hits == 2,
        "nominalTicketCount": 21,
        "uniqueTicketCount": 21,
    }


def _control_utilities(
    controls: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    observed_front: Sequence[int],
    observed_back: Sequence[int],
) -> np.ndarray:
    front = np.asarray([item[0] for item in controls], dtype=np.int16)
    back = np.asarray([item[1] for item in controls], dtype=np.int16)
    front_hits = np.isin(front, np.asarray(observed_front)).sum(axis=1)
    back_hits = np.isin(back, np.asarray(observed_back)).sum(axis=1)
    return front_hits / 5.0 + back_hits / 2.0


def evaluate_gates(metrics: Mapping[str, object]) -> dict[str, bool]:
    """应用 Validation 冻结闸门；只有全部为真才 eligible。"""

    joint_blocks = [float(value) for value in metrics["jointBlockImprovements"]]  # type: ignore[index]
    u_blocks = [float(value) for value in metrics["uBlockImprovementsVsRandom512"]]  # type: ignore[index]
    if len(joint_blocks) != 5 or len(u_blocks) != 5:
        raise ValueError("Validation 必须恰含 5 个连续 100 期块")
    return {
        "jointMeanPositive": float(metrics["meanJointLogLossImprovement"]) > 0.0,
        "holmAdjustedBootstrapPAtMost005": float(metrics["holmAdjustedP"]) <= 0.05,
        "frontSetLogLossAtMostUniform": float(metrics["meanFrontSetLogLoss"])
        <= UNIFORM_FRONT_SET_LL,
        "backSetLogLossAtMostUniform": float(metrics["meanBackSetLogLoss"])
        <= UNIFORM_BACK_SET_LL,
        "frontBrierAtMostUniform": float(metrics["meanFrontMarginalBrier"])
        <= UNIFORM_FRONT_BRIER,
        "backBrierAtMostUniform": float(metrics["meanBackMarginalBrier"])
        <= UNIFORM_BACK_BRIER,
        "jointFourOfFiveNonnegative": sum(value >= 0.0 for value in joint_blocks) >= 4,
        "jointWorstBlockFloor": min(joint_blocks) >= -0.05,
        "uMeanAboveRandom512": float(metrics["meanUImprovementVsRandom512"]) > 0.0,
        "uFourOfFiveNonnegative": sum(value >= 0.0 for value in u_blocks) >= 4,
    }


def select_candidate(candidates: Sequence[Mapping[str, object]]) -> str | None:
    """按 LL、联合 Brier、U、ID 的冻结顺序唯一选择。"""

    eligible = [candidate for candidate in candidates if candidate["eligible"] is True]
    if not eligible:
        return None

    def key(candidate: Mapping[str, object]) -> tuple[float, float, float, str]:
        metrics = candidate["metrics"]
        assert isinstance(metrics, Mapping)
        return (
            -float(metrics["meanJointLogLossImprovement"]),
            -float(metrics["combinedBrierImprovement"]),
            -float(metrics["meanUImprovementVsRandom512"]),
            str(candidate["candidateId"]),
        )

    return str(min(eligible, key=key)["candidateId"])


def _default_score_provider(
    cache: FeatureSnapshotCache, cutoff: int
) -> dict[str, CandidateScores]:
    components = {
        candidate_id: fit_candidate_scores_from_cache(cache, cutoff, candidate_id)
        for candidate_id in CANDIDATE_IDS[:3]
    }
    components[CANDIDATE_IDS[3]] = CandidateScores(
        tuple(
            math.fsum(components[item].front[index] for item in CANDIDATE_IDS[:3]) / 3.0
            for index in range(FRONT_SIZE)
        ),
        tuple(
            math.fsum(components[item].back[index] for item in CANDIDATE_IDS[:3]) / 3.0
            for index in range(BACK_SIZE)
        ),
    )
    return components


def _validate_search_report(
    report: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    if (
        report.get("schemaVersion") != 1
        or report.get("study") != "dlt_7plus2_search_v1"
    ):
        raise ValueError("Search 报告身份不匹配")
    partitions = report.get("partitions")
    if (
        not isinstance(partitions, Mapping)
        or partitions.get("searchCount") != 1301
        or partitions.get("maximumConsumedIndex") != 1900
    ):
        raise ValueError("Search 分区绑定不匹配")
    raw_candidates = report.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        raise ValueError("Search 必须恰含四候选")
    calibrations: dict[str, dict[str, float]] = {}
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise TypeError("Search 候选格式非法")
        candidate_id = str(raw.get("candidateId"))
        selected = raw.get("selectedCalibration")
        if (
            candidate_id in calibrations
            or candidate_id not in CANDIDATE_IDS
            or not isinstance(selected, Mapping)
        ):
            raise ValueError("Search 候选 ID/校准非法")
        zone_result: dict[str, float] = {}
        for zone in ("front", "back"):
            parameters = selected.get(zone)
            if not isinstance(parameters, Mapping):
                raise TypeError("Search 固定校准参数缺失")
            tau, epsilon = float(parameters["tau"]), float(parameters["epsilon"])
            if tau not in (0.5, 0.75, 1.0, 1.5, 2.0) or epsilon not in (
                0.0,
                0.05,
                0.1,
                0.2,
            ):
                raise ValueError("Search 固定校准参数超出协议网格")
            zone_result[f"{zone}Tau"] = tau
            zone_result[f"{zone}Epsilon"] = epsilon
        calibrations[candidate_id] = zone_result
    if tuple(calibrations) != CANDIDATE_IDS:
        raise ValueError("Search 候选顺序/集合不匹配")
    return calibrations


def _brier(marginals: Sequence[float], observed: Sequence[int]) -> float:
    observed_zero = {value - 1 for value in observed}
    return math.fsum(
        (probability - float(index in observed_zero)) ** 2
        for index, probability in enumerate(marginals)
    ) / len(marginals)


def _fingerprint_floats(values: Sequence[float]) -> str:
    return _sha256(b"".join(struct.pack(">d", float(value)) for value in values))


def run_validation(
    draws: Sequence[DLTDrawLike],
    *,
    search_report: Mapping[str, object],
    search_report_bytes: bytes,
    score_provider: (
        Callable[[FeatureSnapshotCache, int], Mapping[str, CandidateScores]] | None
    ) = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    data_source_sha256: str | None = None,
    protocol_document_sha256: str | None = None,
) -> dict[str, object]:
    """只消费 ``draws[0]`` 至 ``draws[2400]``，执行完整 Validation。"""

    calibrations = _validate_search_report(search_report)
    cache = FeatureSnapshotCache(draws, stop_index=VALIDATION_STOP)
    provider = score_provider or _default_score_provider
    score_by_cutoff = {
        cutoff: dict(provider(cache, cutoff)) for cutoff in global_validation_cutoffs()
    }
    for cutoff, cutoff_scores in score_by_cutoff.items():
        if tuple(cutoff_scores) != CANDIDATE_IDS:
            raise ValueError(f"截止点 {cutoff} 未精确返回四候选")

    candidate_rows: dict[str, list[dict[str, object]]] = {
        candidate_id: [] for candidate_id in CANDIDATE_IDS
    }
    prefix_material: list[list[object]] = []
    for target in range(VALIDATION_START, VALIDATION_STOP):
        # Cache construction already consumed this row; this explicit access binds the observed audit.
        draw = draws[target]
        issue = str(getattr(draw, "issue", target))
        observed_front = tuple(int(value) for value in draw.front)
        observed_back = tuple(int(value) for value in draw.back)
        prefix_material.append([issue, list(observed_front), list(observed_back)])
        cutoff = target - target % BLOCK_SIZE
        random_controls = deterministic_random_controls(FROZEN_PROTOCOL_SHA256, issue)
        control_u = _control_utilities(random_controls, observed_front, observed_back)
        random_mean_u = float(control_u.mean())
        r0_u = float(control_u[0])
        controls_hash = _sha256(_canonical_bytes(random_controls))
        for candidate_id in CANDIDATE_IDS:
            scores = score_by_cutoff[cutoff][candidate_id]
            calibration = calibrations[candidate_id]
            front_distribution = FixedCardinalityDistribution(
                scores.front,
                5,
                tau=calibration["frontTau"],
                epsilon=calibration["frontEpsilon"],
            )
            back_distribution = FixedCardinalityDistribution(
                scores.back,
                2,
                tau=calibration["backTau"],
                epsilon=calibration["backEpsilon"],
            )
            front_ll = -front_distribution.log_probability(
                value - 1 for value in observed_front
            )
            back_ll = -back_distribution.log_probability(
                value - 1 for value in observed_back
            )
            selected_front = tuple(
                sorted(
                    sorted(
                        range(1, 36),
                        key=lambda value: (
                            -front_distribution.marginals[value - 1],
                            value,
                        ),
                    )[:7]
                )
            )
            selected_back = tuple(
                sorted(
                    sorted(
                        range(1, 13),
                        key=lambda value: (
                            -back_distribution.marginals[value - 1],
                            value,
                        ),
                    )[:2]
                )
            )
            builder = builder_metrics(
                selected_front, selected_back, observed_front, observed_back
            )
            distribution_audit = {
                "family": "exact_fixed_cardinality_additive_set_distribution",
                "front": {
                    "N": 35,
                    "K": 5,
                    "tau": calibration["frontTau"],
                    "epsilon": calibration["frontEpsilon"],
                    "scores": list(scores.front),
                    "marginals": list(front_distribution.marginals),
                    "logPartition": front_distribution.log_partition,
                    "observedSetProbability": math.exp(-front_ll),
                    "scoreSha256": _fingerprint_floats(scores.front),
                    "marginalsSha256": _fingerprint_floats(
                        front_distribution.marginals
                    ),
                },
                "back": {
                    "N": 12,
                    "K": 2,
                    "tau": calibration["backTau"],
                    "epsilon": calibration["backEpsilon"],
                    "scores": list(scores.back),
                    "marginals": list(back_distribution.marginals),
                    "logPartition": back_distribution.log_partition,
                    "observedSetProbability": math.exp(-back_ll),
                    "scoreSha256": _fingerprint_floats(scores.back),
                    "marginalsSha256": _fingerprint_floats(back_distribution.marginals),
                },
            }
            issue_audit = {
                "targetIndex": target,
                "targetIssue": issue,
                "fitCutoffExclusive": cutoff,
                "observed": {
                    "front": list(observed_front),
                    "back": list(observed_back),
                },
                "distribution": distribution_audit,
                "properScores": {
                    "frontSetLogLoss": front_ll,
                    "backSetLogLoss": back_ll,
                    "jointLogLoss": front_ll + back_ll,
                    "jointLogLossImprovementVsExactUniform": UNIFORM_JOINT_LL
                    - front_ll
                    - back_ll,
                    "frontMarginalBrier": _brier(
                        front_distribution.marginals, observed_front
                    ),
                    "backMarginalBrier": _brier(
                        back_distribution.marginals, observed_back
                    ),
                },
                "builder21": {
                    "selectedFront7": list(selected_front),
                    "selectedBack2": list(selected_back),
                    **builder,
                },
                "randomControlsEvaluationOnly": {
                    "count": 512,
                    "sameNominalTicketsEach": 21,
                    "sameCostYuanEach": 42,
                    "controlsSha256": controls_hash,
                    "meanU": random_mean_u,
                    "r0": {
                        "front7": list(random_controls[0][0]),
                        "back2": list(random_controls[0][1]),
                        "U": r0_u,
                    },
                },
            }
            issue_audit["issueFingerprintSha256"] = _sha256(
                _canonical_bytes(issue_audit)
            )
            candidate_rows[candidate_id].append(issue_audit)

    candidates: list[dict[str, object]] = []
    raw_p_values = []
    for candidate_id in CANDIDATE_IDS:
        rows = candidate_rows[candidate_id]
        proper = [row["properScores"] for row in rows]
        builders = [row["builder21"] for row in rows]
        control_rows = [row["randomControlsEvaluationOnly"] for row in rows]
        improvements = np.asarray(
            [float(item["jointLogLossImprovementVsExactUniform"]) for item in proper]
        )
        u_values = np.asarray([float(item["U"]) for item in builders])
        random_u = np.asarray([float(item["meanU"]) for item in control_rows])
        u_differences = u_values - random_u
        seed = _derived_seed(
            FROZEN_PROTOCOL_SHA256, "validation-bootstrap", candidate_id
        )
        raw_p = moving_block_bootstrap_p(
            improvements,
            seed=seed,
            replicates=bootstrap_replicates,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
        )
        raw_p_values.append(raw_p)
        joint_blocks = [
            _mean(improvements[start : start + 100]) for start in range(0, 500, 100)
        ]
        u_blocks = [
            _mean(u_differences[start : start + 100]) for start in range(0, 500, 100)
        ]
        hf_distribution = Counter(str(int(item["HF"])) for item in builders)
        hb_distribution = Counter(str(int(item["HB"])) for item in builders)
        aggregate_matrix = np.sum(
            np.asarray([item["ticketHitMatrix"] for item in builders], dtype=np.int64),
            axis=0,
        )
        metrics = {
            "meanFrontSetLogLoss": _mean(
                [float(item["frontSetLogLoss"]) for item in proper]
            ),
            "meanBackSetLogLoss": _mean(
                [float(item["backSetLogLoss"]) for item in proper]
            ),
            "meanJointLogLoss": _mean([float(item["jointLogLoss"]) for item in proper]),
            "meanJointLogLossImprovement": _mean(improvements),
            "meanFrontMarginalBrier": _mean(
                [float(item["frontMarginalBrier"]) for item in proper]
            ),
            "meanBackMarginalBrier": _mean(
                [float(item["backMarginalBrier"]) for item in proper]
            ),
            "combinedBrierImprovement": UNIFORM_FRONT_BRIER
            + UNIFORM_BACK_BRIER
            - _mean([float(item["frontMarginalBrier"]) for item in proper])
            - _mean([float(item["backMarginalBrier"]) for item in proper]),
            "bootstrapRawOneSidedP": raw_p,
            "bootstrapSeedSha256Derivation": _sha256(
                (
                    FROZEN_PROTOCOL_SHA256 + "validation-bootstrap" + candidate_id
                ).encode()
            ),
            "jointBlockImprovements": joint_blocks,
            "meanU": _mean(u_values),
            "meanRandom512U": _mean(random_u),
            "meanUImprovementVsRandom512": _mean(u_differences),
            "uBlockImprovementsVsRandom512": u_blocks,
        }
        candidates.append(
            {
                "candidateId": candidate_id,
                "fixedSearchCalibration": calibrations[candidate_id],
                "metrics": metrics,
                "coverageDescriptions": {
                    "HFDefinition": "size(selectedFront7 intersect observedFront5)",
                    "HBDefinition": "size(selectedBack2 intersect observedBack2)",
                    "UDefinition": "HF/5 + HB/2",
                    "HFDistribution": dict(sorted(hf_distribution.items())),
                    "HBDistribution": dict(sorted(hb_distribution.items())),
                    "aggregateTicketHitMatrix": aggregate_matrix.tolist(),
                    "ticketHitMatrixAxes": {
                        "rows": "front hits 0..5",
                        "columns": "back hits 0..2",
                    },
                    "thresholdSemantics": "at least one of 21 tickets has front hits >=a and back hits >=b",
                    "thresholds": [f"{a}+{b}" for a, b in THRESHOLDS],
                },
                "perIssue": rows,
                "candidateAuditSha256": _sha256(_canonical_bytes(rows)),
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for candidate, adjusted_p in zip(candidates, adjusted, strict=True):
        candidate_metrics = candidate["metrics"]
        assert isinstance(candidate_metrics, dict)
        candidate_metrics["holmAdjustedP"] = adjusted_p
        gates = evaluate_gates(candidate_metrics)
        candidate["gates"] = gates
        candidate["eligible"] = all(gates.values())

    selected = select_candidate(candidates)
    source_path = Path(__file__)
    source_hash = _sha256(source_path.read_bytes())
    search_hash = _sha256(search_report_bytes)
    prefix_hash = _sha256(_canonical_bytes(prefix_material))
    return {
        "schemaVersion": 1,
        "study": "dlt_7plus2_validation_v1",
        "selectionStatus": "selected" if selected is not None else "rejected",
        "selectedCandidateId": selected,
        "currentTargetGroup": None,
        "formalOutput": "uniform_abstain",
        "formalGate": False,
        "autoPromotion": False,
        "noAutoPromotion": True,
        "frozenOpened": False,
        "frozenRowsAccessed": 0,
        "frozenEvaluationImplemented": False,
        "partitions": {
            "validationIndices": [1901, 2401],
            "validationCount": 500,
            "maximumConsumedIndex": 2400,
            "global25Cutoffs": list(global_validation_cutoffs()),
        },
        "protocol": {
            "protocolSha256": FROZEN_PROTOCOL_SHA256,
            "protocolDocumentSha256": protocol_document_sha256,
            "bootstrap": {
                "method": "noncircular_moving_block_resample_then_trim",
                "blockLength": 20,
                "replicates": bootstrap_replicates,
                "oneSidedP": "(1 + count(bootstrap mean <= 0))/(B+1)",
                "multipleTesting": "Holm across exactly four candidates",
            },
            "randomControls": {
                "countPerIssue": 512,
                "rRange": [0, 511],
                "seedBytes": "SHA256(protocolHash||targetIssue||decimal(r))",
                "generator": "numpy.random.PCG64(full_unsigned_SHA256_integer)",
                "purpose": "evaluation_only_not_purchased_or_optimized",
                "r0ReportedSeparately": True,
            },
            "selectionOrder": [
                "mean joint LL improvement descending",
                "combined Brier improvement descending",
                "U improvement descending",
                "candidate ID ascending",
            ],
        },
        "uniformBaselines": {
            "frontSetLogLoss": UNIFORM_FRONT_SET_LL,
            "backSetLogLoss": UNIFORM_BACK_SET_LL,
            "jointSetLogLoss": UNIFORM_JOINT_LL,
            "frontMarginalBrier": UNIFORM_FRONT_BRIER,
            "backMarginalBrier": UNIFORM_BACK_BRIER,
        },
        "bindings": {
            "searchReportSha256": search_hash,
            "searchObservedSha256": search_report.get("searchObservedSha256"),
            "validationObservedPrefixSha256": prefix_hash,
            "dataPrefixThrough2400Sha256": data_source_sha256 or prefix_hash,
            "protocolSha256": FROZEN_PROTOCOL_SHA256,
            "protocolDocumentSha256": protocol_document_sha256,
            "validationCodeSha256": source_hash,
        },
        "candidates": candidates,
        "disclaimer": "research only; no prediction claim; no betting recommendation",
    }


def read_validation_prefix_csv(path: str | Path) -> tuple[ValidationDraw, ...]:
    """精确读取前 2401 行，不调用 len、不探测索引 2401。"""

    result = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for _ in range(VALIDATION_STOP):
            try:
                row = next(reader)
            except StopIteration as error:
                raise ValueError("DLT CSV 不足 2401 行") from error
            result.append(
                ValidationDraw(
                    row["issue"],
                    tuple(int(value) for value in row["front"].split()),
                    tuple(int(value) for value in row["back"].split()),
                )
            )
    return tuple(result)


def write_validation_report(
    report: Mapping[str, object], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/dlt/official_history.csv")
    parser.add_argument(
        "--search-report", default="reports/research/dlt_7plus2_search_v1.json"
    )
    parser.add_argument("--protocol-document", default="docs/dlt_7plus2_v1_protocol.md")
    parser.add_argument(
        "--output-report", default="reports/retrospective/dlt_7plus2_validation_v1.json"
    )
    args = parser.parse_args(argv)
    try:
        search_bytes = Path(args.search_report).read_bytes()
        if _sha256(search_bytes) != EXPECTED_SEARCH_REPORT_SHA256:
            raise ValueError("Search 报告字节哈希不等于冻结工件")
        protocol_bytes = Path(args.protocol_document).read_bytes()
        search_report = json.loads(search_bytes)
        draws = read_validation_prefix_csv(args.input_csv)
        report = run_validation(
            draws,
            search_report=search_report,
            search_report_bytes=search_bytes,
            # 不读取完整 CSV 字节；run_validation 绑定实际消费到 2400 的规范前缀。
            data_source_sha256=None,
            protocol_document_sha256=_sha256(protocol_bytes),
        )
        write_validation_report(report, args.output_report)
    except (
        OSError,
        ValueError,
        ArithmeticError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"DLT Validation 失败：{error}", file=sys.stderr)
        return 1
    print(
        f"DLT Validation 报告已写入：{args.output_report}；状态={report['selectionStatus']}；"
        f"候选={report['selectedCandidateId']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
