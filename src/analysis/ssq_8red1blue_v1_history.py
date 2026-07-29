# -*- coding: utf-8 -*-
"""双色球8红+1蓝研究影子 v1 的全历史严格前序诊断。"""

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

from scipy import stats

from src.analysis.ssq_d8_b35_support import build_diversified_portfolio_v2
from src.analysis.ssq_ensemble_v1 import EVALUATION_WARMUP_DRAWS, FixedEnsembleState
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    protocol_sha256 as builder_protocol_sha256,
)
from src.lotteries.ssq import SSQ_RULE

SCHEMA_VERSION = "ssq_8red1blue_v1_full_history"
CONTROL_COUNT = 32
CONTROL_SEED_PROTOCOL = (
    "ssq_8red1blue_v1_full_history|matched_random_8red1blue|"
    "sha256_counter_rejection_v1"
)
THRESHOLDS = (3, 4, 5, 6)
METRIC_NAMES = (
    "red8Overlap",
    "averageRedHitsPerTicket",
    "maximumRedHitsAnyTicket",
    *(f"anyTicketRedAtLeast{threshold}" for threshold in THRESHOLDS),
    "blueHit",
    "exact6PlusBlue",
    "exact6PlusNoBlue",
    "overlapWithB",
    "combinedNominalTicketCount",
    "combinedUniqueTicketCount",
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
    "purpose": "strict_predict_before_update_retrospective_diagnostic_only",
    "walkForward": {
        "warmupDraws": EVALUATION_WARMUP_DRAWS,
        "evaluatedHistory": "all periods after fixed warmup",
        "sequence": "predict_build_score_then_update",
        "futureLeakage": False,
    },
    "D8": "fixed smallCompound8Red1BlueV1 with zero full-ticket overlap against B35",
    "controls": {
        "countPerIssue": CONTROL_COUNT,
        "construction": "deterministic matched random legal 8-red plus one blue",
        "seedProtocol": CONTROL_SEED_PROTOCOL,
        "resultIndependent": True,
        "modelIndependent": True,
    },
    "metrics": {
        "perIssue": list(METRIC_NAMES),
        "redOverlapDistribution": "actual six reds intersect selected red8, values 0..6",
        "officialPrizeClaims": False,
    },
    "statistics": {
        "comparison": "D8 minus issue mean of 32 controls",
        "method": "one-sided paired t-test alternative greater",
        "status": "posthoc descriptive unadjusted no formal gate",
    },
    "claims": {
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": "uniform_abstain",
        "officialPrizeClaims": False,
    },
}


def protocol_sha256() -> str:
    """返回固定历史诊断协议摘要。"""

    return _sha256_payload(PROTOCOL)


class _Sha256CounterRng:
    """跨运行稳定、结果无关的SHA-256计数器随机源。"""

    def __init__(self, seed: bytes) -> None:
        self._seed = seed
        self._counter = 0

    def _uint256(self) -> int:
        digest = hashlib.sha256(self._seed + self._counter.to_bytes(16, "big")).digest()
        self._counter += 1
        return int.from_bytes(digest, "big")

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("随机上界必须为正数")
        space = 1 << 256
        limit = space - space % upper
        while True:
            value = self._uint256()
            if value < limit:
                return value % upper

    def shuffled(self, values: Sequence[int]) -> list[int]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            other = self.randbelow(index + 1)
            result[index], result[other] = result[other], result[index]
        return result


@dataclass(frozen=True)
class Compound8:
    """一个8红+1蓝及其28注展开票。"""

    red8: tuple[int, ...]
    blue: int
    tickets: tuple[tuple[tuple[int, ...], int], ...]


@dataclass
class _Accumulator:
    observations: int = 0
    totals: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in METRIC_NAMES}
    )
    red_overlap_distribution: list[int] = field(default_factory=lambda: [0] * 7)

    def add(self, metrics: Mapping[str, float]) -> None:
        self.observations += 1
        for name in METRIC_NAMES:
            self.totals[name] += metrics[name]
        self.red_overlap_distribution[int(metrics["red8Overlap"])] += 1

    def report(self) -> dict[str, object]:
        if not self.observations:
            raise ValueError("空诊断不能生成汇总")
        return {
            "observations": self.observations,
            "metricsPerIssue": {
                name: self.totals[name] / self.observations for name in METRIC_NAMES
            },
            "red8OverlapDistribution": {
                str(index): count
                for index, count in enumerate(self.red_overlap_distribution)
            },
        }


def _compound(red8: Sequence[int], blue: int) -> Compound8:
    normalized_red = tuple(sorted(red8))
    if len(normalized_red) != 8 or len(set(normalized_red)) != 8:
        raise ValueError("8红对照必须恰好包含8个唯一红球")
    tickets = tuple((tuple(red6), blue) for red6 in combinations(normalized_red, 6))
    if len(tickets) != 28 or len(set(tickets)) != 28:
        raise ValueError("8红+1蓝必须展开为28注唯一票")
    for red6, ticket_blue in tickets:
        SSQ_RULE.validate_draw(red6, ticket_blue)
    return Compound8(normalized_red, blue, tickets)


def build_matched_random_control(issue: str, control_index: int) -> Compound8:
    """仅由期号和固定编号生成确定性同成本随机8+1对照。"""

    if not issue.isdigit() or not 0 <= control_index < CONTROL_COUNT:
        raise ValueError("随机对照期号或编号非法")
    seed = hashlib.sha256(
        f"{CONTROL_SEED_PROTOCOL}|issue={issue}|control={control_index}".encode()
    ).digest()
    rng = _Sha256CounterRng(seed)
    red8 = tuple(sorted(rng.shuffled(tuple(range(1, 34)))[:8]))
    blue = rng.shuffled(tuple(range(1, 17)))[0]
    return _compound(red8, blue)


def _b_tickets(document: Mapping[str, object]) -> set[tuple[tuple[int, ...], int]]:
    groups = cast(list[dict[str, object]], document["groups"])
    return {
        (tuple(cast(list[int], ticket["red"])), cast(int, ticket["blue"]))
        for group in groups
        for ticket in cast(list[dict[str, object]], group["expandedTickets"])
    }


def _d8_from_document(document: Mapping[str, object]) -> Compound8:
    return _compound(cast(list[int], document["red"]), cast(int, document["blue"]))


def _score(
    compound: Compound8,
    draw: SSQDraw,
    b_tickets: set[tuple[tuple[int, ...], int]],
) -> dict[str, float]:
    actual_red = set(draw.red)
    red_hits = tuple(
        len(set(red6).intersection(actual_red)) for red6, _ in compound.tickets
    )
    maximum = max(red_hits)
    blue_hit = compound.blue == draw.blue
    overlap = len(set(compound.tickets).intersection(b_tickets))
    metrics: dict[str, float] = {
        "red8Overlap": float(len(set(compound.red8).intersection(actual_red))),
        "averageRedHitsPerTicket": sum(red_hits) / 28.0,
        "maximumRedHitsAnyTicket": float(maximum),
        "blueHit": float(blue_hit),
        "exact6PlusBlue": float(maximum == 6 and blue_hit),
        "exact6PlusNoBlue": float(maximum == 6 and not blue_hit),
        "overlapWithB": float(overlap),
        "combinedNominalTicketCount": 63.0,
        "combinedUniqueTicketCount": float(63 - overlap),
    }
    for threshold in THRESHOLDS:
        metrics[f"anyTicketRedAtLeast{threshold}"] = float(maximum >= threshold)
    return metrics


def _issue_mean(scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(scores) != CONTROL_COUNT:
        raise ValueError("每期必须恰好包含32个随机对照")
    return {
        name: sum(score[name] for score in scores) / CONTROL_COUNT
        for name in METRIC_NAMES
    }


def _paired_test(differences: Sequence[float]) -> dict[str, object]:
    if not differences:
        raise ValueError("配对检验缺少观测")
    mean_difference = sum(differences) / len(differences)
    if all(value == differences[0] for value in differences):
        statistic: float | None = 0.0 if mean_difference == 0.0 else None
        p_value = 1.0 if mean_difference <= 0.0 else 0.0
    else:
        result = stats.ttest_1samp(differences, 0.0, alternative="greater")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(statistic) or not math.isfinite(p_value):
            statistic = None
            p_value = 1.0 if mean_difference <= 0.0 else 0.0
    return {
        "observations": len(differences),
        "meanDifference": mean_difference,
        "statistic": statistic,
        "oneSidedPValue": p_value,
        "alternative": "greater",
        "descriptiveOnly": True,
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


def walk_forward_d8_fingerprints(
    draws: Sequence[SSQDraw], stop: int | None = None
) -> list[str]:
    """返回逐期D8与B绑定摘要，供无未来泄漏回归测试。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    limit = len(ordered) if stop is None else min(stop, len(ordered))
    state = FixedEnsembleState()
    fingerprints: list[str] = []
    for draw in ordered[:limit]:
        red, blue, _ = state.predict()
        b_document = build_diversified_portfolio_v2(red, blue)
        d8_document = build_small_compound_8red1blue_v1(red, blue, b_document)
        fingerprints.append(_sha256_payload({"B": b_document, "D8": d8_document}))
        state.score_then_update(draw)
    return fingerprints


def evaluate_full_history(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """执行120期预热及其后全部历史的严格先预测后更新诊断。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) <= EVALUATION_WARMUP_DRAWS:
        raise ValueError("双色球历史不足120期预热加至少1期诊断")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("双色球历史包含重复期号")
    state = FixedEnsembleState()
    for draw in ordered[:EVALUATION_WARMUP_DRAWS]:
        state.score_then_update(draw)
    accumulators = {label: _Accumulator() for label in ("D8", "C32")}
    differences: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    per_issue: list[dict[str, object]] = []
    for draw in ordered[EVALUATION_WARMUP_DRAWS:]:
        red_probabilities, blue_probabilities, _ = state.predict()
        b_document = build_diversified_portfolio_v2(
            red_probabilities, blue_probabilities
        )
        b_tickets = _b_tickets(b_document)
        d8_document = build_small_compound_8red1blue_v1(
            red_probabilities, blue_probabilities, b_document
        )
        d8 = _d8_from_document(d8_document)
        d8_score = _score(d8, draw, b_tickets)
        if d8_score["overlapWithB"] != 0.0:
            raise ValueError("D8历史诊断发现与B重叠，失败关闭")
        control_scores = [
            _score(build_matched_random_control(draw.issue, index), draw, b_tickets)
            for index in range(CONTROL_COUNT)
        ]
        control_mean = _issue_mean(control_scores)
        accumulators["D8"].add(d8_score)
        for score in control_scores:
            accumulators["C32"].add(score)
        for name in METRIC_NAMES:
            differences[name].append(d8_score[name] - control_mean[name])
        audit = cast(Mapping[str, object], d8_document["audit"])
        per_issue.append(
            {
                "issue": draw.issue,
                "date": draw.draw_date,
                "D8": {
                    "red": d8_document["red"],
                    "blue": d8_document["blue"],
                    "selectedCandidateRank": d8_document["selectedCandidateRank"],
                    "expandedTicketSha256": _sha256_payload(
                        d8_document["expandedTickets"]
                    ),
                    "audit": {
                        "overlapWithB": audit["overlapWithB"],
                        "combinedNominalTicketCount": audit[
                            "combinedNominalTicketCount"
                        ],
                        "combinedUniqueTicketCount": audit["combinedUniqueTicketCount"],
                    },
                    "metrics": d8_score,
                },
                "C32Mean": control_mean,
                "BExpandedTicketSha256": _sha256_payload(sorted(b_tickets)),
            }
        )
        state.score_then_update(draw)
    evaluated = ordered[EVALUATION_WARMUP_DRAWS:]
    summaries = {
        label: accumulator.report() for label, accumulator in accumulators.items()
    }
    d8_metrics = cast(dict[str, float], summaries["D8"]["metricsPerIssue"])
    c_metrics = cast(dict[str, float], summaries["C32"]["metricsPerIssue"])
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedDeterministically": True,
        "researchOnly": True,
        "predictionClaim": False,
        "formalGate": False,
        "formalRecommendationStatus": "uniform_abstain",
        "officialPrizeClaims": False,
        "protocol": PROTOCOL,
        "protocolSha256": protocol_sha256(),
        "builderProtocolSha256": builder_protocol_sha256(),
        "dataSha256": _sha256_payload(_data_payload(ordered)),
        "history": {
            "totalPeriods": len(ordered),
            "warmupPeriods": EVALUATION_WARMUP_DRAWS,
            "evaluatedPeriods": len(evaluated),
            "firstEvaluatedIssue": evaluated[0].issue,
            "firstEvaluatedDate": evaluated[0].draw_date,
            "lastEvaluatedIssue": evaluated[-1].issue,
            "lastEvaluatedDate": evaluated[-1].draw_date,
        },
        "metricDefinitions": {
            "red8Overlap": "当期6个开奖号码与D8红球集合的交集数",
            "thresholds": "atLeast3/4/5/6按28注中任一票最大红球命中数判断",
            "blueHit": "共享蓝球是否命中",
            "exact6PlusBlue": "存在红6全中且共享蓝球命中",
            "exact6PlusNoBlue": "存在红6全中且共享蓝球未命中",
            "overlapWithB": "D8的28注与同期既有B35完整红6+蓝票交集数",
            "combinedCounts": "D8+B的名义与唯一完整票数",
            "officialPrizeClaims": False,
        },
        "summary": summaries,
        "D8MinusC32Mean": {
            name: d8_metrics[name] - c_metrics[name] for name in METRIC_NAMES
        },
        "descriptivePairedTests": {
            "method": "one-sided paired t-test; alternative D8 greater",
            "multipleComparisonStatus": "posthoc descriptive unadjusted; no gate",
            "D8MinusC32Mean": {
                name: _paired_test(differences[name]) for name in METRIC_NAMES
            },
        },
        "perIssue": per_issue,
    }
    report["reportSha256"] = _sha256_payload(report)
    return report


def write_report(report: Mapping[str, object], output_path: str | Path) -> Path:
    """原子写入历史诊断JSON。"""

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
    "CONTROL_COUNT",
    "METRIC_NAMES",
    "build_matched_random_control",
    "evaluate_full_history",
    "protocol_sha256",
    "walk_forward_d8_fingerprints",
    "write_report",
]
