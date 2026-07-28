# -*- coding: utf-8 -*-
"""双色球分散组合 v2 的全历史严格前序 A/B/C 回溯评估。"""

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

from src.analysis.ssq_diversified_portfolio_v2 import (
    EXPANDED_TICKET_COUNT,
    build_diversified_portfolio_v2,
)
from src.analysis.ssq_diversified_portfolio_v2 import (
    protocol_sha256 as builder_protocol_sha256,
)
from src.analysis.ssq_diversified_portfolio_v2 import (
    validate_diversified_portfolio_v2,
)
from src.analysis.ssq_ensemble_v1 import (
    BEAM_WIDTH,
    EVALUATION_WARMUP_DRAWS,
    FixedEnsembleState,
    beam_red_combinations,
    blue_top1,
)
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_reference_helpers import _small_compound_7_red_1_blue_top5
from src.lotteries.ssq import SSQ_RULE

SCHEMA_VERSION = "ssq_diversified_portfolio_v2_full_history"
CONTROL_COUNT = 32
GROUP_COUNT = 5
RED_PER_GROUP = 7
CONTROL_SEED_PROTOCOL = (
    "ssq_diversified_portfolio_v2_history|matched_cost_random_c|"
    "sha256_counter_rejection_v1"
)
THRESHOLDS = (3, 4, 5, 6)
METRIC_NAMES = (
    "averageRedHitsPerTicket",
    "maximumRedHitsAnyTicket",
    *(f"anyTicketRedAtLeast{threshold}" for threshold in THRESHOLDS),
    "blueAnyHit",
    *(f"ticketRedAtLeast{threshold}PlusBlue" for threshold in THRESHOLDS),
    *(f"ticketRedAtLeast{threshold}PlusNoBlue" for threshold in THRESHOLDS),
)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "retrospective_research_only_no_formal_gate",
    "walkForward": {
        "warmupDraws": EVALUATION_WARMUP_DRAWS,
        "sequence": "predict_before_score_then_update",
        "evaluatedHistory": "all periods after fixed warmup",
        "futureLeakage": False,
    },
    "portfolios": {
        "A": "existing concentrated Top5 exact construction with shared blue Top1",
        "B": "fixed coverage-first diversifiedPortfolioV2",
        "C": {
            "controlsPerIssue": CONTROL_COUNT,
            "construction": (
                "five random legal distinct red7 groups, five random distinct blues, "
                "one blue per group, exactly 35 unique expanded tickets"
            ),
            "seedProtocol": CONTROL_SEED_PROTOCOL,
            "resultIndependent": True,
            "modelIndependent": True,
        },
    },
    "metrics": {
        "perIssue": list(METRIC_NAMES),
        "blueLinkage": "ticket-specific assigned blue",
        "officialPrizeClaims": False,
    },
    "statistics": {
        "comparisons": ["B-A", "B-C_issue_mean"],
        "method": "one-sided paired t-test, alternative greater",
        "multipleComparison": "posthoc descriptive unadjusted family",
        "formalGate": False,
    },
    "fixed": {"cliTuning": False, "controlCount": CONTROL_COUNT},
}


def protocol_sha256() -> str:
    """返回历史评估固定协议摘要。"""

    return _sha256_payload(PROTOCOL)


class _Sha256CounterRng:
    """结果无关且跨运行稳定的 SHA-256 计数器随机源。"""

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
        shuffled = list(values)
        for index in range(len(shuffled) - 1, 0, -1):
            other = self.randbelow(index + 1)
            shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
        return shuffled


@dataclass(frozen=True)
class Portfolio:
    """五组红7与逐组蓝球的统一组合表示。"""

    red7_groups: tuple[tuple[int, ...], ...]
    blues: tuple[int, ...]
    tickets: tuple[tuple[tuple[int, ...], int], ...]


@dataclass(frozen=True)
class PortfolioScore:
    """一期组合的逐票评分结果。"""

    ticket_red_hits: tuple[int, ...]
    ticket_blue_hits: tuple[bool, ...]
    metrics: Mapping[str, float]


@dataclass
class _Accumulator:
    observations: int = 0
    totals: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in METRIC_NAMES}
    )
    max_red_distribution: list[int] = field(default_factory=lambda: [0] * 7)

    def add(self, score: PortfolioScore) -> None:
        self.observations += 1
        for name in METRIC_NAMES:
            self.totals[name] += score.metrics[name]
        maximum = int(score.metrics["maximumRedHitsAnyTicket"])
        self.max_red_distribution[maximum] += 1

    def report(self) -> dict[str, object]:
        if self.observations <= 0:
            raise ValueError("空评估不能生成汇总")
        return {
            "portfolioObservations": self.observations,
            "metricsPerIssue": {
                name: self.totals[name] / self.observations for name in METRIC_NAMES
            },
            "maximumRedHitsAnyTicketDistribution": {
                str(index): count
                for index, count in enumerate(self.max_red_distribution)
            },
        }


def _expand(
    red7_groups: Sequence[Sequence[int]], blues: Sequence[int]
) -> tuple[tuple[tuple[int, ...], int], ...]:
    tickets = tuple(
        (tuple(red6), blue)
        for red7, blue in zip(red7_groups, blues)
        for red6 in combinations(red7, 6)
    )
    if (
        len(tickets) != EXPANDED_TICKET_COUNT
        or len(set(tickets)) != EXPANDED_TICKET_COUNT
    ):
        raise ValueError("组合必须形成35注全局唯一票")
    return tickets


def _validate_portfolio(portfolio: Portfolio) -> None:
    if (
        len(portfolio.red7_groups) != GROUP_COUNT
        or len(set(portfolio.red7_groups)) != GROUP_COUNT
    ):
        raise ValueError("组合必须包含5组互异红7")
    if len(portfolio.blues) != GROUP_COUNT or len(set(portfolio.blues)) != GROUP_COUNT:
        raise ValueError("组合必须包含5个互异蓝球")
    for red7, blue in zip(portfolio.red7_groups, portfolio.blues):
        if (
            len(red7) != RED_PER_GROUP
            or tuple(sorted(red7)) != red7
            or len(set(red7)) != RED_PER_GROUP
        ):
            raise ValueError("红7组必须合法、升序且互异")
        if not 1 <= blue <= 16:
            raise ValueError("蓝球越界")
        for red6 in combinations(red7, 6):
            SSQ_RULE.validate_draw(red6, blue)
    if portfolio.tickets != _expand(portfolio.red7_groups, portfolio.blues):
        raise ValueError("组合展开票与红7/蓝球分配不一致")


def _portfolio_a(
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    pair_modifiers: Sequence[float],
) -> Portfolio:
    ranked = beam_red_combinations(red_probabilities, pair_modifiers)
    ranked_entries = [
        {
            "rank": index,
            "red": [f"{ball:02d}" for ball in red],
            "redScore": score,
        }
        for index, (red, score) in enumerate(ranked, start=1)
    ]
    construction = _small_compound_7_red_1_blue_top5(
        ranked_entries, f"{blue_top1(blue_probabilities):02d}"
    )
    groups_raw = cast(list[dict[str, object]], construction["compounds"])
    red7_groups = tuple(
        tuple(int(value) for value in cast(list[str], group["red"]))
        for group in groups_raw
    )
    shared_blue = blue_top1(blue_probabilities)
    blues = (shared_blue,) * GROUP_COUNT
    portfolio = Portfolio(red7_groups, blues, _expand(red7_groups, blues))
    if len(portfolio.tickets) != EXPANDED_TICKET_COUNT:
        raise ValueError("控制A未形成35注")
    return portfolio


def _portfolio_b(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> tuple[Portfolio, dict[str, object]]:
    document = build_diversified_portfolio_v2(red_probabilities, blue_probabilities)
    validate_diversified_portfolio_v2(
        document,
        red_probabilities=red_probabilities,
        blue_probabilities=blue_probabilities,
    )
    groups_raw = cast(list[dict[str, object]], document["groups"])
    red7_groups = tuple(tuple(cast(list[int], group["red"])) for group in groups_raw)
    blues = tuple(cast(int, group["blue"]) for group in groups_raw)
    portfolio = Portfolio(red7_groups, blues, _expand(red7_groups, blues))
    _validate_portfolio(portfolio)
    return portfolio, document


def build_matched_control_c(issue: str, control_index: int) -> Portfolio:
    """仅由协议、期号和编号生成确定性同成本随机对照 C。"""

    if not issue.isdigit():
        raise ValueError("对照期号必须为数字")
    if not 0 <= control_index < CONTROL_COUNT:
        raise ValueError("对照编号越过固定范围")
    seed = hashlib.sha256(
        f"{CONTROL_SEED_PROTOCOL}|issue={issue}|control={control_index}".encode()
    ).digest()
    rng = _Sha256CounterRng(seed)
    blues = tuple(rng.shuffled(tuple(range(1, 17)))[:GROUP_COUNT])
    red7_groups: list[tuple[int, ...]] = []
    attempts = 0
    while len(red7_groups) < GROUP_COUNT:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("对照C未能在固定安全次数内完成构造")
        red7 = tuple(sorted(rng.shuffled(tuple(range(1, 34)))[:RED_PER_GROUP]))
        if red7 not in red7_groups:
            red7_groups.append(red7)
    portfolio = Portfolio(tuple(red7_groups), blues, _expand(red7_groups, blues))
    _validate_portfolio(portfolio)
    return portfolio


def _score(portfolio: Portfolio, draw: SSQDraw) -> PortfolioScore:
    actual_red = set(draw.red)
    red_hits = tuple(
        len(set(red6).intersection(actual_red)) for red6, _ in portfolio.tickets
    )
    blue_hits = tuple(blue == draw.blue for _, blue in portfolio.tickets)
    maximum = max(red_hits)
    metrics: dict[str, float] = {
        "averageRedHitsPerTicket": sum(red_hits) / EXPANDED_TICKET_COUNT,
        "maximumRedHitsAnyTicket": float(maximum),
        "blueAnyHit": float(any(blue_hits)),
    }
    for threshold in THRESHOLDS:
        metrics[f"anyTicketRedAtLeast{threshold}"] = float(maximum >= threshold)
        metrics[f"ticketRedAtLeast{threshold}PlusBlue"] = float(
            any(red >= threshold and blue for red, blue in zip(red_hits, blue_hits))
        )
        metrics[f"ticketRedAtLeast{threshold}PlusNoBlue"] = float(
            any(red >= threshold and not blue for red, blue in zip(red_hits, blue_hits))
        )
    return PortfolioScore(red_hits, blue_hits, metrics)


def _issue_mean(scores: Sequence[PortfolioScore]) -> dict[str, float]:
    if len(scores) != CONTROL_COUNT:
        raise ValueError("每期对照C必须恰好32组")
    return {
        name: sum(score.metrics[name] for score in scores) / CONTROL_COUNT
        for name in METRIC_NAMES
    }


def _delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {name: left[name] - right[name] for name in METRIC_NAMES}


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


def walk_forward_portfolio_fingerprints(
    draws: Sequence[SSQDraw], stop: int | None = None
) -> list[str]:
    """返回逐期 A/B 构造摘要，供前缀不变与无未来泄漏测试。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    limit = len(ordered) if stop is None else min(stop, len(ordered))
    state = FixedEnsembleState()
    fingerprints: list[str] = []
    for draw in ordered[:limit]:
        red, blue, pairs = state.predict()
        portfolio_a = _portfolio_a(red, blue, pairs)
        portfolio_b, document_b = _portfolio_b(red, blue)
        fingerprints.append(
            _sha256_payload(
                {
                    "A": {"red7": portfolio_a.red7_groups, "blues": portfolio_a.blues},
                    "B": document_b,
                    "B_tickets": portfolio_b.tickets,
                }
            )
        )
        state.score_then_update(draw)
    return fingerprints


def evaluate_full_history(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """运行固定120期预热、其后全部历史的严格先预测后更新评估。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) <= EVALUATION_WARMUP_DRAWS:
        raise ValueError("双色球历史不足120期预热加至少1期评分")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("双色球历史包含重复期号")
    state = FixedEnsembleState()
    for draw in ordered[:EVALUATION_WARMUP_DRAWS]:
        state.score_then_update(draw)

    accumulators = {label: _Accumulator() for label in ("A", "B", "C")}
    differences: dict[str, dict[str, list[float]]] = {
        comparison: {name: [] for name in METRIC_NAMES} for comparison in ("B-A", "B-C")
    }
    per_issue: list[dict[str, object]] = []
    for draw in ordered[EVALUATION_WARMUP_DRAWS:]:
        red_probabilities, blue_probabilities, pair_modifiers = state.predict()
        portfolio_a = _portfolio_a(
            red_probabilities, blue_probabilities, pair_modifiers
        )
        portfolio_b, document_b = _portfolio_b(red_probabilities, blue_probabilities)
        score_a = _score(portfolio_a, draw)
        score_b = _score(portfolio_b, draw)
        accumulators["A"].add(score_a)
        accumulators["B"].add(score_b)
        control_scores: list[PortfolioScore] = []
        for control_index in range(CONTROL_COUNT):
            score_c = _score(build_matched_control_c(draw.issue, control_index), draw)
            control_scores.append(score_c)
            accumulators["C"].add(score_c)
        control_mean = _issue_mean(control_scores)
        for name in METRIC_NAMES:
            differences["B-A"][name].append(
                score_b.metrics[name] - score_a.metrics[name]
            )
            differences["B-C"][name].append(score_b.metrics[name] - control_mean[name])
        per_issue.append(
            {
                "issue": draw.issue,
                "date": draw.draw_date,
                "A": {
                    "red7Groups": [list(group) for group in portfolio_a.red7_groups],
                    "blues": list(portfolio_a.blues),
                    "metrics": dict(score_a.metrics),
                },
                "B": {
                    "groups": document_b["groups"],
                    "audit": document_b["audit"],
                    "metrics": dict(score_b.metrics),
                },
                "C32Mean": control_mean,
            }
        )
        state.score_then_update(draw)

    summaries = {
        label: accumulator.report() for label, accumulator in accumulators.items()
    }
    summary_metrics = {
        label: cast(dict[str, float], summary["metricsPerIssue"])
        for label, summary in summaries.items()
    }
    evaluated = ordered[EVALUATION_WARMUP_DRAWS:]
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedDeterministically": True,
        "researchOnly": True,
        "predictionClaim": False,
        "formalGate": False,
        "formalRecommendationStatus": "uniform_abstain",
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
            "averageRedHitsPerTicket": "每期35注红球命中数平均值",
            "maximumRedHitsAnyTicket": "每期35注任一票最大红球命中数",
            "blueAnyHit": "5个组蓝球中是否包含当期蓝球",
            "thresholdPatterns": (
                "3+/4+/5+/6+均按逐票判断；PlusBlue要求同一票所分配蓝球命中，"
                "PlusNoBlue要求同一票所分配蓝球未命中；两类可在同一期同时发生"
            ),
            "officialPrizeClaims": False,
        },
        "summary": summaries,
        "deltas": {
            "BMinusA": _delta(summary_metrics["B"], summary_metrics["A"]),
            "BMinusC": _delta(summary_metrics["B"], summary_metrics["C"]),
        },
        "descriptivePairedTests": {
            "method": "one-sided paired t-test; alternative B greater",
            "multipleComparisonStatus": "posthoc descriptive unadjusted; no gate",
            "BMinusA": {
                name: _paired_test(differences["B-A"][name]) for name in METRIC_NAMES
            },
            "BMinusC": {
                name: _paired_test(differences["B-C"][name]) for name in METRIC_NAMES
            },
        },
        "assertions": {
            "strictPriorPrediction": True,
            "updateAfterScoring": True,
            "historyTotalPeriods": len(ordered),
            "warmupPeriods": EVALUATION_WARMUP_DRAWS,
            "scorePeriods": len(evaluated),
            "controlAPreservedExactConstruction": True,
            "coverageFirstBuilderProtocolBound": True,
            "fiveDistinctBluesForBEveryIssue": True,
            "redUnion33ForBEveryIssue": True,
            "maximumRedExposure2ForBEveryIssue": True,
            "pairwiseIntersectionAtMost3ForBEveryIssue": True,
            "exactly35UniqueTicketsEveryPortfolio": True,
            "controlsIndependentOfActualResult": True,
            "controlsIndependentOfModel": True,
            "beamWidthForA": BEAM_WIDTH,
            "formalGateChanged": False,
            "modelProbabilitiesChanged": False,
            "onlineUpdateChanged": False,
        },
        "perIssue": per_issue,
    }
    report["reportSha256"] = _sha256_payload(report)
    return report


def write_report(report: Mapping[str, object], output_path: str | Path) -> Path:
    """原子写入稳定排序的历史评估 JSON。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                dict(report),
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
    "CONTROL_SEED_PROTOCOL",
    "METRIC_NAMES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "Portfolio",
    "build_matched_control_c",
    "evaluate_full_history",
    "protocol_sha256",
    "walk_forward_portfolio_fingerprints",
    "write_report",
]
