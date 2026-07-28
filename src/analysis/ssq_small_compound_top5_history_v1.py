# -*- coding: utf-8 -*-
"""双色球 Top5 小复式全历史严格前序回溯评估器。"""

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

from src.analysis.ssq_ensemble_v1 import (
    EVALUATION_WARMUP_DRAWS,
    RESEARCH_RED_AUDIT_COUNT,
    FixedEnsembleState,
    beam_red_combinations,
    blue_top1,
)
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_reference_helpers import (
    _small_compound_7_red_1_blue_top5,
)
from src.lotteries.ssq import SSQ_RULE

SCHEMA_VERSION = "ssq_small_compound_top5_full_history_v1"
MODEL_NAME = "ssq_ensemble_v1"
CONTROL_COUNT = 32
COMPOUND_COUNT = 5
RED_PER_COMPOUND = 7
EXPANDED_PER_COMPOUND = 7
EXPANDED_TICKET_COUNT = 35
CONTROL_SEED_PROTOCOL = (
    "ssq_small_compound_top5_history_v1|matched_cost_control|"
    "sha256_seed_and_counter_rejection_v1"
)
EVENT_NAMES = (
    "anyTicketRedAtLeast3",
    "anyTicketRedAtLeast4",
    "anyTicketRedAtLeast5",
    "anyTicketRedEquals6",
    "pattern6PlusBlue",
    "pattern6PlusNoBlue",
    "pattern5PlusBlue",
    "pattern5PlusNoBlue",
    "pattern4PlusBlue",
    "pattern4PlusNoBlue",
    "pattern3PlusBlue",
    "patternAtMost2PlusBlue",
)
PAIRED_METRIC_NAMES = (
    "averageCompoundRedHits",
    "averageExpandedTicketRedHits",
    "maximumTicketRedHits",
    "blueHit",
    *EVENT_NAMES,
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
    "purpose": "retrospective_research_only",
    "model": MODEL_NAME,
    "walkForward": {
        "warmupDraws": EVALUATION_WARMUP_DRAWS,
        "ordering": "official_history_sorted_by_numeric_issue_ascending",
        "sequence": (
            "predict_from_prior_state_then_build_top32_red6_and_blue_top1_then_"
            "construct_and_score_top5_compounds_then_update_state"
        ),
        "futureProbabilitiesStored": False,
    },
    "portfolio": {
        "compoundCount": COMPOUND_COUNT,
        "redPerCompound": RED_PER_COMPOUND,
        "sharedBlue": True,
        "expandedTicketsPerCompound": EXPANDED_PER_COMPOUND,
        "globallyUniqueExpandedTickets": EXPANDED_TICKET_COUNT,
        "construction": "exact_current__small_compound_7_red_1_blue_top5",
        "constructionCandidates": (
            "full fixed beam of 256 red6 rankings; first 32 retained as compact audit"
        ),
    },
    "matchedControl": {
        "controlsPerIssue": CONTROL_COUNT,
        "seedProtocol": CONTROL_SEED_PROTOCOL,
        "seedInput": "UTF-8 protocol string + issue + zero-based control index",
        "generator": (
            "SHA256 seed followed by SHA256 counter-mode rejection sampling and "
            "Fisher-Yates shuffles"
        ),
        "resultDependence": False,
        "modelDependence": False,
        "constraints": (
            "five distinct red7 sets, one shared blue, exactly 35 globally "
            "unique expanded red6+blue tickets"
        ),
    },
    "statistics": {
        "method": "one-sided paired t-test of per-issue model-minus-control-mean",
        "alternative": "greater",
        "metrics": list(PAIRED_METRIC_NAMES),
        "multipleComparison": "post-hoc descriptive family; no correction gate",
        "formalGate": False,
        "pHacking": False,
    },
}


def protocol_sha256() -> str:
    """返回固定回溯协议的稳定 SHA-256。"""

    return _sha256_payload(PROTOCOL)


class _Sha256CounterRng:
    """跨运行稳定的 SHA-256 计数器随机源。"""

    def __init__(self, seed: bytes) -> None:
        self._seed = seed
        self._counter = 0

    def _uint256(self) -> int:
        digest = hashlib.sha256(self._seed + self._counter.to_bytes(16, "big")).digest()
        self._counter += 1
        return int.from_bytes(digest, "big")

    def randbelow(self, upper: int) -> int:
        """通过拒绝采样返回 ``[0, upper)`` 的无偏整数。"""

        if upper <= 0:
            raise ValueError("随机上界必须为正数")
        space = 1 << 256
        limit = space - (space % upper)
        while True:
            value = self._uint256()
            if value < limit:
                return value % upper

    def shuffled(self, values: Sequence[int]) -> list[int]:
        """返回使用固定 Fisher-Yates 算法洗牌后的副本。"""

        shuffled = list(values)
        for index in range(len(shuffled) - 1, 0, -1):
            other = self.randbelow(index + 1)
            shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
        return shuffled


@dataclass(frozen=True)
class ControlPortfolio:
    """一组固定成本随机对照组合。"""

    red7_sets: tuple[tuple[int, ...], ...]
    shared_blue: int
    expanded_tickets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PortfolioScore:
    """一组五小复式组合在一期上的统一评分结果。"""

    compound_red_hits: tuple[int, ...]
    shared_blue_hit: bool
    ticket_red_hit_distribution: tuple[int, ...]
    max_red_hits_any_ticket: int
    events: Mapping[str, bool]

    @property
    def average_compound_red_hits(self) -> float:
        return sum(self.compound_red_hits) / COMPOUND_COUNT

    @property
    def average_expanded_ticket_red_hits(self) -> float:
        total = sum(
            hits * count for hits, count in enumerate(self.ticket_red_hit_distribution)
        )
        return total / EXPANDED_TICKET_COUNT

    def paired_metrics(self) -> dict[str, float]:
        """返回配对统计使用的逐期数值。"""

        return {
            "averageCompoundRedHits": self.average_compound_red_hits,
            "averageExpandedTicketRedHits": self.average_expanded_ticket_red_hits,
            "maximumTicketRedHits": float(self.max_red_hits_any_ticket),
            "blueHit": float(self.shared_blue_hit),
            **{name: float(self.events[name]) for name in EVENT_NAMES},
        }


@dataclass
class _MetricAccumulator:
    periods: int = 0
    compound_red_hit_distribution: list[int] = field(default_factory=lambda: [0] * 7)
    best_compound_red_distribution: list[int] = field(default_factory=lambda: [0] * 7)
    ticket_red_hit_distribution: list[int] = field(default_factory=lambda: [0] * 7)
    max_ticket_red_distribution: list[int] = field(default_factory=lambda: [0] * 7)
    compound_red_hit_total: int = 0
    ticket_red_hit_total: int = 0
    max_ticket_red_total: int = 0
    blue_hits: int = 0
    event_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in EVENT_NAMES}
    )

    def add(self, score: PortfolioScore) -> None:
        """累加一组组合的一期评分。"""

        self.periods += 1
        for hits in score.compound_red_hits:
            self.compound_red_hit_distribution[hits] += 1
            self.compound_red_hit_total += hits
        best_compound = max(score.compound_red_hits)
        self.best_compound_red_distribution[best_compound] += 1
        for hits, count in enumerate(score.ticket_red_hit_distribution):
            self.ticket_red_hit_distribution[hits] += count
            self.ticket_red_hit_total += hits * count
        self.max_ticket_red_distribution[score.max_red_hits_any_ticket] += 1
        self.max_ticket_red_total += score.max_red_hits_any_ticket
        self.blue_hits += int(score.shared_blue_hit)
        for name in EVENT_NAMES:
            self.event_counts[name] += int(score.events[name])

    def report(self) -> dict[str, object]:
        """生成完整、可复核的聚合指标。"""

        if self.periods <= 0:
            raise ValueError("空评估不能生成指标")
        compound_observations = self.periods * COMPOUND_COUNT
        ticket_observations = self.periods * EXPANDED_TICKET_COUNT
        return {
            "periods": self.periods,
            "compoundLevel": {
                "observations": compound_observations,
                "redHitDistribution": _distribution(self.compound_red_hit_distribution),
                "averageRedHitsPer7RedCompound": (
                    self.compound_red_hit_total / compound_observations
                ),
            },
            "periodLevelBestCompound": {
                "redCoverageDistribution": _distribution(
                    self.best_compound_red_distribution
                ),
                "averageBestRedCoverage": sum(
                    hits * count
                    for hits, count in enumerate(self.best_compound_red_distribution)
                )
                / self.periods,
            },
            "expanded35Tickets": {
                "observations": ticket_observations,
                "redHitDistribution": _distribution(self.ticket_red_hit_distribution),
                "averageRedHitsPerTicket": (
                    self.ticket_red_hit_total / ticket_observations
                ),
                "maxRedHitsPerPeriodDistribution": _distribution(
                    self.max_ticket_red_distribution
                ),
                "averageMaxRedHitsPerPeriod": (
                    self.max_ticket_red_total / self.periods
                ),
            },
            "sharedBlue": {
                "hitCount": self.blue_hits,
                "hitRate": self.blue_hits / self.periods,
            },
            "events": {
                name: {
                    "count": self.event_counts[name],
                    "rate": self.event_counts[name] / self.periods,
                }
                for name in EVENT_NAMES
            },
        }


def _distribution(counts: Sequence[int]) -> dict[str, int]:
    return {str(index): count for index, count in enumerate(counts)}


def _expanded_red_tickets(
    red7_sets: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    expanded = tuple(
        tuple(red6) for red7 in red7_sets for red6 in combinations(red7, 6)
    )
    if len(expanded) != EXPANDED_TICKET_COUNT or len(set(expanded)) != len(expanded):
        raise ValueError("五组小复式未形成35注全局唯一红6票")
    return expanded


def build_matched_control(issue: str, control_index: int) -> ControlPortfolio:
    """仅由固定协议、期号和对照编号构造一组确定性同成本对照。"""

    if not issue.isdigit():
        raise ValueError("对照期号必须为数字")
    if control_index < 0 or control_index >= CONTROL_COUNT:
        raise ValueError("对照编号越过固定范围")
    seed_material = (
        f"{CONTROL_SEED_PROTOCOL}|issue={issue}|control={control_index}"
    ).encode("utf-8")
    seed = hashlib.sha256(seed_material).digest()
    rng = _Sha256CounterRng(seed)
    shared_blue = rng.randbelow(16) + 1
    red7_sets: list[tuple[int, ...]] = []
    expanded_seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(red7_sets) < COMPOUND_COUNT:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("确定性对照在安全尝试次数内未能构造合法组合")
        red7 = tuple(sorted(rng.shuffled(tuple(range(1, 34)))[:RED_PER_COMPOUND]))
        expanded = {tuple(red6) for red6 in combinations(red7, 6)}
        if red7 in red7_sets or expanded.intersection(expanded_seen):
            continue
        red7_sets.append(red7)
        expanded_seen.update(expanded)
    expanded_tickets = _expanded_red_tickets(red7_sets)
    return ControlPortfolio(tuple(red7_sets), shared_blue, expanded_tickets)


def _score_portfolio(
    red7_sets: Sequence[Sequence[int]],
    shared_blue: int,
    expanded_tickets: Sequence[Sequence[int]],
    draw: SSQDraw,
) -> PortfolioScore:
    actual_red = set(draw.red)
    compound_hits = tuple(len(set(red7).intersection(actual_red)) for red7 in red7_sets)
    ticket_hits = [
        len(set(ticket).intersection(actual_red)) for ticket in expanded_tickets
    ]
    distribution = tuple(ticket_hits.count(hits) for hits in range(7))
    if sum(distribution) != EXPANDED_TICKET_COUNT:
        raise ValueError("展开票评分数量不是35")
    maximum = max(ticket_hits)
    blue_hit = shared_blue == draw.blue
    events = {
        "anyTicketRedAtLeast3": maximum >= 3,
        "anyTicketRedAtLeast4": maximum >= 4,
        "anyTicketRedAtLeast5": maximum >= 5,
        "anyTicketRedEquals6": maximum == 6,
        "pattern6PlusBlue": maximum == 6 and blue_hit,
        "pattern6PlusNoBlue": maximum == 6 and not blue_hit,
        "pattern5PlusBlue": maximum == 5 and blue_hit,
        "pattern5PlusNoBlue": maximum == 5 and not blue_hit,
        "pattern4PlusBlue": maximum == 4 and blue_hit,
        "pattern4PlusNoBlue": maximum == 4 and not blue_hit,
        "pattern3PlusBlue": maximum == 3 and blue_hit,
        "patternAtMost2PlusBlue": maximum <= 2 and blue_hit,
    }
    return PortfolioScore(compound_hits, blue_hit, distribution, maximum, events)


def _model_portfolio(
    state: FixedEnsembleState,
) -> tuple[
    list[dict[str, object]],
    int,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    list[dict[str, object]],
]:
    red_probabilities, blue_probabilities, pair_modifiers = state.predict()
    ranked = beam_red_combinations(red_probabilities, pair_modifiers)
    if len(ranked) < RESEARCH_RED_AUDIT_COUNT:
        raise ValueError("束搜索红6审计候选不足32个")
    ranked_entries = [
        {
            "rank": index,
            "red": [f"{ball:02d}" for ball in red],
            "redScore": score,
        }
        for index, (red, score) in enumerate(ranked, start=1)
    ]
    audit = ranked_entries[:RESEARCH_RED_AUDIT_COUNT]
    shared_blue = blue_top1(blue_probabilities)
    construction = _small_compound_7_red_1_blue_top5(
        ranked_entries, f"{shared_blue:02d}"
    )
    compounds = cast(list[dict[str, object]], construction["compounds"])
    red7_sets = tuple(
        tuple(int(value) for value in cast(list[str], compound["red"]))
        for compound in compounds
    )
    expanded_tickets = tuple(
        tuple(int(value) for value in cast(list[str], ticket["red"]))
        for compound in compounds
        for ticket in cast(list[dict[str, object]], compound["expandedTickets"])
    )
    construction_audit = [
        {
            "baseRank": compound["baseRank"],
            "pairedLaterRank": compound["pairedLaterRank"],
            "addedRed": compound["addedRed"],
        }
        for compound in compounds
    ]
    _validate_portfolio(red7_sets, shared_blue, expanded_tickets)
    return audit, shared_blue, red7_sets, expanded_tickets, construction_audit


def _validate_portfolio(
    red7_sets: Sequence[Sequence[int]],
    shared_blue: int,
    expanded_tickets: Sequence[Sequence[int]],
) -> None:
    if len(red7_sets) != COMPOUND_COUNT or len({tuple(red) for red in red7_sets}) != 5:
        raise ValueError("组合必须包含5组互异红7")
    for red7 in red7_sets:
        if len(red7) != RED_PER_COMPOUND or len(set(red7)) != RED_PER_COMPOUND:
            raise ValueError("每组小复式必须恰好包含7个互异红球")
        if tuple(sorted(red7)) != tuple(red7) or not all(
            1 <= value <= 33 for value in red7
        ):
            raise ValueError("小复式红球必须合法且升序")
    if not 1 <= shared_blue <= 16:
        raise ValueError("共享蓝球非法")
    expected = _expanded_red_tickets(red7_sets)
    if tuple(expanded_tickets) != expected:
        raise ValueError("展开票与五组红7的固定组合展开不一致")
    for ticket in expanded_tickets:
        SSQ_RULE.validate_draw(ticket, shared_blue)


def _control_issue_means(scores: Sequence[PortfolioScore]) -> dict[str, float]:
    if len(scores) != CONTROL_COUNT:
        raise ValueError("每期必须恰好包含32组对照")
    metrics = [score.paired_metrics() for score in scores]
    return {
        name: sum(metric[name] for metric in metrics) / CONTROL_COUNT
        for name in PAIRED_METRIC_NAMES
    }


def _paired_test(differences: Sequence[float]) -> dict[str, object]:
    if not differences:
        raise ValueError("配对检验缺少观测")
    mean_difference = sum(differences) / len(differences)
    if all(value == differences[0] for value in differences):
        statistic: float | None = 0.0 if mean_difference == 0.0 else None
        p_value = 1.0 if mean_difference <= 0.0 else 0.0
    else:
        result = stats.ttest_1samp(differences, popmean=0.0, alternative="greater")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(statistic) or not math.isfinite(p_value):
            statistic = None
            p_value = 1.0 if mean_difference <= 0.0 else 0.0
    return {
        "observations": len(differences),
        "meanModelMinusControl": mean_difference,
        "statistic": statistic,
        "oneSidedPValue": p_value,
        "method": "scipy.stats.ttest_1samp",
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


def evaluate_full_history(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """运行固定 120 期预热后的全历史严格前序回溯。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) <= EVALUATION_WARMUP_DRAWS:
        raise ValueError("双色球历史不足120期预热加至少1期评估")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("双色球历史包含重复期号")
    state = FixedEnsembleState()
    for draw in ordered[:EVALUATION_WARMUP_DRAWS]:
        state.score_then_update(draw)

    model_accumulator = _MetricAccumulator()
    control_accumulator = _MetricAccumulator()
    paired_differences: dict[str, list[float]] = {
        name: [] for name in PAIRED_METRIC_NAMES
    }
    per_issue: list[dict[str, object]] = []

    for draw in ordered[EVALUATION_WARMUP_DRAWS:]:
        (
            audit,
            shared_blue,
            red7_sets,
            expanded_tickets,
            construction_audit,
        ) = _model_portfolio(state)
        model_score = _score_portfolio(red7_sets, shared_blue, expanded_tickets, draw)
        model_accumulator.add(model_score)

        control_scores: list[PortfolioScore] = []
        for control_index in range(CONTROL_COUNT):
            control = build_matched_control(draw.issue, control_index)
            control_score = _score_portfolio(
                control.red7_sets,
                control.shared_blue,
                control.expanded_tickets,
                draw,
            )
            control_scores.append(control_score)
            control_accumulator.add(control_score)
        control_means = _control_issue_means(control_scores)
        model_metrics = model_score.paired_metrics()
        for name in PAIRED_METRIC_NAMES:
            paired_differences[name].append(model_metrics[name] - control_means[name])

        per_issue.append(
            {
                "issue": draw.issue,
                "date": draw.draw_date,
                "red6AuditTop32": audit,
                "constructionCandidateCount": 256,
                "compoundConstructionAudit": construction_audit,
                "compoundRedSets": [list(red7) for red7 in red7_sets],
                "sharedBlue": shared_blue,
                "compoundRedHitCounts": list(model_score.compound_red_hits),
                "sharedBlueHit": model_score.shared_blue_hit,
                "expanded35TicketRedHitDistribution": _distribution(
                    model_score.ticket_red_hit_distribution
                ),
                "maxRedHitsAnyTicket": model_score.max_red_hits_any_ticket,
                "patterns": dict(model_score.events),
                "matchedControlMeans": control_means,
            }
        )
        state.score_then_update(draw)

    model_metrics_report = model_accumulator.report()
    control_metrics_report = control_accumulator.report()
    model_events = cast(dict[str, dict[str, float]], model_metrics_report["events"])
    control_events = cast(dict[str, dict[str, float]], control_metrics_report["events"])
    model_compound = cast(dict[str, object], model_metrics_report["compoundLevel"])
    control_compound = cast(dict[str, object], control_metrics_report["compoundLevel"])
    model_tickets = cast(dict[str, object], model_metrics_report["expanded35Tickets"])
    control_tickets = cast(
        dict[str, object], control_metrics_report["expanded35Tickets"]
    )
    model_blue = cast(dict[str, object], model_metrics_report["sharedBlue"])
    control_blue = cast(dict[str, object], control_metrics_report["sharedBlue"])
    deltas = {
        "averageCompoundRedHits": cast(
            float, model_compound["averageRedHitsPer7RedCompound"]
        )
        - cast(float, control_compound["averageRedHitsPer7RedCompound"]),
        "averageExpandedTicketRedHits": cast(
            float, model_tickets["averageRedHitsPerTicket"]
        )
        - cast(float, control_tickets["averageRedHitsPerTicket"]),
        "maximumTicketRedHits": cast(float, model_tickets["averageMaxRedHitsPerPeriod"])
        - cast(float, control_tickets["averageMaxRedHitsPerPeriod"]),
        "blueHit": cast(float, model_blue["hitRate"])
        - cast(float, control_blue["hitRate"]),
        **{
            name: model_events[name]["rate"] - control_events[name]["rate"]
            for name in EVENT_NAMES
        },
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
        "theoreticalBaselines": {
            "single7RedCompoundMeanRedHits": 42.0 / 33.0,
            "single6RedTicketMeanRedHits": 36.0 / 33.0,
            "sharedBlueHitProbability": 1.0 / 16.0,
            "exact6PlusBlueUnique35PortfolioProbability": (
                EXPANDED_TICKET_COUNT / (math.comb(33, 6) * 16)
            ),
            "exact6PlusBlueFormula": "35/(C(33,6)*16)",
        },
        "hitPatternDefinitions": {
            "scope": "按每期35注中的最大红球命中数与共享蓝球是否命中分类",
            "labelsAreOfficialPrizeClaims": False,
            "patterns": {
                "pattern6PlusBlue": "maxRedHitsAnyTicket=6 且共享蓝球命中",
                "pattern6PlusNoBlue": "maxRedHitsAnyTicket=6 且共享蓝球未命中",
                "pattern5PlusBlue": "maxRedHitsAnyTicket=5 且共享蓝球命中",
                "pattern5PlusNoBlue": "maxRedHitsAnyTicket=5 且共享蓝球未命中",
                "pattern4PlusBlue": "maxRedHitsAnyTicket=4 且共享蓝球命中",
                "pattern4PlusNoBlue": "maxRedHitsAnyTicket=4 且共享蓝球未命中",
                "pattern3PlusBlue": "maxRedHitsAnyTicket=3 且共享蓝球命中",
                "patternAtMost2PlusBlue": ("maxRedHitsAnyTicket<=2 且共享蓝球命中"),
            },
        },
        "model": {
            "metrics": model_metrics_report,
        },
        "matchedControl": {
            "controlsPerIssue": CONTROL_COUNT,
            "portfolioObservations": len(evaluated) * CONTROL_COUNT,
            "protocolSha256": protocol_sha256(),
            "seedProtocol": CONTROL_SEED_PROTOCOL,
            "noFutureDependence": True,
            "noResultDependence": True,
            "noModelDependence": True,
            "metrics": control_metrics_report,
        },
        "modelMinusMatchedControl": deltas,
        "descriptivePairedTests": {
            "method": (
                "one-sided paired t-test over each issue's model metric minus "
                "the mean of its 32 matched controls"
            ),
            "alternative": "greater",
            "multipleComparisonStatus": (
                "multiple-comparison post-hoc descriptive family; unadjusted; "
                "no formal gate"
            ),
            "tests": {
                name: _paired_test(paired_differences[name])
                for name in PAIRED_METRIC_NAMES
            },
        },
        "assertions": {
            "strictPriorPrediction": True,
            "updateAfterScoring": True,
            "exactlyFiveDistinctCompoundsEveryIssue": True,
            "sharedBlueEveryIssue": True,
            "exactly35GloballyUniqueExpandedTicketsEveryIssue": True,
            "controlsIndependentOfActualResult": True,
            "futureProbabilitiesStored": False,
            "modelChanged": False,
            "tuningChanged": False,
            "protocolOrFormalGateChanged": False,
            "dataOrStateChanged": False,
        },
        "perIssue": per_issue,
    }
    report["reportSha256"] = _sha256_payload(report)
    return report


def write_report(report: Mapping[str, object], output_path: str | Path) -> Path:
    """原子写入稳定排序的 JSON 报告。"""

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
    "EVENT_NAMES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "ControlPortfolio",
    "PortfolioScore",
    "build_matched_control",
    "evaluate_full_history",
    "protocol_sha256",
    "write_report",
]
