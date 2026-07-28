# -*- coding: utf-8 -*-
"""双色球静态约束权重训练 v2：一次性严格抗过拟合挑战。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence, cast

from src.analysis.ssq_ensemble_v1 import (
    BLUE_UNIFORM_BRIER,
    BLUE_UNIFORM_LOG_LOSS,
    BLUE_UNIFORM_PROBABILITY,
    EVALUATION_WARMUP_DRAWS,
    EXACT_TOP20_UNIFORM_PROBABILITY,
    PAIR_MODIFIER_WEIGHT,
    RED_UNIFORM_BRIER,
    RED_UNIFORM_LOG_LOSS,
    RED_UNIFORM_PROBABILITY,
    BlueMarginalExpert,
    RedMarginalExpert,
    RedPairExpert,
    _blue_brier,
    _blue_log_loss,
    _mixture,
    _red_brier,
    _red_log_loss,
    blue_top1,
    generate_research_tickets,
)
from src.analysis.ssq_ensemble_v1 import protocol_sha256 as v1_protocol_sha256
from src.analysis.ssq_ensemble_v1 import (
    red_top6,
)
from src.analysis.ssq_history import SSQDraw

MODEL_NAME = "ssq_weight_training_v2"
GRID_STEP = 0.10
VALIDATION_DRAWS = 500
FROZEN_TEST_DRAWS = 500
BLOCK_SIZE = 100
RED_EXPERT_NAMES = ("uniform", "ewma30", "ewma120")
BLUE_EXPERT_NAMES = ("uniform", "ewma60")
RED_UNIFORM_WEIGHTS = (1.0, 0.0, 0.0)
BLUE_UNIFORM_WEIGHTS = (1.0, 0.0)
RED_COVERAGE_BASELINE = 36.0 / 33.0
BLUE_COVERAGE_BASELINE = 1.0 / 16.0
COMPARISON_TOLERANCE = 1e-12
MetricValue = float | int | bool
MetricReport = dict[str, MetricValue]


PROTOCOL: dict[str, object] = {
    "model": MODEL_NAME,
    "purpose": "仅研究双色球静态集成权重，不修改ssq_ensemble_v1且不自动启用",
    "data": "仅复用data/ssq/official_history.csv与v1严格前序专家概率",
    "split": {
        "warmup": EVALUATION_WARMUP_DRAWS,
        "search": "预热后、Validation之前的全部较早合格期次",
        "validation": VALIDATION_DRAWS,
        "frozenTest": FROZEN_TEST_DRAWS,
        "chronology": "升序；最新500期Frozen，其前500期Validation",
    },
    "weights": {
        "redExperts": list(RED_EXPERT_NAMES),
        "redGrid": "非负单纯形，步长0.10",
        "blueExperts": list(BLUE_EXPERT_NAMES),
        "blueGrid": "非负单纯形，步长0.10",
        "pairModifierWeight": PAIR_MODIFIER_WEIGHT,
        "pairModifierSearched": False,
    },
    "selection": {
        "segment": "Search only",
        "eligibility": (
            "红Bernoulli LogLoss/Brier、蓝多分类LogLoss/Brier均不劣于均匀，"
            "且红Top6与蓝Top1覆盖分别不劣于均匀基线"
        ),
        "order": [
            "最低四项平均归一化proper-score指数",
            "最高红蓝平均归一化coverage指数",
            "红权重元组后蓝权重元组字典序最小",
        ],
        "retries": 0,
    },
    "gates": {
        "validation": "全部5个完整100期块及汇总逐项通过六项门槛",
        "frozenTest": "仅Validation通过后开封；同样逐项通过六项门槛",
    },
    "fixed": {
        "cliParameterOverrides": False,
        "windows": [30, 120, 60],
        "features": list(RED_EXPERT_NAMES) + list(BLUE_EXPERT_NAMES),
        "topK": 20,
        "gridStep": GRID_STEP,
        "objective": "固定字典序选择规则",
        "splitLengths": [EVALUATION_WARMUP_DRAWS, 500, 500],
        "gatesOverridable": False,
    },
    "output": {
        "researchOnly": True,
        "formalCandidates": [],
        "automaticProductionActivation": False,
        "exactTop20": "仅描述，不参与选择或闸门",
    },
}


def protocol_sha256() -> str:
    """返回 v2 固定协议的稳定 SHA-256。"""

    serialized = json.dumps(
        PROTOCOL,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _simplex_grid(size: int) -> tuple[tuple[float, ...], ...]:
    units = round(1.0 / GRID_STEP)

    def build(prefix: tuple[int, ...], remaining: int) -> list[tuple[int, ...]]:
        if len(prefix) == size - 1:
            return [(*prefix, remaining)]
        rows: list[tuple[int, ...]] = []
        for value in range(remaining + 1):
            rows.extend(build((*prefix, value), remaining - value))
        return rows

    return tuple(tuple(value / units for value in row) for row in build((), units))


RED_WEIGHT_GRID = _simplex_grid(3)
BLUE_WEIGHT_GRID = _simplex_grid(2)


@dataclass
class StaticExpertState:
    """复用 v1 固定专家，仅维护严格前序统计，不维护在线聚合权重。"""

    red30: RedMarginalExpert = field(default_factory=lambda: RedMarginalExpert(30))
    red120: RedMarginalExpert = field(default_factory=lambda: RedMarginalExpert(120))
    blue60: BlueMarginalExpert = field(default_factory=lambda: BlueMarginalExpert(60))
    pair120: RedPairExpert = field(default_factory=RedPairExpert)

    def observe_prior(self, draw: SSQDraw) -> "PriorObservation":
        """先输出 v1 专家概率，再用当期开奖更新状态。"""

        observation = PriorObservation(
            draw=draw,
            red_experts=(
                tuple([RED_UNIFORM_PROBABILITY] * 33),
                tuple(self.red30.probabilities()),
                tuple(self.red120.probabilities()),
            ),
            blue_experts=(
                tuple([BLUE_UNIFORM_PROBABILITY] * 16),
                tuple(self.blue60.probabilities()),
            ),
            pair_modifiers=tuple(self.pair120.modifiers()),
        )
        self.update(draw)
        return observation

    def update(self, draw: SSQDraw) -> None:
        """仅在当前期预测已产生后更新全部 v1 专家。"""

        self.red30.update(draw.red)
        self.red120.update(draw.red)
        self.blue60.update(draw.blue)
        self.pair120.update(draw.red)


@dataclass(frozen=True)
class PriorObservation:
    """单期严格前序专家输出及当期真实开奖。"""

    draw: SSQDraw
    red_experts: tuple[tuple[float, ...], ...]
    blue_experts: tuple[tuple[float, ...], ...]
    pair_modifiers: tuple[float, ...]


@dataclass
class MetricAccumulator:
    """累加固定静态权重的 proper score 与覆盖指标。"""

    count: int = 0
    red_hits: float = 0.0
    blue_hits: float = 0.0
    red_log_loss: float = 0.0
    red_brier: float = 0.0
    blue_log_loss: float = 0.0
    blue_brier: float = 0.0
    exact_top20_hits: int = 0

    def add(
        self,
        observation: PriorObservation,
        red_weights: Sequence[float],
        blue_weights: Sequence[float],
        *,
        descriptive_top20: bool,
        exact_top20_hit: int | None = None,
    ) -> None:
        """加入一期；Top20 始终只作描述。"""

        red_probabilities = _mixture(observation.red_experts, red_weights)
        blue_probabilities = _mixture(observation.blue_experts, blue_weights)
        self.count += 1
        self.red_hits += len(
            set(red_top6(red_probabilities)).intersection(observation.draw.red)
        )
        self.blue_hits += float(blue_top1(blue_probabilities) == observation.draw.blue)
        self.red_log_loss += _red_log_loss(red_probabilities, observation.draw.red)
        self.red_brier += _red_brier(red_probabilities, observation.draw.red)
        self.blue_log_loss += _blue_log_loss(blue_probabilities, observation.draw.blue)
        self.blue_brier += _blue_brier(blue_probabilities, observation.draw.blue)
        if exact_top20_hit is not None:
            self.exact_top20_hits += exact_top20_hit
        elif descriptive_top20:
            tickets = generate_research_tickets(
                red_probabilities,
                blue_probabilities,
                observation.pair_modifiers,
            )
            actual_red = [f"{ball:02d}" for ball in observation.draw.red]
            self.exact_top20_hits += int(
                any(
                    ticket["red"] == actual_red
                    and ticket["blue"] == f"{observation.draw.blue:02d}"
                    for ticket in tickets
                )
            )

    def report(self, *, include_top20: bool) -> MetricReport:
        """生成固定字段指标报告。"""

        if self.count <= 0:
            raise ValueError("空评估段不能生成指标")
        report: MetricReport = {
            "draws": self.count,
            "redTop6MeanHits": self.red_hits / self.count,
            "blueTop1HitRate": self.blue_hits / self.count,
            "redLogLossPerBall": self.red_log_loss / self.count,
            "redBrierPerBall": self.red_brier / self.count,
            "blueLogLoss": self.blue_log_loss / self.count,
            "blueBrier": self.blue_brier / self.count,
        }
        report["properScoreIndexVsUniform"] = _proper_score_index(report)
        report["coverageIndexVsUniform"] = _coverage_index(report)
        if include_top20:
            report["exactTop20Hits"] = self.exact_top20_hits
            report["exactTop20ExpectedHitsUniform"] = (
                self.count * EXACT_TOP20_UNIFORM_PROBABILITY
            )
        return report


def _proper_score_index(metrics: Mapping[str, MetricValue]) -> float:
    return (
        float(metrics["redLogLossPerBall"]) / RED_UNIFORM_LOG_LOSS
        + float(metrics["redBrierPerBall"]) / RED_UNIFORM_BRIER
        + float(metrics["blueLogLoss"]) / BLUE_UNIFORM_LOG_LOSS
        + float(metrics["blueBrier"]) / BLUE_UNIFORM_BRIER
    ) / 4.0


def _coverage_index(metrics: Mapping[str, MetricValue]) -> float:
    return (
        float(metrics["redTop6MeanHits"]) / RED_COVERAGE_BASELINE
        + float(metrics["blueTop1HitRate"]) / BLUE_COVERAGE_BASELINE
    ) / 2.0


def _gate_reasons(metrics: Mapping[str, MetricValue]) -> list[str]:
    checks = (
        ("redLogLossPerBall", RED_UNIFORM_LOG_LOSS, "le"),
        ("redBrierPerBall", RED_UNIFORM_BRIER, "le"),
        ("blueLogLoss", BLUE_UNIFORM_LOG_LOSS, "le"),
        ("blueBrier", BLUE_UNIFORM_BRIER, "le"),
        ("redTop6MeanHits", RED_COVERAGE_BASELINE, "ge"),
        ("blueTop1HitRate", BLUE_COVERAGE_BASELINE, "ge"),
    )
    reasons: list[str] = []
    for name, baseline, direction in checks:
        value = float(metrics[name])
        passed = (
            value <= baseline + COMPARISON_TOLERANCE
            if direction == "le"
            else value + COMPARISON_TOLERANCE >= baseline
        )
        if not passed:
            operator = "<=" if direction == "le" else ">="
            reasons.append(
                f"{name}={value:.12g} 未满足均匀门槛 {operator}{baseline:.12g}"
            )
    return reasons


def _candidate_summary(
    observations: Sequence[PriorObservation],
    red_weights: tuple[float, ...],
    blue_weights: tuple[float, ...],
) -> dict[str, object]:
    accumulator = MetricAccumulator()
    for observation in observations:
        accumulator.add(
            observation,
            red_weights,
            blue_weights,
            descriptive_top20=False,
        )
    metrics = accumulator.report(include_top20=False)
    reasons = _gate_reasons(metrics)
    return {
        "redWeights": list(red_weights),
        "blueWeights": list(blue_weights),
        "metrics": metrics,
        "eligible": not reasons,
        "gateReasons": reasons,
        "exactTop20": "not_computed_not_used_for_selection",
    }


def _is_uniform_candidate(summary: Mapping[str, object]) -> bool:
    return (
        tuple(cast(Sequence[float], summary["redWeights"])) == RED_UNIFORM_WEIGHTS
        and tuple(cast(Sequence[float], summary["blueWeights"])) == BLUE_UNIFORM_WEIGHTS
    )


def _select_search_winner(
    summaries: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    eligible = [
        summary
        for summary in summaries
        if bool(summary["eligible"]) and not _is_uniform_candidate(summary)
    ]
    if not eligible:
        return None

    def key(summary: Mapping[str, object]) -> tuple[object, ...]:
        metrics = summary["metrics"]
        if not isinstance(metrics, Mapping):
            raise TypeError("候选指标结构非法")
        return (
            float(metrics["properScoreIndexVsUniform"]),
            -float(metrics["coverageIndexVsUniform"]),
            tuple(cast(Sequence[float], summary["redWeights"])),
            tuple(cast(Sequence[float], summary["blueWeights"])),
        )

    return min(eligible, key=key)


def _segment_report(
    draws: Sequence[SSQDraw],
    state: StaticExpertState,
    start: int,
    end: int,
    red_weights: tuple[float, ...],
    blue_weights: tuple[float, ...],
) -> dict[str, object]:
    if end - start != VALIDATION_DRAWS:
        raise ValueError("Validation/Frozen 固定段必须恰好500期")
    aggregate = MetricAccumulator()
    blocks = [MetricAccumulator() for _ in range(VALIDATION_DRAWS // BLOCK_SIZE)]
    for index in range(start, end):
        observation = state.observe_prior(draws[index])
        red_probabilities = _mixture(observation.red_experts, red_weights)
        blue_probabilities = _mixture(observation.blue_experts, blue_weights)
        tickets = generate_research_tickets(
            red_probabilities,
            blue_probabilities,
            observation.pair_modifiers,
        )
        actual_red = [f"{ball:02d}" for ball in observation.draw.red]
        exact_top20_hit = int(
            any(
                ticket["red"] == actual_red
                and ticket["blue"] == f"{observation.draw.blue:02d}"
                for ticket in tickets
            )
        )
        aggregate.add(
            observation,
            red_weights,
            blue_weights,
            descriptive_top20=False,
            exact_top20_hit=exact_top20_hit,
        )
        blocks[(index - start) // BLOCK_SIZE].add(
            observation,
            red_weights,
            blue_weights,
            descriptive_top20=False,
            exact_top20_hit=exact_top20_hit,
        )
    aggregate_report = aggregate.report(include_top20=True)
    block_reports: list[dict[str, object]] = []
    reasons: list[str] = []
    for block_index, accumulator in enumerate(blocks):
        metrics = accumulator.report(include_top20=True)
        block_reasons = _gate_reasons(metrics)
        block_reports.append(
            {
                "block": block_index + 1,
                "startIndex": start + block_index * BLOCK_SIZE,
                "endIndexExclusive": start + (block_index + 1) * BLOCK_SIZE,
                "metrics": metrics,
                "passed": not block_reasons,
                "gateReasons": block_reasons,
            }
        )
        reasons.extend(f"block{block_index + 1}: {reason}" for reason in block_reasons)
    aggregate_reasons = _gate_reasons(aggregate_report)
    reasons.extend(f"aggregate: {reason}" for reason in aggregate_reasons)
    return {
        "opened": True,
        "aggregate": aggregate_report,
        "blocks": block_reports,
        "passed": not reasons,
        "gateReasons": reasons,
    }


def _selected_search_top20(
    observations: Sequence[PriorObservation],
    red_weights: tuple[float, ...],
    blue_weights: tuple[float, ...],
) -> dict[str, object]:
    accumulator = MetricAccumulator()
    for observation in observations:
        accumulator.add(
            observation,
            red_weights,
            blue_weights,
            descriptive_top20=True,
        )
    metrics = accumulator.report(include_top20=True)
    return {
        "exactTop20Hits": metrics["exactTop20Hits"],
        "exactTop20ExpectedHitsUniform": metrics["exactTop20ExpectedHitsUniform"],
        "statement": "稀疏精确Top20仅作描述，未参与选权或任何闸门",
    }


def _canonical_hashes(draws: Sequence[SSQDraw]) -> tuple[str, str]:
    data_rows = [
        {
            "issue": draw.issue,
            "date": draw.draw_date,
            "red": list(draw.red),
            "blue": draw.blue,
        }
        for draw in draws
    ]
    source_rows = [
        {
            "issue": draw.issue,
            "sourceUrl": draw.source_url,
            "rawHash": draw.raw_hash,
        }
        for draw in draws
    ]

    def digest(rows: object) -> str:
        serialized = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return digest(data_rows), digest(source_rows)


def _boundary(draws: Sequence[SSQDraw], start: int, end: int) -> dict[str, object]:
    return {
        "startIndex": start,
        "endIndexExclusive": end,
        "draws": end - start,
        "firstIssue": draws[start].issue if start < end else None,
        "firstDate": draws[start].draw_date if start < end else None,
        "lastIssue": draws[end - 1].issue if start < end else None,
        "lastDate": draws[end - 1].draw_date if start < end else None,
    }


def evaluate_ssq_weight_training(
    draws: Sequence[SSQDraw], *, csv_sha256: str | None = None
) -> dict[str, object]:
    """执行唯一固定 v2 挑战；Validation/Frozen 只会条件式开封一次。

    示例：
        ``evaluate_ssq_weight_training(load_official_history_csv(path))``
    """

    minimum_draws = EVALUATION_WARMUP_DRAWS + VALIDATION_DRAWS + FROZEN_TEST_DRAWS + 1
    if len(draws) < minimum_draws:
        raise ValueError(f"双色球v2至少需要{minimum_draws}期，才能保留非空Search")
    if any(
        int(draws[index - 1].issue) >= int(draws[index].issue)
        for index in range(1, len(draws))
    ):
        raise ValueError("双色球v2输入必须按期号严格升序且无重复")

    frozen_start = len(draws) - FROZEN_TEST_DRAWS
    validation_start = frozen_start - VALIDATION_DRAWS
    search_start = EVALUATION_WARMUP_DRAWS
    search_end = validation_start
    data_hash, source_hash = _canonical_hashes(draws)
    state = StaticExpertState()
    for index in range(EVALUATION_WARMUP_DRAWS):
        state.update(draws[index])
    search_observations = [
        state.observe_prior(draws[index]) for index in range(search_start, search_end)
    ]
    summaries = [
        _candidate_summary(search_observations, red_weights, blue_weights)
        for red_weights in RED_WEIGHT_GRID
        for blue_weights in BLUE_WEIGHT_GRID
    ]
    winner = _select_search_winner(summaries)
    base_report: dict[str, object] = {
        "schemaVersion": 1,
        "model": MODEL_NAME,
        "protocolSha256": protocol_sha256(),
        "v1ProtocolSha256": v1_protocol_sha256(),
        "dataSha256": data_hash,
        "sourceSha256": source_hash,
        "csvFileSha256": csv_sha256,
        "researchOnly": True,
        "recommendationEnabled": False,
        "productionActivation": False,
        "formalCandidates": [],
        "protocol": PROTOCOL,
        "split": {
            "warmup": _boundary(draws, 0, EVALUATION_WARMUP_DRAWS),
            "search": _boundary(draws, search_start, search_end),
            "validation": _boundary(draws, validation_start, frozen_start),
            "frozenTest": _boundary(draws, frozen_start, len(draws)),
        },
        "candidateCount": len(summaries),
        "uniformBaselines": {
            "redLogLossPerBall": RED_UNIFORM_LOG_LOSS,
            "redBrierPerBall": RED_UNIFORM_BRIER,
            "blueLogLoss": BLUE_UNIFORM_LOG_LOSS,
            "blueBrier": BLUE_UNIFORM_BRIER,
            "redTop6MeanHits": RED_COVERAGE_BASELINE,
            "blueTop1HitRate": BLUE_COVERAGE_BASELINE,
            "exactTop20PerDraw": EXACT_TOP20_UNIFORM_PROBABILITY,
        },
        "searchCandidates": summaries,
        "validationOpened": False,
        "frozenTestOpened": False,
        "validation": {"opened": False},
        "frozenTest": {"opened": False},
        "retryCount": 0,
        "postFrozenTuning": False,
    }
    if winner is None:
        return {
            **base_report,
            "selectedWeights": {
                "red": list(RED_UNIFORM_WEIGHTS),
                "blue": list(BLUE_UNIFORM_WEIGHTS),
                "pairModifier": PAIR_MODIFIER_WEIGHT,
            },
            "selection": "uniform_abstention",
            "selectedSearchMetrics": None,
            "searchExactTop20Descriptive": None,
            "decision": "uniform_abstention_no_eligible_search_candidate",
            "failureDisposition": (
                "Search无合格非均匀候选；按协议放弃，Validation与Frozen均未开封"
            ),
        }

    red_weights = tuple(
        float(value) for value in cast(Sequence[float], winner["redWeights"])
    )
    blue_weights = tuple(
        float(value) for value in cast(Sequence[float], winner["blueWeights"])
    )
    selected_search_top20 = _selected_search_top20(
        search_observations, red_weights, blue_weights
    )
    validation = _segment_report(
        draws,
        state,
        validation_start,
        frozen_start,
        red_weights,
        blue_weights,
    )
    report = {
        **base_report,
        "selectedWeights": {
            "red": list(red_weights),
            "blue": list(blue_weights),
            "pairModifier": PAIR_MODIFIER_WEIGHT,
        },
        "selection": "unique_non_uniform_search_winner",
        "selectedSearchMetrics": winner["metrics"],
        "searchExactTop20Descriptive": selected_search_top20,
        "validationOpened": True,
        "validation": validation,
    }
    if not bool(validation["passed"]):
        return {
            **report,
            "decision": "research_rejected_validation_failure",
            "failureDisposition": (
                "固定赢家Validation任一完整块或汇总失败；Frozen保持未开封，禁止重试"
            ),
        }

    frozen = _segment_report(
        draws,
        state,
        frozen_start,
        len(draws),
        red_weights,
        blue_weights,
    )
    report = {
        **report,
        "frozenTestOpened": True,
        "frozenTest": frozen,
    }
    if not bool(frozen["passed"]):
        return {
            **report,
            "decision": "research_rejected_frozen_failure",
            "failureDisposition": (
                "固定赢家Frozen任一完整块或汇总失败；挑战终止，禁止调参或重试"
            ),
        }
    return {
        **report,
        "decision": "research_challenge_passed_no_activation",
        "failureDisposition": (
            "六项门槛全部通过，但结果仍仅供研究；正式候选为空且不自动生产启用"
        ),
    }


__all__ = [
    "BLOCK_SIZE",
    "BLUE_WEIGHT_GRID",
    "FROZEN_TEST_DRAWS",
    "GRID_STEP",
    "MODEL_NAME",
    "PROTOCOL",
    "RED_WEIGHT_GRID",
    "VALIDATION_DRAWS",
    "evaluate_ssq_weight_training",
    "protocol_sha256",
]
