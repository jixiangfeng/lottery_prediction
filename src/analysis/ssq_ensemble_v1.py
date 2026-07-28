# -*- coding: utf-8 -*-
"""双色球固定基线集成 v1：严格前序在线回放与研究候选。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from src.analysis.ssq_diversified_portfolio_v2 import (
    build_diversified_portfolio_v2,
)
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)

MODEL_NAME = "ssq_ensemble_v1"
RED_HALF_LIVES = (30, 120)
PAIR_HALF_LIFE = 120
BLUE_HALF_LIFE = 60
LAPLACE_ALPHA = 1.0
EXPONENTIAL_WEIGHTS_ETA = 0.25
PAIR_MODIFIER_WEIGHT = 0.20
PAIR_LOG_RATIO_BOUND = 1.0
EVALUATION_WARMUP_DRAWS = 120
EVALUATION_BLOCK_SIZE = 100
TOP_K = 20
BEAM_WIDTH = 256
RESEARCH_RED_AUDIT_COUNT = 32
EPSILON = 1e-12
RED_UNIFORM_PROBABILITY = 6.0 / 33.0
BLUE_UNIFORM_PROBABILITY = 1.0 / 16.0
EXACT_TOP20_UNIFORM_PROBABILITY = TOP_K / (math.comb(33, 6) * 16)
MetricValue = float | int | bool
MetricReport = dict[str, MetricValue]


PREREGISTRATION_PROTOCOL: dict[str, object] = {
    "model": MODEL_NAME,
    "purpose": "仅用于双色球历史数据的固定基线研究，不构成预测或投注建议",
    "data": "只接受经单一福彩官网来源证据链对账的完整历史",
    "walkForward": "每期先用严格前序状态预测并评分，再用该期开奖更新全部专家",
    "experts": {
        "red": [
            "永久保留的均匀基线专家",
            "Laplace平滑、半衰期30期的红球边际EWMA固定基线专家",
            "Laplace平滑、半衰期120期的红球边际EWMA固定基线专家",
            "Laplace平滑、半衰期120期的528对红球共现有界修饰器",
        ],
        "blue": [
            "永久保留的均匀基线专家",
            "Laplace平滑、半衰期60期的蓝球EWMA固定基线专家",
        ],
    },
    "aggregation": {
        "method": "固定eta的指数权重在线聚合",
        "eta": EXPONENTIAL_WEIGHTS_ETA,
        "update": "只在当期开奖完成评分后更新",
    },
    "tickets": {
        "method": "固定宽度束搜索生成6-of-33红球，再与16个蓝球联合排序",
        "beamWidth": BEAM_WIDTH,
        "topK": TOP_K,
        "tieBreak": "红球元组、蓝球按字典序升序",
        "predictionClaim": False,
    },
    "evaluation": {
        "warmupDraws": EVALUATION_WARMUP_DRAWS,
        "blockSize": EVALUATION_BLOCK_SIZE,
        "blocks": "从固定起点开始的全部不重叠完整块；尾部不完整块排除",
        "primary": [
            "红球Top6平均命中数，相对均匀期望36/33",
            "蓝球Top1命中率，相对均匀期望1/16",
            "红球逐球Bernoulli LogLoss与Brier",
            "蓝球多分类LogLoss与Brier",
        ],
        "descriptiveOnly": (
            "精确Top20整票命中仅描述，不作为主要证据；均匀期望为" "20/(C(33,6)*16)"
        ),
    },
    "hardGate": (
        "必须有完整块；每个完整块的四项合并proper score指数不劣于均匀基线，"
        "且红蓝覆盖合并指数不低于均匀基线；汇总层四项proper score与红蓝覆盖"
        "也必须分别不劣于均匀基线。失败统一输出uniform_abstain。"
    ),
    "fixed": {
        "parameterCliOverrides": False,
        "gridSearch": False,
        "featureSearch": False,
        "modelRetries": 0,
    },
}


def protocol_sha256() -> str:
    """返回固定预注册协议的稳定摘要。"""

    serialized = json.dumps(
        PREREGISTRATION_PROTOCOL,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _decay(half_life: int) -> float:
    return math.exp(math.log(0.5) / half_life)


@dataclass
class RedMarginalExpert:
    """Laplace 平滑的红球边际 EWMA 固定基线专家。"""

    half_life: int
    counts: list[float] = field(default_factory=lambda: [0.0] * 33)
    effective_draws: float = 0.0

    def probabilities(self) -> list[float]:
        denominator = 6.0 * self.effective_draws + 33.0 * LAPLACE_ALPHA
        return [6.0 * (count + LAPLACE_ALPHA) / denominator for count in self.counts]

    def update(self, red: Sequence[int]) -> None:
        decay = _decay(self.half_life)
        selected = set(red)
        self.counts = [
            decay * count + (1.0 if ball in selected else 0.0)
            for ball, count in enumerate(self.counts, start=1)
        ]
        self.effective_draws = decay * self.effective_draws + 1.0


@dataclass
class BlueMarginalExpert:
    """Laplace 平滑的蓝球多分类 EWMA 固定基线专家。"""

    half_life: int
    counts: list[float] = field(default_factory=lambda: [0.0] * 16)
    effective_draws: float = 0.0

    def probabilities(self) -> list[float]:
        denominator = self.effective_draws + 16.0 * LAPLACE_ALPHA
        return [(count + LAPLACE_ALPHA) / denominator for count in self.counts]

    def update(self, blue: int) -> None:
        decay = _decay(self.half_life)
        self.counts = [
            decay * count + (1.0 if ball == blue else 0.0)
            for ball, count in enumerate(self.counts, start=1)
        ]
        self.effective_draws = decay * self.effective_draws + 1.0


def _all_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right) for left in range(1, 34) for right in range(left + 1, 34)
    )


RED_PAIRS = _all_pairs()
PAIR_INDEX = {pair: index for index, pair in enumerate(RED_PAIRS)}


@dataclass
class RedPairExpert:
    """528 对红球共现的 Laplace 平滑 EWMA 有界修饰器。"""

    counts: list[float] = field(default_factory=lambda: [0.0] * len(RED_PAIRS))
    effective_draws: float = 0.0

    def modifiers(self) -> list[float]:
        denominator = 15.0 * self.effective_draws + len(RED_PAIRS) * LAPLACE_ALPHA
        uniform_pair_probability = 15.0 / len(RED_PAIRS)
        modifiers: list[float] = []
        for count in self.counts:
            probability = 15.0 * (count + LAPLACE_ALPHA) / denominator
            log_ratio = math.log(max(probability, EPSILON) / uniform_pair_probability)
            modifiers.append(
                max(-PAIR_LOG_RATIO_BOUND, min(PAIR_LOG_RATIO_BOUND, log_ratio))
            )
        return modifiers

    def update(self, red: Sequence[int]) -> None:
        decay = _decay(PAIR_HALF_LIFE)
        selected_pairs = {
            (red[left], red[right])
            for left in range(len(red))
            for right in range(left + 1, len(red))
        }
        self.counts = [
            decay * count + (1.0 if pair in selected_pairs else 0.0)
            for pair, count in zip(RED_PAIRS, self.counts)
        ]
        self.effective_draws = decay * self.effective_draws + 1.0


def _normalized_weights(log_weights: Sequence[float]) -> list[float]:
    maximum = max(log_weights)
    exponentials = [math.exp(value - maximum) for value in log_weights]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _red_log_loss(probabilities: Sequence[float], red: Sequence[int]) -> float:
    selected = set(red)
    total = 0.0
    for ball, probability in enumerate(probabilities, start=1):
        clipped = min(1.0 - EPSILON, max(EPSILON, probability))
        total -= math.log(clipped if ball in selected else 1.0 - clipped)
    return total / 33.0


def _red_brier(probabilities: Sequence[float], red: Sequence[int]) -> float:
    selected = set(red)
    return (
        sum(
            (probability - (1.0 if ball in selected else 0.0)) ** 2
            for ball, probability in enumerate(probabilities, start=1)
        )
        / 33.0
    )


def _blue_log_loss(probabilities: Sequence[float], blue: int) -> float:
    return -math.log(max(EPSILON, probabilities[blue - 1]))


def _blue_brier(probabilities: Sequence[float], blue: int) -> float:
    return sum(
        (probability - (1.0 if ball == blue else 0.0)) ** 2
        for ball, probability in enumerate(probabilities, start=1)
    )


def _mixture(
    probabilities: Sequence[Sequence[float]], weights: Sequence[float]
) -> list[float]:
    return [
        sum(weight * expert[index] for weight, expert in zip(weights, probabilities))
        for index in range(len(probabilities[0]))
    ]


@dataclass
class FixedEnsembleState:
    """仅包含固定专家与严格在线权重的可变状态。"""

    red_experts: tuple[RedMarginalExpert, ...] = field(
        default_factory=lambda: tuple(
            RedMarginalExpert(half_life) for half_life in RED_HALF_LIVES
        )
    )
    blue_expert: BlueMarginalExpert = field(
        default_factory=lambda: BlueMarginalExpert(BLUE_HALF_LIFE)
    )
    pair_expert: RedPairExpert = field(default_factory=RedPairExpert)
    red_log_weights: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    blue_log_weights: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def predict(self) -> tuple[list[float], list[float], list[float]]:
        """读取当前严格前序状态，不做任何更新。"""

        red_expert_probabilities = [
            [RED_UNIFORM_PROBABILITY] * 33,
            *[expert.probabilities() for expert in self.red_experts],
        ]
        blue_expert_probabilities = [
            [BLUE_UNIFORM_PROBABILITY] * 16,
            self.blue_expert.probabilities(),
        ]
        red_probabilities = _mixture(
            red_expert_probabilities, _normalized_weights(self.red_log_weights)
        )
        blue_probabilities = _mixture(
            blue_expert_probabilities, _normalized_weights(self.blue_log_weights)
        )
        return red_probabilities, blue_probabilities, self.pair_expert.modifiers()

    def score_then_update(self, draw: SSQDraw) -> None:
        """用当前期评分专家权重，随后才更新专家统计。"""

        red_expert_probabilities = [
            [RED_UNIFORM_PROBABILITY] * 33,
            *[expert.probabilities() for expert in self.red_experts],
        ]
        blue_expert_probabilities = [
            [BLUE_UNIFORM_PROBABILITY] * 16,
            self.blue_expert.probabilities(),
        ]
        for index, probabilities in enumerate(red_expert_probabilities):
            self.red_log_weights[index] -= EXPONENTIAL_WEIGHTS_ETA * _red_log_loss(
                probabilities, draw.red
            )
        for index, probabilities in enumerate(blue_expert_probabilities):
            self.blue_log_weights[index] -= EXPONENTIAL_WEIGHTS_ETA * _blue_log_loss(
                probabilities, draw.blue
            )
        for expert in self.red_experts:
            expert.update(draw.red)
        self.blue_expert.update(draw.blue)
        self.pair_expert.update(draw.red)


def red_top6(probabilities: Sequence[float]) -> tuple[int, ...]:
    """按概率降序、球号升序选择固定 Top6。"""

    ranked = sorted(range(1, 34), key=lambda ball: (-probabilities[ball - 1], ball))
    return tuple(sorted(ranked[:6]))


def blue_top1(probabilities: Sequence[float]) -> int:
    """按概率降序、球号升序选择固定 Top1。"""

    return min(range(1, 17), key=lambda ball: (-probabilities[ball - 1], ball))


def _red_ticket_score(
    red: Sequence[int],
    logits: Sequence[float],
    pair_modifiers: Sequence[float],
) -> float:
    marginal_score = sum(logits[ball - 1] for ball in red)
    pair_score = sum(
        pair_modifiers[PAIR_INDEX[(red[left], red[right])]]
        for left in range(len(red))
        for right in range(left + 1, len(red))
    )
    return marginal_score + PAIR_MODIFIER_WEIGHT * pair_score / 15.0


def beam_red_combinations(
    red_probabilities: Sequence[float], pair_modifiers: Sequence[float]
) -> list[tuple[tuple[int, ...], float]]:
    """以固定束宽搜索高分有效红球组合。"""

    logits = [
        math.log(max(EPSILON, probability)) - math.log(max(EPSILON, 1.0 - probability))
        for probability in red_probabilities
    ]
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for depth in range(6):
        candidates: list[tuple[tuple[int, ...], float]] = []
        for prefix, _ in beam:
            start = prefix[-1] + 1 if prefix else 1
            maximum = 33 - (5 - depth)
            for ball in range(start, maximum + 1):
                red = (*prefix, ball)
                candidates.append((red, _red_ticket_score(red, logits, pair_modifiers)))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        beam = candidates[:BEAM_WIDTH]
    return beam


def generate_research_tickets(
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    pair_modifiers: Sequence[float],
) -> list[dict[str, object]]:
    """确定性生成 Top20 研究票；分数不是开奖概率。"""

    red_candidates = beam_red_combinations(red_probabilities, pair_modifiers)
    tickets: list[tuple[float, tuple[int, ...], int]] = []
    for red, red_score in red_candidates:
        for blue, probability in enumerate(blue_probabilities, start=1):
            tickets.append((red_score + math.log(max(probability, EPSILON)), red, blue))
    tickets.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "red": [f"{ball:02d}" for ball in red],
            "blue": f"{blue:02d}",
            "rankingScore": score,
            "predictionClaim": False,
        }
        for score, red, blue in tickets[:TOP_K]
    ]


@dataclass
class MetricAccumulator:
    """一个固定评估块的指标累加器。"""

    start_index: int
    end_index: int
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
        draw: SSQDraw,
        red_probabilities: Sequence[float],
        blue_probabilities: Sequence[float],
        tickets: Sequence[dict[str, object]],
    ) -> None:
        selected_red = red_top6(red_probabilities)
        selected_blue = blue_top1(blue_probabilities)
        self.count += 1
        self.red_hits += len(set(selected_red).intersection(draw.red))
        self.blue_hits += float(selected_blue == draw.blue)
        self.red_log_loss += _red_log_loss(red_probabilities, draw.red)
        self.red_brier += _red_brier(red_probabilities, draw.red)
        self.blue_log_loss += _blue_log_loss(blue_probabilities, draw.blue)
        self.blue_brier += _blue_brier(blue_probabilities, draw.blue)
        actual_red = [f"{ball:02d}" for ball in draw.red]
        self.exact_top20_hits += int(
            any(
                ticket["red"] == actual_red and ticket["blue"] == f"{draw.blue:02d}"
                for ticket in tickets
            )
        )

    def report(self) -> MetricReport:
        if self.count <= 0:
            raise ValueError("空评估块不能生成指标")
        metrics: MetricReport = {
            "draws": self.count,
            "redTop6MeanHits": self.red_hits / self.count,
            "blueTop1HitRate": self.blue_hits / self.count,
            "redLogLossPerBall": self.red_log_loss / self.count,
            "redBrierPerBall": self.red_brier / self.count,
            "blueLogLoss": self.blue_log_loss / self.count,
            "blueBrier": self.blue_brier / self.count,
            "exactTop20Hits": self.exact_top20_hits,
            "exactTop20ExpectedHitsUniform": (
                self.count * EXACT_TOP20_UNIFORM_PROBABILITY
            ),
        }
        proper_index = _proper_score_index(metrics)
        coverage_index = _coverage_index(metrics)
        return {
            "startIndex": self.start_index,
            "endIndexExclusive": self.end_index,
            **metrics,
            "properScoreIndexVsUniform": proper_index,
            "coverageIndexVsUniform": coverage_index,
            "stable": proper_index <= 1.0 and coverage_index >= 1.0,
        }


RED_UNIFORM_LOG_LOSS = -(
    RED_UNIFORM_PROBABILITY * math.log(RED_UNIFORM_PROBABILITY)
    + (1.0 - RED_UNIFORM_PROBABILITY) * math.log(1.0 - RED_UNIFORM_PROBABILITY)
)
RED_UNIFORM_BRIER = RED_UNIFORM_PROBABILITY * (1.0 - RED_UNIFORM_PROBABILITY)
BLUE_UNIFORM_LOG_LOSS = math.log(16.0)
BLUE_UNIFORM_BRIER = 15.0 / 16.0


def _proper_score_index(metrics: Mapping[str, MetricValue]) -> float:
    return (
        sum(
            (
                float(metrics["redLogLossPerBall"]) / RED_UNIFORM_LOG_LOSS,
                float(metrics["redBrierPerBall"]) / RED_UNIFORM_BRIER,
                float(metrics["blueLogLoss"]) / BLUE_UNIFORM_LOG_LOSS,
                float(metrics["blueBrier"]) / BLUE_UNIFORM_BRIER,
            )
        )
        / 4.0
    )


def _coverage_index(metrics: Mapping[str, MetricValue]) -> float:
    return (
        float(metrics["redTop6MeanHits"]) / (36.0 / 33.0)
        + float(metrics["blueTop1HitRate"]) / BLUE_UNIFORM_PROBABILITY
    ) / 2.0


def _aggregate_block_reports(blocks: Sequence[MetricReport]) -> MetricReport:
    total_draws = sum(int(block["draws"]) for block in blocks)
    weighted_keys = (
        "redTop6MeanHits",
        "blueTop1HitRate",
        "redLogLossPerBall",
        "redBrierPerBall",
        "blueLogLoss",
        "blueBrier",
    )
    metrics: MetricReport = {"draws": total_draws}
    for key in weighted_keys:
        metrics[key] = (
            sum(float(block[key]) * int(block["draws"]) for block in blocks)
            / total_draws
        )
    metrics["exactTop20Hits"] = sum(int(block["exactTop20Hits"]) for block in blocks)
    metrics["exactTop20ExpectedHitsUniform"] = (
        total_draws * EXACT_TOP20_UNIFORM_PROBABILITY
    )
    metrics["properScoreIndexVsUniform"] = _proper_score_index(metrics)
    metrics["coverageIndexVsUniform"] = _coverage_index(metrics)
    return metrics


def _hard_gate(
    blocks: Sequence[MetricReport], aggregate: MetricReport | None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not blocks or aggregate is None:
        return False, ["没有可用的完整固定评估块"]
    unstable = [index for index, block in enumerate(blocks) if not block["stable"]]
    if unstable:
        reasons.append(f"完整块稳定性失败：块索引 {unstable}")
    proper_checks = {
        "红球LogLoss": (
            float(aggregate["redLogLossPerBall"]),
            RED_UNIFORM_LOG_LOSS,
        ),
        "红球Brier": (
            float(aggregate["redBrierPerBall"]),
            RED_UNIFORM_BRIER,
        ),
        "蓝球LogLoss": (float(aggregate["blueLogLoss"]), BLUE_UNIFORM_LOG_LOSS),
        "蓝球Brier": (float(aggregate["blueBrier"]), BLUE_UNIFORM_BRIER),
    }
    for label, (actual, baseline) in proper_checks.items():
        if actual > baseline:
            reasons.append(f"汇总{label}劣于均匀基线")
    if float(aggregate["redTop6MeanHits"]) < 36.0 / 33.0:
        reasons.append("汇总红球Top6覆盖低于均匀期望36/33")
    if float(aggregate["blueTop1HitRate"]) < BLUE_UNIFORM_PROBABILITY:
        reasons.append("汇总蓝球Top1覆盖低于均匀期望1/16")
    return not reasons, reasons


def walk_forward_prediction_fingerprints(
    draws: Sequence[SSQDraw], stop: int | None = None
) -> list[str]:
    """返回逐期预测摘要，供无未来泄漏回归测试使用。"""

    state = FixedEnsembleState()
    fingerprints: list[str] = []
    limit = len(draws) if stop is None else min(stop, len(draws))
    for draw in draws[:limit]:
        red, blue, pairs = state.predict()
        serialized = json.dumps(
            {"red": red, "blue": blue, "pairs": pairs},
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprints.append(hashlib.sha256(serialized.encode("utf-8")).hexdigest())
        state.score_then_update(draw)
    return fingerprints


def evaluate_ssq_ensemble(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """按固定协议执行一次严格前序回放，不接受任何模型参数覆盖。"""

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("双色球模型输入包含重复期号")
    full_block_count = max(
        0, (len(ordered) - EVALUATION_WARMUP_DRAWS) // EVALUATION_BLOCK_SIZE
    )
    evaluation_end = EVALUATION_WARMUP_DRAWS + full_block_count * EVALUATION_BLOCK_SIZE
    accumulators = [
        MetricAccumulator(start, start + EVALUATION_BLOCK_SIZE)
        for start in range(
            EVALUATION_WARMUP_DRAWS,
            evaluation_end,
            EVALUATION_BLOCK_SIZE,
        )
    ]
    state = FixedEnsembleState()
    for index, draw in enumerate(ordered):
        red_probabilities, blue_probabilities, pair_modifiers = state.predict()
        if EVALUATION_WARMUP_DRAWS <= index < evaluation_end:
            block_index = (index - EVALUATION_WARMUP_DRAWS) // EVALUATION_BLOCK_SIZE
            tickets = generate_research_tickets(
                red_probabilities, blue_probabilities, pair_modifiers
            )
            accumulators[block_index].add(
                draw, red_probabilities, blue_probabilities, tickets
            )
        state.score_then_update(draw)
    block_reports = [accumulator.report() for accumulator in accumulators]
    aggregate = _aggregate_block_reports(block_reports) if block_reports else None
    gate_passed, gate_reasons = _hard_gate(block_reports, aggregate)
    next_red, next_blue, next_pairs = state.predict()
    research_candidates = generate_research_tickets(next_red, next_blue, next_pairs)
    ranked_red_combinations = beam_red_combinations(next_red, next_pairs)
    diversified_portfolio_v2 = build_diversified_portfolio_v2(next_red, next_blue)
    small_compound_8red1blue_v1 = build_small_compound_8red1blue_v1(
        next_red, next_blue, diversified_portfolio_v2
    )
    return {
        "schemaVersion": 1,
        "model": MODEL_NAME,
        "researchOnly": True,
        "protocolSha256": protocol_sha256(),
        "protocol": PREREGISTRATION_PROTOCOL,
        "historyDraws": len(ordered),
        "evaluatedDraws": max(0, evaluation_end - EVALUATION_WARMUP_DRAWS),
        "excludedIncompleteTailDraws": max(0, len(ordered) - evaluation_end),
        "uniformBaselines": {
            "redTop6MeanHits": 36.0 / 33.0,
            "blueTop1HitRate": BLUE_UNIFORM_PROBABILITY,
            "redLogLossPerBall": RED_UNIFORM_LOG_LOSS,
            "redBrierPerBall": RED_UNIFORM_BRIER,
            "blueLogLoss": BLUE_UNIFORM_LOG_LOSS,
            "blueBrier": BLUE_UNIFORM_BRIER,
            "exactTop20TicketProbability": EXACT_TOP20_UNIFORM_PROBABILITY,
            "exactTop20Statement": "20/(C(33,6)*16)，极度稀疏，仅作描述",
        },
        "blocks": block_reports,
        "aggregate": aggregate,
        "hardGatePassed": gate_passed,
        "hardGateReasons": gate_reasons,
        "decision": "fixed_baseline_research" if gate_passed else "uniform_abstain",
        "recommendationEnabled": False,
        "formalCandidates": [],
        "researchCandidates": research_candidates,
        "diversifiedPortfolioV2": diversified_portfolio_v2,
        "smallCompound8Red1BlueV1": small_compound_8red1blue_v1,
        "auditMetadata": {
            "orderedRed6Combinations": [
                {
                    "rank": rank,
                    "red": [f"{ball:02d}" for ball in red],
                    "redScore": red_score,
                }
                for rank, (red, red_score) in enumerate(
                    ranked_red_combinations[:RESEARCH_RED_AUDIT_COUNT], start=1
                )
            ],
            "orderedRed6CombinationCount": RESEARCH_RED_AUDIT_COUNT,
            "researchBlueTop1": blue_top1(next_blue),
            "finalNextProbabilities": {
                "red": next_red,
                "blue": next_blue,
            },
        },
        "finalExpertWeights": {
            "red": _normalized_weights(state.red_log_weights),
            "blue": _normalized_weights(state.blue_log_weights),
        },
    }


def validate_research_candidates(candidates: Iterable[dict[str, object]]) -> None:
    """校验研究票唯一、合法且不包含预测声明。"""

    seen: set[tuple[tuple[str, ...], str]] = set()
    count = 0
    for candidate in candidates:
        red_raw = candidate.get("red")
        blue_raw = candidate.get("blue")
        if not isinstance(red_raw, list) or not isinstance(blue_raw, str):
            raise ValueError("双色球研究票字段类型非法")
        red = tuple(int(value) for value in red_raw)
        blue = int(blue_raw)
        if len(red) != 6 or tuple(sorted(red)) != red or len(set(red)) != 6:
            raise ValueError("双色球研究票红球非法")
        if any(number < 1 or number > 33 for number in red) or not 1 <= blue <= 16:
            raise ValueError("双色球研究票号码越界")
        if candidate.get("predictionClaim") is not False:
            raise ValueError("双色球研究票必须显式 predictionClaim=false")
        key = (tuple(str(value) for value in red_raw), blue_raw)
        if key in seen:
            raise ValueError("双色球研究票重复")
        seen.add(key)
        count += 1
    if count != TOP_K:
        raise ValueError(f"双色球研究票数量必须为固定 TopK={TOP_K}")


__all__ = [
    "BEAM_WIDTH",
    "EVALUATION_BLOCK_SIZE",
    "EVALUATION_WARMUP_DRAWS",
    "MODEL_NAME",
    "PREREGISTRATION_PROTOCOL",
    "TOP_K",
    "evaluate_ssq_ensemble",
    "generate_research_tickets",
    "protocol_sha256",
    "validate_research_candidates",
    "walk_forward_prediction_fingerprints",
]
