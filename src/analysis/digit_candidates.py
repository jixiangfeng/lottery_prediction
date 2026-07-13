# -*- coding: utf-8 -*-
"""数字彩候选生成器。

基于位置频率与当前遗漏生成福彩3D、排列三、排列五候选，并用和值、跨度、形态做过滤。
它是统计辅助工具，不保证命中。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from src.analysis.digit_statistics import DigitStatisticsResult, classify_digit_shape
from src.lotteries.base import LotteryRule, validate_numbers


@dataclass(frozen=True)
class DigitCandidateConfig:
    """数字彩候选生成配置。"""

    count: int = 10
    sum_min: int | None = None
    sum_max: int | None = None
    span_min: int | None = None
    span_max: int | None = None
    allowed_shapes: tuple[str, ...] | None = None
    top_digits_per_position: int = 6
    frequency_weight: float = 0.7
    omission_weight: float = 0.3
    random_weight: float = 0.15
    exclude_latest: bool = True


@dataclass(frozen=True)
class DigitCandidate:
    """单个数字彩候选。"""

    numbers: list[int]
    text: str
    sum_value: int
    span: int
    shape: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "numbers": self.numbers,
            "sum": self.sum_value,
            "span": self.span,
            "shape": self.shape,
            "score": self.score,
        }


@dataclass(frozen=True)
class DigitCandidateResult:
    """数字彩候选生成结果。"""

    rule_code: str
    display_name: str
    candidates: list[DigitCandidate]
    config: DigitCandidateConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleCode": self.rule_code,
            "displayName": self.display_name,
            "config": {
                "count": self.config.count,
                "sumMin": self.config.sum_min,
                "sumMax": self.config.sum_max,
                "spanMin": self.config.span_min,
                "spanMax": self.config.span_max,
                "allowedShapes": list(self.config.allowed_shapes) if self.config.allowed_shapes else None,
                "excludeLatest": self.config.exclude_latest,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _digit_score(stats: DigitStatisticsResult, position: str, digit: int, config: DigitCandidateConfig, rng: random.Random) -> float:
    frequency = stats.position_frequency[position]
    max_freq = max(frequency.values()) if frequency else 1
    freq_score = frequency.get(digit, 0) / max_freq if max_freq else 0.0
    omission = stats.current_omission.get(position, {})
    max_omission = max(omission.values()) if omission else 1
    omission_score = omission.get(digit, 0) / max_omission if max_omission else 0.0
    return (
        config.frequency_weight * freq_score
        + config.omission_weight * omission_score
        + config.random_weight * rng.random()
    )


def _ranked_digits(stats: DigitStatisticsResult, rule: LotteryRule, position: str, index: int, config: DigitCandidateConfig, rng: random.Random) -> list[tuple[int, float]]:
    spec = rule.ball_specs[index]
    scored = [
        (digit, _digit_score(stats, position, digit, config, rng))
        for digit in range(spec.min_number, spec.max_number + 1)
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[: max(1, config.top_digits_per_position)]


def _effective_config(rule: LotteryRule, config: DigitCandidateConfig) -> DigitCandidateConfig:
    """补齐更适合实战候选的默认过滤条件。"""

    if rule.draw_count == 3:
        return DigitCandidateConfig(
            count=config.count,
            sum_min=6 if config.sum_min is None else config.sum_min,
            sum_max=21 if config.sum_max is None else config.sum_max,
            span_min=2 if config.span_min is None else config.span_min,
            span_max=9 if config.span_max is None else config.span_max,
            allowed_shapes=("组三", "组六") if config.allowed_shapes is None else config.allowed_shapes,
            top_digits_per_position=config.top_digits_per_position,
            frequency_weight=config.frequency_weight,
            omission_weight=config.omission_weight,
            random_weight=config.random_weight,
            exclude_latest=config.exclude_latest,
        )
    if rule.draw_count == 5:
        return DigitCandidateConfig(
            count=config.count,
            sum_min=10 if config.sum_min is None else config.sum_min,
            sum_max=35 if config.sum_max is None else config.sum_max,
            span_min=3 if config.span_min is None else config.span_min,
            span_max=9 if config.span_max is None else config.span_max,
            allowed_shapes=("全不同", "二一一一", "二二一") if config.allowed_shapes is None else config.allowed_shapes,
            top_digits_per_position=config.top_digits_per_position,
            frequency_weight=config.frequency_weight,
            omission_weight=config.omission_weight,
            random_weight=config.random_weight,
            exclude_latest=config.exclude_latest,
        )
    return config


def _passes_filters(numbers: Sequence[int], config: DigitCandidateConfig) -> bool:
    sum_value = sum(numbers)
    span = max(numbers) - min(numbers)
    shape = classify_digit_shape(numbers)
    if config.sum_min is not None and sum_value < config.sum_min:
        return False
    if config.sum_max is not None and sum_value > config.sum_max:
        return False
    if config.span_min is not None and span < config.span_min:
        return False
    if config.span_max is not None and span > config.span_max:
        return False
    if config.allowed_shapes is not None and shape not in config.allowed_shapes:
        return False
    return True


def _candidate_text(numbers: Sequence[int]) -> str:
    return "".join(str(int(number)) for number in numbers)


def _make_candidate(numbers: list[int], score: float) -> DigitCandidate:
    return DigitCandidate(
        numbers=numbers,
        text=_candidate_text(numbers),
        sum_value=sum(numbers),
        span=max(numbers) - min(numbers),
        shape=classify_digit_shape(numbers),
        score=round(score, 4),
    )


def _ranked_dict(ranked: list[tuple[int, float]]) -> dict[int, float]:
    """将位置候选分数字典化，便于构造确定性候选。"""

    return {digit: weight for digit, weight in ranked}


def _group3_seed_candidates(
    ranked_by_position: list[list[tuple[int, float]]],
    config: DigitCandidateConfig,
) -> list[DigitCandidate]:
    """为三位数字彩补充组三种子候选。"""

    if config.allowed_shapes is not None and "组三" not in config.allowed_shapes:
        return []
    if len(ranked_by_position) != 3:
        return []
    score_by_position = [_ranked_dict(ranked) for ranked in ranked_by_position]
    top_digits = sorted({digit for ranked in ranked_by_position for digit, _ in ranked})
    candidates: list[DigitCandidate] = []
    seen: set[str] = set()
    for repeated_digit in top_digits:
        for single_digit in top_digits:
            if single_digit == repeated_digit:
                continue
            for pair_positions in ((0, 1), (0, 2), (1, 2)):
                numbers = [single_digit, single_digit, single_digit]
                for position in pair_positions:
                    numbers[position] = repeated_digit
                text = _candidate_text(numbers)
                if text in seen or not _passes_filters(numbers, config):
                    continue
                score = sum(score_by_position[index].get(number, 0.0) for index, number in enumerate(numbers))
                seen.add(text)
                candidates.append(_make_candidate(numbers, score))
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.text))


def _group3_quota(config: DigitCandidateConfig) -> int | None:
    """三位数字彩组三推荐上限。

    理论形态概率约为：组六 72%、组三 27%、豹子 1%。默认不出豹子，
    因此最终候选应以组六为主，组三只做少量防守。
    如果用户显式只允许组三，则不做上限限制。
    """

    if config.allowed_shapes is not None and "组六" not in config.allowed_shapes:
        return None
    if config.allowed_shapes is not None and "组三" not in config.allowed_shapes:
        return 0
    return max(1, int(config.count * 0.3))


def _would_exceed_group3_quota(candidate: DigitCandidate, candidates: Sequence[DigitCandidate], quota: int | None) -> bool:
    if quota is None or candidate.shape != "组三":
        return False
    return sum(1 for item in candidates if item.shape == "组三") >= quota


def generate_digit_candidates(
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    *,
    config: DigitCandidateConfig | None = None,
    seed: int | None = None,
) -> DigitCandidateResult:
    """生成数字彩候选号码。"""

    if rule.category != "digit":
        raise ValueError(f"数字彩候选生成不适用于玩法：{rule.display_name}")
    config = _effective_config(rule, config or DigitCandidateConfig())
    rng = random.Random(seed if seed is not None else f"{rule.code}:{stats.latest_issue}")
    ranked_by_position = [
        _ranked_digits(stats, rule, position, index, config, rng)
        for index, position in enumerate(rule.number_columns)
    ]
    candidates: list[DigitCandidate] = []
    seen: set[str] = set()
    latest_text_for_exclusion = _candidate_text(stats.latest_numbers) if stats.latest_numbers else None
    if not config.exclude_latest and stats.latest_numbers and _passes_filters(stats.latest_numbers, config):
        latest_candidate = _make_candidate(list(stats.latest_numbers), 0.0)
        seen.add(latest_candidate.text)
        candidates.append(latest_candidate)

    # 三位数字彩保留少量确定性组三候选，但仍按理论概率以组六为主。
    group3_quota = _group3_quota(config) if rule.draw_count == 3 else None
    if rule.draw_count == 3:
        for candidate in _group3_seed_candidates(ranked_by_position, config):
            if _would_exceed_group3_quota(candidate, candidates, group3_quota):
                break
            if config.exclude_latest and candidate.text == latest_text_for_exclusion:
                continue
            if candidate.text in seen:
                continue
            seen.add(candidate.text)
            candidates.append(candidate)

    attempts = 0
    max_attempts = max(200, config.count * 200)
    while len(candidates) < config.count and attempts < max_attempts:
        attempts += 1
        numbers: list[int] = []
        score = 0.0
        for ranked in ranked_by_position:
            digits = [digit for digit, _ in ranked]
            weights = [max(weight, 0.0001) for _, weight in ranked]
            chosen = rng.choices(digits, weights=weights, k=1)[0]
            numbers.append(chosen)
            score += dict(ranked)[chosen]
        validate_numbers(rule, numbers)
        text = _candidate_text(numbers)
        candidate = _make_candidate(numbers, score)
        if config.exclude_latest and text == latest_text_for_exclusion:
            continue
        if text in seen or not _passes_filters(numbers, config):
            continue
        if _would_exceed_group3_quota(candidate, candidates, group3_quota):
            continue
        seen.add(text)
        candidates.append(candidate)

    # 如果过滤条件过严导致不足，用全池随机兜底，但仍遵守过滤条件。
    while len(candidates) < config.count and attempts < max_attempts * 2:
        attempts += 1
        numbers = [rng.randint(spec.min_number, spec.max_number) for spec in rule.ball_specs]
        text = _candidate_text(numbers)
        candidate = _make_candidate(numbers, 0.0)
        if config.exclude_latest and text == latest_text_for_exclusion:
            continue
        if text in seen or not _passes_filters(numbers, config):
            continue
        if _would_exceed_group3_quota(candidate, candidates, group3_quota):
            continue
        seen.add(text)
        candidates.append(candidate)
    return DigitCandidateResult(rule_code=rule.code, display_name=rule.display_name, candidates=candidates, config=config)
