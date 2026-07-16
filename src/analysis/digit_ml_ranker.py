# -*- coding: utf-8 -*-
"""数字彩轻量二分类排序器。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_candidates import (
    DigitCandidateConfig,
    _build_score_context,
    _digit_universe,
    _effective_config,
    _model_component_scores,
    _passes_cached_filters,
    _structure_constraint_penalty,
)
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_statistics import (
    DigitStatisticsResult,
    analyze_digit_history,
    classify_digit_shape,
)
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitMlRanker:
    """已训练的 sklearn 排序模型及训练证据。"""

    model: Any
    training_targets: int
    training_samples: int
    feature_count: int


def _feature_vector(
    numbers: Sequence[int],
    context: dict[str, Any],
) -> list[float]:
    return [float(value) for value in _model_component_scores(numbers, context)[:-2]]


def _sample_negatives(
    rule: LotteryRule,
    config: DigitCandidateConfig,
    positive_text: str,
    latest_text: str,
    count: int,
    rng: random.Random,
    context: dict[str, Any],
) -> list[list[int]]:
    output: list[list[int]] = []
    seen: set[str] = set()
    attempts = 0
    while len(output) < count and attempts < count * 500:
        attempts += 1
        numbers = [rng.randrange(10) for _ in range(rule.draw_count)]
        text = "".join(str(value) for value in numbers)
        if text in seen or text == positive_text:
            continue
        if config.exclude_latest and text == latest_text:
            continue
        if not _passes_cached_filters(
            sum(numbers),
            max(numbers) - min(numbers),
            classify_digit_shape(numbers),
            config,
        ):
            continue
        if np.isinf(_structure_constraint_penalty(numbers, context, config)):
            continue
        seen.add(text)
        output.append(numbers)
    return output


def train_digit_ranker(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: DigitCandidateConfig | None = None,
    *,
    min_train_size: int = 30,
    training_periods: int = 60,
    negative_samples: int = 9,
    seed: int = 20260716,
) -> DigitMlRanker | None:
    """使用逐期历史特征训练逻辑回归排序器，不读取目标期之后的数据。"""

    if min_train_size <= 0 or training_periods <= 0 or negative_samples <= 0:
        raise ValueError("训练窗口和负样本数量必须为正整数")
    normalized = normalize_digit_dataframe(history, rule)
    chronological = sort_digit_dataframe_by_issue(normalized, ascending=True)
    if len(chronological) <= min_train_size:
        return None
    start = max(min_train_size, len(chronological) - training_periods)
    target_indexes = range(start, len(chronological))
    features: list[list[float]] = []
    labels: list[int] = []
    training_target_count = 0
    rng = random.Random(seed)
    effective = _effective_config(rule, config or DigitCandidateConfig())
    for target_index in target_indexes:
        train = chronological.iloc[:target_index]
        target = chronological.iloc[target_index]
        stats = analyze_digit_history(
            train,
            rule,
            frequency_windows=effective.frequency_windows,
        )
        context = _build_score_context(stats, effective)
        positive = [int(target[column]) for column in rule.number_columns]
        positive_text = "".join(str(value) for value in positive)
        latest_text = "".join(str(value) for value in stats.latest_numbers)
        positive_is_eligible = _passes_cached_filters(
            sum(positive),
            max(positive) - min(positive),
            classify_digit_shape(positive),
            effective,
        ) and not np.isinf(_structure_constraint_penalty(positive, context, effective))
        if not positive_is_eligible or (
            effective.exclude_latest and positive_text == latest_text
        ):
            continue
        features.append(_feature_vector(positive, context))
        labels.append(1)
        training_target_count += 1
        for negative in _sample_negatives(
            rule,
            effective,
            positive_text,
            latest_text,
            negative_samples,
            rng,
            context,
        ):
            features.append(_feature_vector(negative, context))
            labels.append(0)
    if not features or len(set(labels)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=seed,
            solver="liblinear",
        ),
    )
    model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=int))
    return DigitMlRanker(
        model=model,
        training_targets=training_target_count,
        training_samples=len(features),
        feature_count=len(features[0]),
    )


def score_digit_ranker(
    ranker: DigitMlRanker | None,
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitCandidateConfig | None = None,
) -> dict[str, float]:
    """对完整过滤空间输出二分类模型排序分；分数不视为真实开奖概率。"""

    if ranker is None:
        return {}
    effective = _effective_config(rule, config or DigitCandidateConfig())
    context = _build_score_context(stats, effective)
    latest_text = "".join(str(value) for value in stats.latest_numbers)
    texts: list[str] = []
    features: list[list[float]] = []
    for values, text, sum_value, span, shape in _digit_universe(rule.draw_count):
        if effective.exclude_latest and text == latest_text:
            continue
        if not _passes_cached_filters(sum_value, span, shape, effective):
            continue
        if np.isinf(_structure_constraint_penalty(values, context, effective)):
            continue
        texts.append(text)
        features.append(_feature_vector(values, context))
    if not features:
        return {}
    probabilities = ranker.model.predict_proba(np.asarray(features, dtype=float))[:, 1]
    return {text: float(score) for text, score in zip(texts, probabilities)}


__all__ = ["DigitMlRanker", "score_digit_ranker", "train_digit_ranker"]
