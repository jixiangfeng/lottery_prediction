# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 的可复现参数搜索。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    LearnedFeatureConfig,
    build_candidate_features,
    build_history_state,
)
from src.analysis.digit_learned_ranker import (
    DEFAULT_WEIGHTS,
    LearnedRankerParams,
    aggregate_group_candidates,
    probabilities_from_scores,
    rank_candidate_indices,
    score_candidates,
)
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class LearnedSplit:
    """按升序历史行号定义 search/validation/frozen-test 边界。"""

    search_end: int
    validation_end: int
    test_end: int

    def __post_init__(self) -> None:
        if not 0 <= self.search_end <= self.validation_end <= self.test_end:
            raise ValueError("切分边界必须满足 0 <= search <= validation <= test")

    @classmethod
    def from_length(cls, length: int) -> "LearnedSplit":
        if length < 4:
            raise ValueError("历史至少需要 4 期才能切分")
        search_end = max(1, int(length * 0.50))
        validation_end = max(search_end + 1, int(length * 0.75))
        return cls(search_end, min(validation_end, length - 1), length)

    def to_dict(self) -> dict[str, int]:
        return {
            "searchEnd": self.search_end,
            "validationEnd": self.validation_end,
            "testEnd": self.test_end,
        }


@dataclass(frozen=True)
class LearnedSearchConfig:
    """搜索配置；smoke 可把 trials 和 stride 调小，但算法保持完整。"""

    split: LearnedSplit
    min_train_size: int = 100
    random_trials: int = 24
    local_trials: int = 12
    evaluation_stride: int = 1
    seed: int = 20260717
    feature_config: LearnedFeatureConfig = LearnedFeatureConfig()
    feature_configs: tuple[LearnedFeatureConfig, ...] | None = None

    def __post_init__(self) -> None:
        if self.min_train_size <= 0 or self.evaluation_stride <= 0:
            raise ValueError("最小训练期数和评估步长必须为正整数")
        if self.random_trials <= 0 or self.local_trials < 0:
            raise ValueError("随机搜索次数必须为正，局部搜索次数不得为负")
        if self.feature_configs is not None and not self.feature_configs:
            raise ValueError("feature_configs 不得为空")


@dataclass(frozen=True)
class LearnedSearchTrial:
    params: LearnedRankerParams
    feature_config: LearnedFeatureConfig
    search_objective: float
    validation_objective: float

    def to_dict(self) -> dict[str, object]:
        return {
            "params": self.params.to_dict(),
            "featureConfig": {
                "windows": list(self.feature_config.windows),
                "alpha": self.feature_config.alpha,
                "halfLife": self.feature_config.half_life,
                "omissionCap": self.feature_config.omission_cap,
            },
            "searchObjective": self.search_objective,
            "validationObjective": self.validation_objective,
        }


@dataclass(frozen=True)
class LearnedSearchResult:
    params: LearnedRankerParams
    feature_config: LearnedFeatureConfig
    search_objective: float
    validation_objective: float
    split: LearnedSplit
    trials: tuple[LearnedSearchTrial, ...]
    selection_target_indices: tuple[int, ...]
    test_segment_used_for_selection: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "params": self.params.to_dict(),
            "featureConfig": {
                "windows": list(self.feature_config.windows),
                "alpha": self.feature_config.alpha,
                "halfLife": self.feature_config.half_life,
                "omissionCap": self.feature_config.omission_cap,
            },
            "searchObjective": self.search_objective,
            "validationObjective": self.validation_objective,
            "split": self.split.to_dict(),
            "trials": [item.to_dict() for item in self.trials],
            "selectionTargetIndices": list(self.selection_target_indices),
            "testSegmentUsedForSelection": self.test_segment_used_for_selection,
        }


@dataclass(frozen=True)
class _PreparedTarget:
    features: pd.DataFrame
    actual_text: str


def _prepare_targets(
    chronological: pd.DataFrame,
    rule: LotteryRule,
    indices: Sequence[int],
    feature_config: LearnedFeatureConfig,
) -> tuple[_PreparedTarget, ...]:
    output = []
    for index in indices:
        target = chronological.iloc[index]
        target_issue = str(target["期数"])
        state = build_history_state(
            chronological.iloc[:index], rule, feature_config, target_issue=target_issue
        )
        features = build_candidate_features(state, rule)
        actual_text = "".join(
            str(int(target[column])) for column in rule.number_columns
        )
        output.append(_PreparedTarget(features, actual_text))
    return tuple(output)


def _objective(
    targets: Sequence[_PreparedTarget], params: LearnedRankerParams
) -> float:
    if not targets:
        return -math.inf
    log_ranks = []
    rank_percentiles = []
    direct_hits = []
    group_hits = []
    pool_coverages = []
    block_values: list[float] = []
    for target in targets:
        texts = target.features["candidate"].astype(str).tolist()
        scores = score_candidates(target.features, params)
        probabilities = probabilities_from_scores(
            scores, temperature=params.temperature
        )
        order = rank_candidate_indices(scores, texts)
        candidate_indices = {text: index for index, text in enumerate(texts)}
        try:
            actual_index = candidate_indices[target.actual_text]
        except KeyError as exc:
            raise ValueError(f"候选集合缺少真实号码：{target.actual_text}") from exc
        rank = int(np.flatnonzero(order == actual_index)[0]) + 1
        log_ranks.append(math.log(rank))
        rank_percentiles.append((rank - 1) / 999)
        direct_hits.append(rank <= params.direct_top_k)
        groups = aggregate_group_candidates(
            texts, probabilities, aggregation=params.group_aggregation
        )
        top_group_keys = {item.group_key for item in groups[: params.group_top_k]}
        group_hits.append("".join(sorted(target.actual_text)) in top_group_keys)
        top_direct = [texts[index] for index in order[: max(params.direct_top_k, 50)]]
        pools = [
            set(text[position] for text in top_direct[:50]) for position in range(3)
        ]
        pool_coverages.append(
            sum(
                target.actual_text[position] in pools[position] for position in range(3)
            )
            / 3
        )
        block_values.append((rank - 1) / 999)
    chunks = [
        chunk
        for chunk in np.array_split(np.asarray(block_values), min(3, len(block_values)))
        if len(chunk)
    ]
    instability = (
        float(np.std([float(chunk.mean()) for chunk in chunks]))
        if len(chunks) > 1
        else 0.0
    )
    return float(
        -0.35 * np.mean(log_ranks)
        - 0.25 * np.mean(rank_percentiles)
        + 0.20 * np.mean(direct_hits)
        + 0.15 * np.mean(group_hits)
        + 0.05 * np.mean(pool_coverages)
        - 0.10 * instability
    )


def _random_params(rng: np.random.Generator) -> LearnedRankerParams:
    weights = {
        name: float(np.clip(base + rng.normal(0.0, 0.45), -1.5, 1.5))
        for name, base in DEFAULT_WEIGHTS.items()
    }
    weights["constraint_penalty"] = -abs(weights["constraint_penalty"])
    return LearnedRankerParams(
        weights=weights,
        temperature=float(rng.choice([0.1, 0.2, 0.5, 1.0, 2.0])),
        group_aggregation=str(rng.choice(["sum_prob", "max_perm", "mean_top_perm"])),
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _local_params(
    base: LearnedRankerParams, rng: np.random.Generator
) -> LearnedRankerParams:
    weights = {
        name: float(np.clip(value + rng.normal(0.0, 0.12), -2.0, 2.0))
        for name, value in base.weights.items()
    }
    weights["constraint_penalty"] = -abs(weights["constraint_penalty"])
    temperatures = [0.1, 0.2, 0.5, 1.0, 2.0]
    current = min(
        range(len(temperatures)),
        key=lambda index: abs(temperatures[index] - base.temperature),
    )
    next_index = int(
        np.clip(current + int(rng.choice([-1, 0, 1])), 0, len(temperatures) - 1)
    )
    return replace(
        base,
        weights=weights,
        temperature=temperatures[next_index],
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def search_learned_ranker_params(
    history: pd.DataFrame, rule: LotteryRule, search_config: LearnedSearchConfig
) -> LearnedSearchResult:
    """仅用 search 搜索、validation 选参；绝不读取 frozen test 值。"""

    if rule.draw_count != 3 or rule.code not in {"fc3d", "pl3"}:
        raise ValueError("learned_ranker_v4 首版只支持 fc3d/pl3")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    if search_config.split.test_end > len(chronological):
        raise ValueError("切分终点超过历史长度")
    search_indices = tuple(
        range(
            search_config.min_train_size,
            search_config.split.search_end,
            search_config.evaluation_stride,
        )
    )
    validation_indices = tuple(
        range(
            search_config.split.search_end,
            search_config.split.validation_end,
            search_config.evaluation_stride,
        )
    )
    if not search_indices or not validation_indices:
        raise ValueError("search 和 validation 均需至少一个前向目标期")
    feature_configs = search_config.feature_configs or (search_config.feature_config,)
    prepared = {
        feature_config: (
            _prepare_targets(chronological, rule, search_indices, feature_config),
            _prepare_targets(chronological, rule, validation_indices, feature_config),
        )
        for feature_config in feature_configs
    }
    rng = np.random.default_rng(search_config.seed)
    candidates = [
        LearnedRankerParams(),
        *[_random_params(rng) for _ in range(search_config.random_trials)],
    ]
    coarse = [
        LearnedSearchTrial(
            params,
            feature_config,
            _objective(prepared[feature_config][0], params),
            _objective(prepared[feature_config][1], params),
        )
        for feature_config in feature_configs
        for params in candidates
    ]
    search_best = max(
        coarse,
        key=lambda item: item.search_objective,
    )
    local_params = [
        _local_params(search_best.params, rng)
        for _ in range(search_config.local_trials)
    ]
    local = [
        LearnedSearchTrial(
            params,
            search_best.feature_config,
            _objective(prepared[search_best.feature_config][0], params),
            _objective(prepared[search_best.feature_config][1], params),
        )
        for params in local_params
    ]
    trials = tuple([*coarse, *local])
    best = max(
        trials,
        key=lambda item: (
            item.validation_objective,
            item.search_objective,
            -item.params.temperature,
        ),
    )
    return LearnedSearchResult(
        params=best.params,
        feature_config=best.feature_config,
        search_objective=best.search_objective,
        validation_objective=best.validation_objective,
        split=search_config.split,
        trials=trials,
        selection_target_indices=(*search_indices, *validation_indices),
        test_segment_used_for_selection=False,
    )
