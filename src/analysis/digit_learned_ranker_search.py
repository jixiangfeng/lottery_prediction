# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 的可复现参数搜索。"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, MutableMapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.digit_daily_policy import rank_daily_candidates
from src.analysis.digit_data import (
    canonical_digit_data_sha256,
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_evaluation import evaluate_binary_calibration
from src.analysis.digit_learned_features import (
    FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import (
    DEFAULT_WEIGHTS,
    LearnedRankerParams,
    aggregate_group_candidates,
    probabilities_from_scores,
    rank_candidate_indices,
    score_candidates,
)
from src.analysis.prediction_viability import evaluate_viability_metric
from src.lotteries.base import LotteryRule

FULL_HALF_LIVES = (None, 20, 30, 50, 80, 100, 150, 200)
FULL_OMISSION_CAPS = (20, 30, 50, 80)
FULL_TEMPERATURES = (0.5, 1.0, 2.0, 5.0, 10.0)
FULL_UNIFORM_SHRINKAGES = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
RANKING_UNIFORM_SHRINKAGES = tuple(
    value for value in FULL_UNIFORM_SHRINKAGES if value > 0
)
CALIBRATION_FINALISTS = 3
_PREPARED_CACHE_SCHEMA_VERSION = 2
FULL_ALPHAS = (0.5, 1.0, 2.0, 5.0)
FULL_WINDOW_SETS = (
    (10, 20, 30, 50, 100, 300, "all"),
    (20, 30, 50, 100, 300, "all"),
    (30, 50, 100, 300, "all"),
)
FULL_GROUP_AGGREGATIONS = ("sum_prob", "max_perm", "mean_top_perm")
# (log_rank, rank_percentile, direct, group, pool, instability) weights.
OBJECTIVE_PROFILES: dict[str, tuple[float, float, float, float, float, float]] = {
    "balanced": (0.35, 0.25, 0.20, 0.15, 0.05, 0.10),
    "direct_focus": (0.25, 0.15, 0.45, 0.05, 0.10, 0.15),
    "group_focus": (0.25, 0.15, 0.05, 0.45, 0.10, 0.15),
    "pool_focus": (0.25, 0.15, 0.10, 0.10, 0.40, 0.20),
    # 以下只用于 Search/Validation 开发调参，不能据此反复查看 frozen-test。
    "direct_hit_only": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "group_hit_only": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "pool_hit_only": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    # 相对均匀基线的proper scoring、排名、ECE和时间稳定性。
    "research_calibrated": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}
DIRECT_BUDGETS = (10, 20, 50, 100, 250, 500, 700, 900, 990, 1000)
GROUP_BUDGETS = (10, 20, 50, 100, 150, 220)
POSITION_BUDGETS = (3, 5, 7, 10)
WINDOW_WEIGHT_PROFILES = {
    "balanced": {
        "10": 3,
        "20": 2.5,
        "30": 2,
        "50": 1.5,
        "100": 1,
        "300": 0.5,
        "all": 0.25,
    },
    "recent_heavy": {
        "10": 6,
        "20": 5,
        "30": 4,
        "50": 3,
        "100": 2,
        "300": 0.5,
        "all": 0.25,
    },
    "medium_heavy": {
        "10": 1,
        "20": 2,
        "30": 3,
        "50": 4,
        "100": 4,
        "300": 1,
        "all": 0.25,
    },
    "long_stable": {
        "10": 1,
        "20": 1,
        "30": 1,
        "50": 1.5,
        "100": 2,
        "300": 2,
        "all": 0.5,
    },
}


def build_search_space_manifest(
    *,
    smoke: bool,
    evaluated_feature_configs: Sequence[LearnedFeatureConfig] = (),
    direct_top_k: int = 50,
    group_top_k: int = 10,
    position_pool_size: int = 5,
    incumbent_included: bool = False,
) -> dict[str, object]:
    """记录完整正式空间；smoke 只改变采样预算，不删减空间声明。"""

    space = {
        "halfLife": list(FULL_HALF_LIVES),
        "omissionCap": list(FULL_OMISSION_CAPS),
        "temperature": list(FULL_TEMPERATURES),
        "uniformShrinkage": list(RANKING_UNIFORM_SHRINKAGES),
        "abstentionUniformShrinkage": 0.0,
        "alpha": list(FULL_ALPHAS),
        "windowSets": [list(values) for values in FULL_WINDOW_SETS],
        "groupAggregation": list(FULL_GROUP_AGGREGATIONS),
        "objectiveProfiles": {
            name: list(weights) for name, weights in OBJECTIVE_PROFILES.items()
        },
        "featureNormalization": "robust_zscore",
        "featureWeightRanges": {
            name: (
                {"minimum": -1.5, "maximum": 0.0}
                if name == "constraint_penalty"
                else {"minimum": -1.5, "maximum": 1.5}
            )
            for name in DEFAULT_WEIGHTS
        },
        "recommendationConfig": {
            "directTopK": [direct_top_k],
            "groupTopK": [group_top_k],
            "positionPoolSize": [position_pool_size],
            "groupDigitPoolSize": [7],
        },
        "windowWeightProfiles": [
            {"name": name, "weights": weights}
            for name, weights in WINDOW_WEIGHT_PROFILES.items()
        ],
    }
    return {
        **space,
        "space": space,
        "sampling": {
            "deterministic": True,
            "materializesFullCartesianProduct": False,
            "strategy": "seeded staged sampling with per-feature-config lazy preparation",
            "mode": "smoke" if smoke else "formal",
            "incumbentIncluded": incumbent_included,
            "rankingCalibrationSeparated": True,
            "calibrationFinalists": CALIBRATION_FINALISTS,
            "evaluatedFeatureConfigs": [
                {
                    "windows": list(config.windows),
                    "alpha": config.alpha,
                    "halfLife": config.half_life,
                    "omissionCap": config.omission_cap,
                    "windowWeights": dict(config.window_weights or ()),
                }
                for config in evaluated_feature_configs
            ],
        },
    }


def sample_feature_configs(
    *, seed: int, smoke: bool, limit: int | None = None
) -> tuple[LearnedFeatureConfig, ...]:
    """保留默认基线，再从完整空间做确定性边际均衡采样。"""

    if smoke:
        return (LearnedFeatureConfig(),)
    sample_size = min(limit or 16, 1 + 3 * 4 * 8 * 4 * 4)
    if sample_size == 1:
        return (LearnedFeatureConfig(),)
    combinations = list(
        itertools.product(
            FULL_WINDOW_SETS,
            FULL_ALPHAS,
            FULL_HALF_LIVES,
            FULL_OMISSION_CAPS,
            WINDOW_WEIGHT_PROFILES.items(),
        )
    )
    rng = np.random.default_rng(seed)
    tie_breaks = rng.random(len(combinations))
    counts: list[dict[object, int]] = [dict() for _ in range(5)]
    remaining = set(range(len(combinations)))
    selected = [LearnedFeatureConfig()]
    for _ in range(sample_size - 1):

        def balance_score(index: int) -> tuple[float, float]:
            windows, alpha, half_life, omission_cap, (profile, _) = combinations[index]
            dimensions = (windows, alpha, half_life, omission_cap, profile)
            score = sum(
                1.0 / (1 + counts[position].get(value, 0))
                for position, value in enumerate(dimensions)
            )
            return score, float(tie_breaks[index])

        index = max(remaining, key=balance_score)
        remaining.remove(index)
        windows, alpha, half_life, omission_cap, (profile, weights) = combinations[
            index
        ]
        selected.append(
            LearnedFeatureConfig(
                windows=windows,
                alpha=alpha,
                half_life=half_life,
                omission_cap=omission_cap,
                window_weights={str(value): weights[str(value)] for value in windows},
            )
        )
        for position, value in enumerate(
            (windows, alpha, half_life, omission_cap, profile)
        ):
            counts[position][value] = counts[position].get(value, 0) + 1
    return tuple(selected)


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
    def from_length(
        cls, length: int, *, frozen_test_periods: int | None = None
    ) -> "LearnedSplit":
        if length < 4:
            raise ValueError("历史至少需要 4 期才能切分")
        if frozen_test_periods is None:
            search_end = max(1, int(length * 0.50))
            validation_end = max(search_end + 1, int(length * 0.75))
            return cls(search_end, min(validation_end, length - 1), length)
        if frozen_test_periods <= 0 or frozen_test_periods >= length - 1:
            raise ValueError("冻结测试期数必须小于历史长度减一")
        pretest_end = length - frozen_test_periods
        search_end = max(1, pretest_end // 2)
        return cls(search_end, pretest_end, length)

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
    incumbent_params: LearnedRankerParams | None = None
    incumbent_feature_config: LearnedFeatureConfig | None = None
    objective_profile: str = "balanced"
    direct_objective_top_k: int = 50
    group_objective_top_k: int = 10
    position_objective_pool_size: int = 5
    smoke: bool = False
    progress_checkpoint_path: str | Path | None = None
    require_search_qualification: bool = True
    validation_lock_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.min_train_size <= 0 or self.evaluation_stride <= 0:
            raise ValueError("最小训练期数和评估步长必须为正整数")
        if self.random_trials <= 0 or self.local_trials < 0:
            raise ValueError("随机搜索次数必须为正，局部搜索次数不得为负")
        if self.feature_configs is not None and not self.feature_configs:
            raise ValueError("feature_configs 不得为空")
        if (self.incumbent_params is None) != (self.incumbent_feature_config is None):
            raise ValueError("incumbent参数与特征配置必须同时提供")
        if self.objective_profile not in OBJECTIVE_PROFILES:
            raise ValueError(f"未知 objective profile：{self.objective_profile}")
        if not 1 <= self.direct_objective_top_k <= 1000:
            raise ValueError("直选目标预算必须位于 1..1000")
        if not 1 <= self.group_objective_top_k <= 220:
            raise ValueError("组选目标预算必须位于 1..220")
        if not 1 <= self.position_objective_pool_size <= 10:
            raise ValueError("位置池目标预算必须位于 1..10")


@dataclass(frozen=True)
class LearnedSearchTrial:
    params: LearnedRankerParams
    feature_config: LearnedFeatureConfig
    search_objective: float
    validation_objective: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "params": self.params.to_dict(),
            "featureConfig": {
                "windows": list(self.feature_config.windows),
                "alpha": self.feature_config.alpha,
                "halfLife": self.feature_config.half_life,
                "omissionCap": self.feature_config.omission_cap,
                "windowWeights": dict(self.feature_config.window_weights or ()),
            },
            "searchObjective": self.search_objective,
            "validationObjective": self.validation_objective,
        }


@dataclass(frozen=True)
class LearnedSearchResult:
    params: LearnedRankerParams
    feature_config: LearnedFeatureConfig
    search_objective: float
    validation_objective: float | None
    split: LearnedSplit
    trials: tuple[LearnedSearchTrial, ...]
    selection_target_indices: tuple[int, ...]
    search_passed: bool
    search_reasons: tuple[str, ...]
    validation_evaluated: bool
    validation_passed: bool
    validation_reasons: tuple[str, ...]
    test_segment_used_for_selection: bool = False
    objective_profile: str = "balanced"
    search_space_manifest: dict[str, object] | None = None
    budget_curves: dict[str, object] | None = None
    search_confirmation: dict[str, object] | None = None
    validation_confirmation: dict[str, object] | None = None
    validation_lock_fingerprint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "params": self.params.to_dict(),
            "featureConfig": {
                "windows": list(self.feature_config.windows),
                "alpha": self.feature_config.alpha,
                "halfLife": self.feature_config.half_life,
                "omissionCap": self.feature_config.omission_cap,
                "windowWeights": dict(self.feature_config.window_weights or ()),
            },
            "searchObjective": self.search_objective,
            "validationObjective": self.validation_objective,
            "split": self.split.to_dict(),
            "trials": [item.to_dict() for item in self.trials],
            "selectionTargetIndices": list(self.selection_target_indices),
            "searchPassed": self.search_passed,
            "searchReasons": list(self.search_reasons),
            "validationEvaluated": self.validation_evaluated,
            "validationPassed": self.validation_passed,
            "validationReasons": list(self.validation_reasons),
            "testSegmentUsedForSelection": self.test_segment_used_for_selection,
            "objectiveProfile": self.objective_profile,
            "budgetCurves": self.budget_curves or {},
            "searchConfirmation": self.search_confirmation or {},
            "validationConfirmation": self.validation_confirmation or {},
            "validationLockFingerprint": self.validation_lock_fingerprint,
            "searchSpaceManifest": self.search_space_manifest
            or build_search_space_manifest(smoke=False),
        }


@dataclass(frozen=True)
class _PreparedTarget:
    features: pd.DataFrame
    actual_text: str
    latest_exact: str | None = None


def _budget_entry(
    hits: int, observations: int, random_baseline: float
) -> dict[str, float | int]:
    hit_rate = hits / observations if observations else 0.0
    return {
        "hits": hits,
        "observations": observations,
        "hitRate": hit_rate,
        "randomBaseline": random_baseline,
        "lift": hit_rate / random_baseline if random_baseline > 0 else 0.0,
    }


def build_development_budget_curves(
    targets: Sequence[_PreparedTarget],
    params: LearnedRankerParams,
    *,
    include_time_blocks: bool = True,
) -> dict[str, Any]:
    """在同一批已构建目标上统计多个固定候选预算，不参与选参。"""

    direct_budgets = tuple(sorted({*DIRECT_BUDGETS, params.direct_top_k}))
    group_budgets = tuple(sorted({*GROUP_BUDGETS, params.group_top_k}))
    position_budgets = tuple(sorted({*POSITION_BUDGETS, params.position_pool_size}))
    direct_hits = {budget: 0 for budget in direct_budgets}
    group_hits = {budget: 0 for budget in group_budgets}
    group_baselines: dict[int, list[float]] = {budget: [] for budget in group_budgets}
    position_hits = {budget: 0 for budget in position_budgets}
    for target in targets:
        texts = target.features["candidate"].astype(str).tolist()
        scores = score_candidates(target.features, params)
        probabilities = probabilities_from_scores(
            scores,
            temperature=params.temperature,
            uniform_shrinkage=params.uniform_shrinkage,
        )
        order = rank_candidate_indices(probabilities, texts)
        ranked_texts = [texts[int(index)] for index in order]
        if target.latest_exact is not None:
            ranked_texts = list(
                rank_daily_candidates(
                    ranked_texts,
                    latest_exact=target.latest_exact,
                    maximum_triples=1,
                )
            )
        if target.actual_text not in set(texts):
            raise ValueError(f"候选集合缺少真实号码：{target.actual_text}")
        actual_rank = ranked_texts.index(target.actual_text) + 1
        for budget in direct_budgets:
            direct_hits[budget] += int(actual_rank <= budget)

        groups = aggregate_group_candidates(
            texts, probabilities, aggregation=params.group_aggregation
        )
        actual_group = "".join(sorted(target.actual_text))
        group_rank = next(
            position
            for position, item in enumerate(groups, start=1)
            if item.group_key == actual_group
        )
        for budget in group_budgets:
            selected = groups[:budget]
            group_hits[budget] += int(group_rank <= budget)
            group_baselines[budget].append(
                sum(item.permutations for item in selected) / 1000.0
            )

        position_ranks = []
        for position in range(3):
            masses = {
                digit: math.fsum(
                    float(probabilities[index])
                    for index, text in enumerate(texts)
                    if int(text[position]) == digit
                )
                for digit in range(10)
            }
            ordered = sorted(masses, key=lambda digit: (-masses[digit], digit))
            position_ranks.append(ordered.index(int(target.actual_text[position])) + 1)
        for budget in position_budgets:
            position_hits[budget] += sum(rank <= budget for rank in position_ranks)

    periods = len(targets)
    result: dict[str, Any] = {
        "direct": {
            str(budget): _budget_entry(direct_hits[budget], periods, budget / 1000.0)
            for budget in direct_budgets
        },
        "group": {
            str(budget): _budget_entry(
                group_hits[budget],
                periods,
                float(np.mean(group_baselines[budget])) if periods else 0.0,
            )
            for budget in group_budgets
        },
        "position": {
            str(budget): _budget_entry(
                position_hits[budget], periods * 3, budget / 10.0
            )
            for budget in position_budgets
        },
    }
    if include_time_blocks and periods:
        block_indices = np.array_split(np.arange(periods), min(3, periods))
        block_curves = [
            build_development_budget_curves(
                tuple(targets[int(index)] for index in indices),
                params,
                include_time_blocks=False,
            )
            for indices in block_indices
            if len(indices)
        ]
        for kind, budgets in result.items():
            for budget, metrics in budgets.items():
                metrics["timeBlocks"] = [block[kind][budget] for block in block_curves]
    return result


def select_joint_budget(
    curves_by_lottery: dict[str, dict[str, Any]],
    *,
    kind: str,
    full_coverage_budget: int,
    minimum_stable_blocks: int = 2,
) -> dict[str, Any]:
    """只用 Search 跨彩种选择预算，再用 Validation 验证已选预算。"""

    lottery_codes = tuple(sorted(curves_by_lottery))
    if len(lottery_codes) < 2:
        raise ValueError("联合预算选择至少需要两个彩种")
    first = curves_by_lottery[lottery_codes[0]]["search"][kind]
    candidates = []
    for budget_text in first:
        budget = int(budget_text)
        if budget == full_coverage_budget:
            continue
        search_metrics = {
            code: curves_by_lottery[code]["search"][kind][budget_text]
            for code in lottery_codes
        }
        stable_counts = {
            code: sum(
                float(block["lift"]) >= 1.0
                for block in search_metrics[code].get("timeBlocks", [])
            )
            for code in lottery_codes
        }
        if all(
            float(search_metrics[code]["lift"]) > 1.0
            and stable_counts[code] >= minimum_stable_blocks
            for code in lottery_codes
        ):
            candidates.append(
                (
                    min(float(search_metrics[code]["lift"]) for code in lottery_codes),
                    -budget,
                    budget_text,
                    search_metrics,
                    stable_counts,
                )
            )
    if not candidates:
        return {
            "selectedBudget": None,
            "searchQualified": False,
            "validationConfirmed": False,
            "validationUsedForSelection": False,
        }

    _, _, selected_text, search_metrics, search_stable_counts = max(candidates)
    validation_metrics = {
        code: curves_by_lottery[code]["validation"][kind][selected_text]
        for code in lottery_codes
    }
    validation_stable_counts = {
        code: sum(
            float(block["lift"]) >= 1.0
            for block in validation_metrics[code].get("timeBlocks", [])
        )
        for code in lottery_codes
    }
    validation_confirmed = all(
        float(validation_metrics[code]["lift"]) > 1.0
        and validation_stable_counts[code] >= minimum_stable_blocks
        for code in lottery_codes
    )
    return {
        "selectedBudget": int(selected_text),
        "searchQualified": True,
        "search": search_metrics,
        "searchStableBlocks": search_stable_counts,
        "validation": validation_metrics,
        "validationStableBlocks": validation_stable_counts,
        "validationConfirmed": validation_confirmed,
        "validationUsedForSelection": False,
    }


def _write_progress_checkpoint(
    path: str | Path | None, payload: dict[str, object]
) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(target)


def _prepared_cache_path(
    checkpoint_path: str | Path | None,
    *,
    rule_code: str,
    data_fingerprint: str,
    feature_config: LearnedFeatureConfig,
    indices: tuple[int, ...],
) -> Path | None:
    if checkpoint_path is None:
        return None
    payload = json.dumps(
        {
            "schemaVersion": _PREPARED_CACHE_SCHEMA_VERSION,
            "rule": rule_code,
            "data": data_fingerprint,
            "featureConfig": str(feature_config),
            "indices": indices,
            "features": FEATURE_NAMES,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    checkpoint = Path(checkpoint_path)
    return checkpoint.parent / f"{checkpoint.stem}.prepared" / f"{digest}.npz"


def _write_prepared_cache(
    path: Path | None, targets: tuple[_PreparedTarget, ...]
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.stack(
        [
            target.features[list(FEATURE_NAMES)].to_numpy(dtype=float)
            for target in targets
        ]
    )
    actual = np.asarray([target.actual_text for target in targets], dtype="U3")
    latest = np.asarray([target.latest_exact or "" for target in targets], dtype="U3")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray([_PREPARED_CACHE_SCHEMA_VERSION]),
                values=values,
                actual=actual,
                latest=latest,
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_prepared_cache(path: Path | None) -> tuple[_PreparedTarget, ...] | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        schema_version = int(payload["schema_version"][0])
        values = payload["values"]
        actual = payload["actual"].astype(str)
        latest = payload["latest"].astype(str)
    if schema_version != _PREPARED_CACHE_SCHEMA_VERSION:
        raise ValueError(f"目标特征缓存版本错误：{path}")
    if values.ndim != 3 or values.shape[1:] != (1000, len(FEATURE_NAMES)):
        raise ValueError(f"目标特征缓存结构错误：{path}")
    if len(actual) != len(values) or len(latest) != len(values):
        raise ValueError(f"目标特征缓存期数错误：{path}")
    candidates = [f"{value:03d}" for value in range(1000)]
    return tuple(
        _PreparedTarget(
            pd.DataFrame(
                {
                    "candidate": candidates,
                    **{
                        name: matrix[:, feature_index]
                        for feature_index, name in enumerate(FEATURE_NAMES)
                    },
                }
            ),
            actual_text,
            latest_exact or None,
        )
        for matrix, actual_text, latest_exact in zip(values, actual, latest)
    )


def _prepare_targets(
    chronological: pd.DataFrame,
    rule: LotteryRule,
    indices: Sequence[int],
    feature_config: LearnedFeatureConfig,
) -> tuple[_PreparedTarget, ...]:
    output = []
    total = len(indices)
    states = iter_rolling_history_states(chronological, rule, indices, feature_config)
    for completed, (index, state) in enumerate(zip(indices, states), start=1):
        target = chronological.iloc[index]
        features = build_candidate_features(state, rule)
        features = features.loc[:, ["candidate", *FEATURE_NAMES]].copy()
        actual_text = "".join(
            str(int(target[column])) for column in rule.number_columns
        )
        latest_exact = (
            "".join(str(value) for value in state.latest_numbers)
            if state.latest_numbers is not None
            else None
        )
        output.append(_PreparedTarget(features, actual_text, latest_exact))
        if completed == total or completed % 50 == 0:
            print(
                f"[learned_ranker_v4] 特征准备 {completed}/{total} "
                f"({feature_config.windows})",
                flush=True,
            )
    return tuple(output)


def _objective(
    targets: Sequence[_PreparedTarget],
    params: LearnedRankerParams,
    *,
    objective_profile: str = "balanced",
) -> float:
    if objective_profile not in OBJECTIVE_PROFILES:
        raise ValueError(f"未知 objective profile：{objective_profile}")
    if not targets:
        return -math.inf
    log_ranks = []
    rank_percentiles = []
    log_losses = []
    brier_scores = []
    top_confidences = []
    top_hits = []
    direct_hits = []
    group_hits = []
    pool_coverages = []
    block_values: list[float] = []
    for target in targets:
        texts = target.features["candidate"].astype(str).tolist()
        scores = score_candidates(target.features, params)
        probabilities = probabilities_from_scores(
            scores,
            temperature=params.temperature,
            uniform_shrinkage=params.uniform_shrinkage,
        )
        order = rank_candidate_indices(probabilities, texts)
        ranked_texts = [texts[int(index)] for index in order]
        if target.latest_exact is not None:
            ranked_texts = list(
                rank_daily_candidates(
                    ranked_texts,
                    latest_exact=target.latest_exact,
                    maximum_triples=1,
                )
            )
        candidate_indices = {text: index for index, text in enumerate(texts)}
        try:
            actual_index = candidate_indices[target.actual_text]
        except KeyError as exc:
            raise ValueError(f"候选集合缺少真实号码：{target.actual_text}") from exc
        rank = ranked_texts.index(target.actual_text) + 1
        log_ranks.append(math.log(rank))
        rank_percentiles.append((rank - 1) / 999)
        actual_probability = float(probabilities[actual_index])
        log_losses.append(-math.log(max(actual_probability, 1e-300)))
        brier_scores.append(
            float(np.sum(probabilities**2) - 2 * actual_probability + 1.0)
        )
        policy_top_indices = [
            candidate_indices[text] for text in ranked_texts[: params.direct_top_k]
        ]
        top_confidences.append(
            float(
                math.fsum(float(probabilities[index]) for index in policy_top_indices)
            )
        )
        top_hits.append(rank <= params.direct_top_k)
        direct_hits.append(rank <= params.direct_top_k)
        groups = aggregate_group_candidates(
            texts, probabilities, aggregation=params.group_aggregation
        )
        top_group_keys = {item.group_key for item in groups[: params.group_top_k]}
        group_hits.append("".join(sorted(target.actual_text)) in top_group_keys)
        pools = []
        for position in range(3):
            masses = {
                digit: math.fsum(
                    float(probabilities[candidate_index])
                    for candidate_index, text in enumerate(texts)
                    if int(text[position]) == digit
                )
                for digit in range(10)
            }
            pools.append(
                set(
                    sorted(masses, key=lambda digit: (-masses[digit], digit))[
                        : params.position_pool_size
                    ]
                )
            )
        pool_coverages.append(
            sum(
                int(target.actual_text[position]) in pools[position]
                for position in range(3)
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
    if objective_profile == "research_calibrated":
        _, ece = evaluate_binary_calibration(top_confidences, top_hits)
        uniform_log_loss = math.log(1000)
        uniform_brier = 1.0 - 1.0 / 1000
        log_gain = (uniform_log_loss - float(np.mean(log_losses))) / uniform_log_loss
        brier_gain = (uniform_brier - float(np.mean(brier_scores))) / uniform_brier
        rank_gain = 0.5 - float(np.mean(rank_percentiles))
        secondary = float(
            0.40 * np.clip(log_gain, -1.0, 1.0)
            + 0.30 * np.clip(brier_gain, -1.0, 1.0)
            + 0.20 * np.clip(2.0 * rank_gain, -1.0, 1.0)
            - 0.05 * np.clip(ece, 0.0, 1.0)
            - 0.05 * np.clip(2.0 * instability, 0.0, 1.0)
        )
        # 次级指标的最大差值小于一次命中，确保Top50命中数严格优先。
        return float(
            np.mean(direct_hits) + (0.49 / len(targets)) * np.clip(secondary, -1.0, 1.0)
        )
    (
        log_weight,
        percentile_weight,
        direct_weight,
        group_weight,
        pool_weight,
        instability_weight,
    ) = OBJECTIVE_PROFILES[objective_profile]
    return float(
        -log_weight * np.mean(log_ranks)
        - percentile_weight * np.mean(rank_percentiles)
        + direct_weight * np.mean(direct_hits)
        + group_weight * np.mean(group_hits)
        + pool_weight * np.mean(pool_coverages)
        - instability_weight * instability
    )


def _random_params(rng: np.random.Generator) -> LearnedRankerParams:
    weights = {
        name: float(np.clip(base + rng.normal(0.0, 0.18), -1.5, 1.5))
        for name, base in DEFAULT_WEIGHTS.items()
    }
    weights["constraint_penalty"] = -abs(weights["constraint_penalty"])
    return LearnedRankerParams(
        weights=weights,
        temperature=1.0,
        uniform_shrinkage=1.0,
        group_aggregation=str(rng.choice(FULL_GROUP_AGGREGATIONS)),
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _local_params(
    base: LearnedRankerParams, rng: np.random.Generator
) -> LearnedRankerParams:
    weights = {
        name: float(np.clip(value + rng.normal(0.0, 0.08), -1.5, 1.5))
        for name, value in base.weights.items()
    }
    weights["constraint_penalty"] = -abs(weights["constraint_penalty"])
    return replace(
        base,
        weights=weights,
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _profiled_objective(
    targets: Sequence[_PreparedTarget],
    params: LearnedRankerParams,
    objective_profile: str,
) -> float:
    return _objective(targets, params, objective_profile=objective_profile)


def _calibration_metrics(
    targets: Sequence[_PreparedTarget], params: LearnedRankerParams
) -> dict[str, float]:
    """在排名结构固定后计算proper scoring；选择时按LogLoss/Brier/ECE依次裁决。"""

    log_losses = []
    brier_scores = []
    top_confidences = []
    top_hits = []
    for target in targets:
        texts = target.features["candidate"].astype(str).tolist()
        probabilities = probabilities_from_scores(
            score_candidates(target.features, params),
            temperature=params.temperature,
            uniform_shrinkage=params.uniform_shrinkage,
        )
        candidate_indices = {text: index for index, text in enumerate(texts)}
        actual_index = candidate_indices[target.actual_text]
        actual_probability = float(probabilities[actual_index])
        log_losses.append(-math.log(max(actual_probability, 1e-300)))
        brier_scores.append(
            float(np.sum(probabilities**2) - 2 * actual_probability + 1.0)
        )
        order = rank_candidate_indices(probabilities, texts)
        ranked_texts = [texts[int(index)] for index in order]
        if target.latest_exact is not None:
            ranked_texts = list(
                rank_daily_candidates(
                    ranked_texts,
                    latest_exact=target.latest_exact,
                    maximum_triples=1,
                )
            )
        policy_top_indices = [
            candidate_indices[text] for text in ranked_texts[: params.direct_top_k]
        ]
        top_confidences.append(
            float(
                math.fsum(float(probabilities[index]) for index in policy_top_indices)
            )
        )
        top_hits.append(target.actual_text in ranked_texts[: params.direct_top_k])
    _, ece = evaluate_binary_calibration(top_confidences, top_hits)
    return {
        "meanLogLoss": float(np.mean(log_losses)),
        "meanBrierScore": float(np.mean(brier_scores)),
        "topKExpectedCalibrationError": ece,
    }


def _calibration_selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        -metrics["meanLogLoss"],
        -metrics["meanBrierScore"],
        -metrics["topKExpectedCalibrationError"],
    )


@dataclass(frozen=True)
class _ConfirmationSequences:
    kind: str
    hits: tuple[bool, ...]
    random_probabilities: tuple[float, ...]
    log_loss_improvements: tuple[float, ...]
    brier_improvements: tuple[float, ...]


def _confirmation_sequences(
    targets: Sequence[_PreparedTarget],
    params: LearnedRankerParams,
    objective_profile: str,
) -> _ConfirmationSequences:
    if objective_profile in {"group_focus", "group_hit_only"}:
        kind = "group"
    elif objective_profile in {"pool_focus", "pool_hit_only"}:
        kind = "position"
    else:
        kind = "direct"

    hits = []
    random_probabilities = []
    log_loss_improvements = []
    brier_improvements = []
    uniform_log_loss = math.log(1000)
    uniform_brier = 1.0 - 1.0 / 1000
    for target in targets:
        texts = target.features["candidate"].astype(str).tolist()
        probabilities = probabilities_from_scores(
            score_candidates(target.features, params),
            temperature=params.temperature,
            uniform_shrinkage=params.uniform_shrinkage,
        )
        candidate_indices = {text: index for index, text in enumerate(texts)}
        actual_index = candidate_indices[target.actual_text]
        actual_probability = float(probabilities[actual_index])
        log_loss = -math.log(max(actual_probability, 1e-300))
        brier = float(np.sum(probabilities**2) - 2 * actual_probability + 1.0)
        log_loss_improvements.append(uniform_log_loss - log_loss)
        brier_improvements.append(uniform_brier - brier)

        if kind == "direct":
            order = rank_candidate_indices(probabilities, texts)
            ranked_texts = [texts[int(index)] for index in order]
            if target.latest_exact is not None:
                ranked_texts = list(
                    rank_daily_candidates(
                        ranked_texts,
                        latest_exact=target.latest_exact,
                        maximum_triples=1,
                    )
                )
            hits.append(target.actual_text in ranked_texts[: params.direct_top_k])
            random_probabilities.append(params.direct_top_k / 1000.0)
        elif kind == "group":
            groups = aggregate_group_candidates(
                texts, probabilities, aggregation=params.group_aggregation
            )
            selected = groups[: params.group_top_k]
            hits.append(
                "".join(sorted(target.actual_text))
                in {item.group_key for item in selected}
            )
            random_probabilities.append(
                sum(item.permutations for item in selected) / 1000.0
            )
        else:
            for position in range(3):
                masses = {
                    digit: math.fsum(
                        float(probabilities[index])
                        for index, text in enumerate(texts)
                        if int(text[position]) == digit
                    )
                    for digit in range(10)
                }
                selected_digits = set(
                    sorted(masses, key=lambda digit: (-masses[digit], digit))[
                        : params.position_pool_size
                    ]
                )
                hits.append(int(target.actual_text[position]) in selected_digits)
                random_probabilities.append(params.position_pool_size / 10.0)
    return _ConfirmationSequences(
        kind=kind,
        hits=tuple(hits),
        random_probabilities=tuple(random_probabilities),
        log_loss_improvements=tuple(log_loss_improvements),
        brier_improvements=tuple(brier_improvements),
    )


def _block_bootstrap_lower_bound(
    values: Sequence[float],
    *,
    seed: int,
    block_size: int = 10,
    resamples: int = 999,
    alpha: float = 0.01,
) -> float:
    """对连续非重叠时间块做确定性bootstrap，返回单侧下界。"""

    array = np.asarray(values, dtype=float)
    if array.size < block_size * 2:
        return -math.inf
    block_means = np.asarray(
        [
            float(array[start : start + block_size].mean())
            for start in range(0, len(array), block_size)
        ]
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(block_means), size=(resamples, len(block_means)))
    bootstrap_means = block_means[sampled].mean(axis=1)
    return float(np.quantile(bootstrap_means, alpha))


def _validation_confirmation(
    targets: Sequence[_PreparedTarget],
    params: LearnedRankerParams,
    objective_profile: str,
    *,
    seed: int,
) -> tuple[bool, tuple[str, ...], dict[str, object]]:
    """按显著性、提升幅度、置信下界和时间块确认唯一胜者。"""

    sequences = _confirmation_sequences(targets, params, objective_profile)
    viability = evaluate_viability_metric(
        sequences.kind,
        sequences.hits,
        sequences.random_probabilities,
    )
    log_loss_lower = _block_bootstrap_lower_bound(
        sequences.log_loss_improvements,
        seed=seed,
    )
    brier_lower = _block_bootstrap_lower_bound(
        sequences.brier_improvements,
        seed=seed + 1,
    )
    reasons = [] if viability.viable else [viability.reason]
    if log_loss_lower <= 0:
        reasons.append("LogLoss改善的99%时间块bootstrap下界未高于0")
    if brier_lower <= 0:
        reasons.append("Brier改善的99%时间块bootstrap下界未高于0")
    diagnostics: dict[str, object] = {
        "metricKind": sequences.kind,
        "viability": viability.to_dict(),
        "meanLogLossImprovement": float(np.mean(sequences.log_loss_improvements)),
        "logLossImprovementLowerBound99": log_loss_lower,
        "meanBrierImprovement": float(np.mean(sequences.brier_improvements)),
        "brierImprovementLowerBound99": brier_lower,
        "bootstrapBlockSize": 10,
        "bootstrapResamples": 999,
    }
    return not reasons, tuple(reasons), diagnostics


def _claim_validation_lock(
    path: str | Path | None,
    payload: dict[str, object],
) -> str:
    serialized_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    fingerprint = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
    if path is None:
        return fingerprint
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {**payload, "validationLockFingerprint": fingerprint}
    content = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise RuntimeError(f"Validation已被打开，禁止重复使用：{destination}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    return fingerprint


def search_learned_ranker_params(
    history: pd.DataFrame,
    rule: LotteryRule,
    search_config: LearnedSearchConfig,
    *,
    prepared_target_cache: (
        MutableMapping[
            tuple[str, LearnedFeatureConfig, tuple[int, ...]],
            tuple[_PreparedTarget, ...],
        ]
        | None
    ) = None,
) -> LearnedSearchResult:
    """仅用Search选择参数，Validation只确认唯一胜者；绝不读取Frozen Test。"""

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
    feature_configs = list(
        search_config.feature_configs or (search_config.feature_config,)
    )
    if (
        search_config.incumbent_feature_config is not None
        and search_config.incumbent_feature_config not in feature_configs
    ):
        feature_configs.append(search_config.incumbent_feature_config)
    evaluated_feature_configs = tuple(feature_configs)
    data_fingerprint = canonical_digit_data_sha256(chronological, rule)

    def prepared(
        feature_config: LearnedFeatureConfig, indices: tuple[int, ...]
    ) -> tuple[_PreparedTarget, ...]:
        key = (rule.code, feature_config, indices)
        if prepared_target_cache is not None and key in prepared_target_cache:
            return prepared_target_cache[key]
        disk_path = _prepared_cache_path(
            search_config.progress_checkpoint_path,
            rule_code=rule.code,
            data_fingerprint=data_fingerprint,
            feature_config=feature_config,
            indices=indices,
        )
        restored = _read_prepared_cache(disk_path)
        if restored is not None:
            if prepared_target_cache is not None:
                prepared_target_cache[key] = restored
            print(f"[learned_ranker_v4] 已恢复目标缓存：{disk_path}", flush=True)
            return restored
        targets = _prepare_targets(chronological, rule, indices, feature_config)
        _write_prepared_cache(disk_path, targets)
        if prepared_target_cache is not None:
            prepared_target_cache[key] = targets
        return targets

    def ranking_params(params: LearnedRankerParams) -> LearnedRankerParams:
        return replace(
            params,
            temperature=1.0,
            uniform_shrinkage=1.0,
            direct_top_k=search_config.direct_objective_top_k,
            group_top_k=search_config.group_objective_top_k,
            position_pool_size=search_config.position_objective_pool_size,
        )

    def params_key(params: LearnedRankerParams) -> str:
        return json.dumps(params.to_dict(), sort_keys=True, separators=(",", ":"))

    def unique_params(
        values: Sequence[LearnedRankerParams],
    ) -> tuple[LearnedRankerParams, ...]:
        unique: dict[str, LearnedRankerParams] = {}
        for value in values:
            normalized = ranking_params(value)
            unique.setdefault(params_key(normalized), normalized)
        return tuple(unique.values())

    rng = np.random.default_rng(search_config.seed)
    base_candidates = unique_params(
        [
            LearnedRankerParams(),
            *[_random_params(rng) for _ in range(search_config.random_trials)],
        ]
    )
    coarse: list[LearnedSearchTrial] = []
    completed_configs: list[dict[str, object]] = []
    for config_number, feature_config in enumerate(evaluated_feature_configs, start=1):
        prepared_search = prepared(feature_config, search_indices)
        config_candidates = list(base_candidates)
        if feature_config == search_config.incumbent_feature_config:
            assert search_config.incumbent_params is not None
            config_candidates.append(search_config.incumbent_params)
        config_trials = [
            LearnedSearchTrial(
                params,
                feature_config,
                _profiled_objective(
                    prepared_search, params, search_config.objective_profile
                ),
            )
            for params in unique_params(config_candidates)
        ]
        coarse.extend(config_trials)
        config_best = max(config_trials, key=lambda item: item.search_objective)
        completed_configs.append(
            {
                "configNumber": config_number,
                "configTotal": len(evaluated_feature_configs),
                "featureConfig": config_best.feature_config,
                "searchObjective": config_best.search_objective,
            }
        )
        _write_progress_checkpoint(
            search_config.progress_checkpoint_path,
            {
                "status": "searching",
                "lottery": rule.code,
                "searchTargets": len(search_indices),
                "validationTargets": len(validation_indices),
                "completedFeatureConfigs": completed_configs,
            },
        )
    search_best = max(coarse, key=lambda item: item.search_objective)
    local_params = unique_params(
        [
            _local_params(search_best.params, rng)
            for _ in range(search_config.local_trials)
        ]
    )
    local_search_targets = prepared(search_best.feature_config, search_indices)
    local = [
        LearnedSearchTrial(
            params,
            search_best.feature_config,
            _profiled_objective(
                local_search_targets, params, search_config.objective_profile
            ),
        )
        for params in local_params
    ]
    ranking_trials = tuple([*coarse, *local])
    ranking_finalists: list[LearnedSearchTrial] = []
    finalist_keys: set[tuple[str, LearnedFeatureConfig]] = set()
    for trial in sorted(
        ranking_trials, key=lambda item: item.search_objective, reverse=True
    ):
        key = (params_key(trial.params), trial.feature_config)
        if key in finalist_keys:
            continue
        finalist_keys.add(key)
        ranking_finalists.append(trial)
        if len(ranking_finalists) == CALIBRATION_FINALISTS:
            break

    calibration_trials: list[LearnedSearchTrial] = []
    calibrated_finalists = []
    for finalist in ranking_finalists:
        finalist_targets = prepared(finalist.feature_config, search_indices)
        options = []
        for temperature, uniform_shrinkage in itertools.product(
            FULL_TEMPERATURES, RANKING_UNIFORM_SHRINKAGES
        ):
            params = replace(
                finalist.params,
                temperature=temperature,
                uniform_shrinkage=uniform_shrinkage,
            )
            metrics = _calibration_metrics(finalist_targets, params)
            trial = LearnedSearchTrial(
                params,
                finalist.feature_config,
                finalist.search_objective,
            )
            calibration_trials.append(trial)
            options.append((trial, metrics))
        calibrated_trial, calibration_metrics = max(
            options, key=lambda item: _calibration_selection_key(item[1])
        )
        calibrated_finalists.append(
            (calibrated_trial, calibration_metrics, finalist.search_objective)
        )
    selected, selected_calibration, _ = max(
        calibrated_finalists,
        key=lambda item: (
            item[2],
            *_calibration_selection_key(item[1]),
        ),
    )
    unvalidated_trials = tuple([*ranking_trials, *calibration_trials])
    best_search_targets = prepared(selected.feature_config, search_indices)
    search_budget_curves = build_development_budget_curves(
        best_search_targets, selected.params
    )
    search_passed, search_reasons, search_confirmation = _validation_confirmation(
        best_search_targets,
        selected.params,
        search_config.objective_profile,
        seed=search_config.seed,
    )
    search_confirmation["calibration"] = {
        "selectionOrder": ["meanLogLoss", "meanBrierScore", "topKECE"],
        "selected": selected_calibration,
        "finalistsEvaluated": len(calibrated_finalists),
        "gridSizePerFinalist": len(FULL_TEMPERATURES) * len(RANKING_UNIFORM_SHRINKAGES),
    }

    validation_evaluated = False
    validation_passed = False
    validation_reasons: tuple[str, ...] = ("Search未通过严格确认，Validation未打开",)
    validation_objective = None
    validation_confirmation: dict[str, object] | None = None
    validation_lock_fingerprint = None
    budget_curves: dict[str, object] = {"search": search_budget_curves}
    best = selected
    status = "search_rejected"
    if search_passed or not search_config.require_search_qualification:
        if search_config.require_search_qualification:
            if search_config.validation_lock_path is None:
                raise ValueError("正式Search必须配置一次性Validation锁路径")
            validation_lock_fingerprint = _claim_validation_lock(
                search_config.validation_lock_path,
                {
                    "schemaVersion": 1,
                    "lottery": rule.code,
                    "canonicalDataSha256": data_fingerprint,
                    "split": search_config.split.to_dict(),
                    "objectiveProfile": search_config.objective_profile,
                    "params": selected.params.to_dict(),
                    "featureConfig": {
                        "windows": list(selected.feature_config.windows),
                        "alpha": selected.feature_config.alpha,
                        "halfLife": selected.feature_config.half_life,
                        "omissionCap": selected.feature_config.omission_cap,
                        "windowWeights": dict(
                            selected.feature_config.window_weights or ()
                        ),
                    },
                },
            )
        best_validation_targets = prepared(selected.feature_config, validation_indices)
        validation_evaluated = True
        validation_objective = _profiled_objective(
            best_validation_targets,
            selected.params,
            search_config.objective_profile,
        )
        validation_budget_curves = build_development_budget_curves(
            best_validation_targets, selected.params
        )
        budget_curves["validation"] = validation_budget_curves
        (
            validation_passed,
            validation_reasons,
            validation_confirmation,
        ) = _validation_confirmation(
            best_validation_targets,
            selected.params,
            search_config.objective_profile,
            seed=search_config.seed + 1000,
        )
        best = replace(selected, validation_objective=validation_objective)
        status = "validation_complete"
    trials = tuple(
        (
            best
            if trial.params == selected.params
            and trial.feature_config == selected.feature_config
            else trial
        )
        for trial in unvalidated_trials
    )
    _write_progress_checkpoint(
        search_config.progress_checkpoint_path,
        {
            "status": status,
            "lottery": rule.code,
            "searchTargets": len(search_indices),
            "validationTargets": len(validation_indices),
            "selectedBy": "search_objective_only",
            "searchObjective": selected.search_objective,
            "validationObjective": validation_objective,
            "searchPassed": search_passed,
            "searchReasons": list(search_reasons),
            "validationEvaluated": validation_evaluated,
            "validationPassed": validation_passed,
            "validationReasons": list(validation_reasons),
        },
    )
    return LearnedSearchResult(
        params=best.params,
        feature_config=best.feature_config,
        search_objective=best.search_objective,
        validation_objective=validation_objective,
        split=search_config.split,
        trials=trials,
        selection_target_indices=search_indices,
        search_passed=search_passed,
        search_reasons=search_reasons,
        validation_evaluated=validation_evaluated,
        validation_passed=validation_passed,
        validation_reasons=validation_reasons,
        test_segment_used_for_selection=False,
        objective_profile=search_config.objective_profile,
        search_space_manifest=build_search_space_manifest(
            smoke=search_config.smoke,
            evaluated_feature_configs=evaluated_feature_configs,
            direct_top_k=search_config.direct_objective_top_k,
            group_top_k=search_config.group_objective_top_k,
            position_pool_size=search_config.position_objective_pool_size,
            incumbent_included=search_config.incumbent_params is not None,
        ),
        budget_curves=budget_curves,
        search_confirmation=search_confirmation,
        validation_confirmation=validation_confirmation,
        validation_lock_fingerprint=validation_lock_fingerprint,
    )
