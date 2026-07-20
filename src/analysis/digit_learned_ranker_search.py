# -*- coding: utf-8 -*-
"""三位彩固定评分 v4 的可复现参数搜索。"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, MutableMapping, Sequence

import numpy as np
import pandas as pd

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
from src.lotteries.base import LotteryRule

FULL_HALF_LIVES = (None, 20, 30, 50, 80, 100, 150, 200)
FULL_OMISSION_CAPS = (20, 30, 50, 80)
FULL_TEMPERATURES = (0.1, 0.2, 0.5, 1.0, 2.0)
FULL_UNIFORM_SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
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


def build_search_space_manifest(*, smoke: bool) -> dict[str, object]:
    """记录完整正式空间；smoke 只改变采样预算，不删减空间声明。"""

    space = {
        "halfLife": list(FULL_HALF_LIVES),
        "omissionCap": list(FULL_OMISSION_CAPS),
        "temperature": list(FULL_TEMPERATURES),
        "uniformShrinkage": list(FULL_UNIFORM_SHRINKAGES),
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
            "directTopK": [10],
            "groupTopK": [10],
            "positionPoolSize": [5],
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
        },
    }


def sample_feature_configs(
    *, seed: int, smoke: bool, limit: int | None = None
) -> tuple[LearnedFeatureConfig, ...]:
    """从完整空间做确定性有界采样，不预计算各配置特征矩阵。"""

    if smoke:
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
    order = rng.permutation(len(combinations))
    sample_size = min(limit or 16, len(combinations))
    selected = []
    for index in order[:sample_size]:
        windows, alpha, half_life, omission_cap, (_, weights) = combinations[int(index)]
        selected.append(
            LearnedFeatureConfig(
                windows=windows,
                alpha=alpha,
                half_life=half_life,
                omission_cap=omission_cap,
                window_weights={str(value): weights[str(value)] for value in windows},
            )
        )
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
    objective_profile: str = "balanced"
    direct_objective_top_k: int = 10
    group_objective_top_k: int = 10
    position_objective_pool_size: int = 5
    smoke: bool = False
    progress_checkpoint_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.min_train_size <= 0 or self.evaluation_stride <= 0:
            raise ValueError("最小训练期数和评估步长必须为正整数")
        if self.random_trials <= 0 or self.local_trials < 0:
            raise ValueError("随机搜索次数必须为正，局部搜索次数不得为负")
        if self.feature_configs is not None and not self.feature_configs:
            raise ValueError("feature_configs 不得为空")
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
    validation_objective: float
    split: LearnedSplit
    trials: tuple[LearnedSearchTrial, ...]
    selection_target_indices: tuple[int, ...]
    test_segment_used_for_selection: bool = False
    objective_profile: str = "balanced"
    search_space_manifest: dict[str, object] | None = None
    budget_curves: dict[str, object] | None = None

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
            "testSegmentUsedForSelection": self.test_segment_used_for_selection,
            "objectiveProfile": self.objective_profile,
            "budgetCurves": self.budget_curves or {},
            "searchSpaceManifest": self.search_space_manifest
            or build_search_space_manifest(smoke=False),
        }


@dataclass(frozen=True)
class _PreparedTarget:
    features: pd.DataFrame
    actual_text: str


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

    direct_hits = {budget: 0 for budget in DIRECT_BUDGETS}
    group_hits = {budget: 0 for budget in GROUP_BUDGETS}
    group_baselines: dict[int, list[float]] = {budget: [] for budget in GROUP_BUDGETS}
    position_hits = {budget: 0 for budget in POSITION_BUDGETS}
    for target in targets:
        texts = target.features["candidate"].astype(str).tolist()
        scores = score_candidates(target.features, params)
        probabilities = probabilities_from_scores(
            scores,
            temperature=params.temperature,
            uniform_shrinkage=params.uniform_shrinkage,
        )
        order = rank_candidate_indices(probabilities, texts)
        candidate_indices = {text: index for index, text in enumerate(texts)}
        actual_index = candidate_indices[target.actual_text]
        actual_rank = int(np.flatnonzero(order == actual_index)[0]) + 1
        for budget in DIRECT_BUDGETS:
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
        for budget in GROUP_BUDGETS:
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
        for budget in POSITION_BUDGETS:
            position_hits[budget] += sum(rank <= budget for rank in position_ranks)

    periods = len(targets)
    result: dict[str, Any] = {
        "direct": {
            str(budget): _budget_entry(direct_hits[budget], periods, budget / 1000.0)
            for budget in DIRECT_BUDGETS
        },
        "group": {
            str(budget): _budget_entry(
                group_hits[budget],
                periods,
                float(np.mean(group_baselines[budget])) if periods else 0.0,
            )
            for budget in GROUP_BUDGETS
        },
        "position": {
            str(budget): _budget_entry(
                position_hits[budget], periods * 3, budget / 10.0
            )
            for budget in POSITION_BUDGETS
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, values=values, actual=actual)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_prepared_cache(path: Path | None) -> tuple[_PreparedTarget, ...] | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        values = payload["values"]
        actual = payload["actual"].astype(str)
    if values.ndim != 3 or values.shape[1:] != (1000, len(FEATURE_NAMES)):
        raise ValueError(f"目标特征缓存结构错误：{path}")
    if len(actual) != len(values):
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
        )
        for matrix, actual_text in zip(values, actual)
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
        output.append(_PreparedTarget(features, actual_text))
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
        candidate_indices = {text: index for index, text in enumerate(texts)}
        try:
            actual_index = candidate_indices[target.actual_text]
        except KeyError as exc:
            raise ValueError(f"候选集合缺少真实号码：{target.actual_text}") from exc
        rank = int(np.flatnonzero(order == actual_index)[0]) + 1
        log_ranks.append(math.log(rank))
        rank_percentiles.append((rank - 1) / 999)
        actual_probability = float(probabilities[actual_index])
        log_losses.append(-math.log(max(actual_probability, 1e-300)))
        brier_scores.append(
            float(np.sum(probabilities**2) - 2 * actual_probability + 1.0)
        )
        top_confidences.append(float(probabilities[int(order[0])]))
        top_hits.append(int(order[0]) == actual_index)
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
        return float(
            0.40 * log_gain
            + 0.30 * brier_gain
            + 0.20 * rank_gain
            - 0.05 * ece
            - 0.05 * instability
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
        temperature=float(rng.choice(FULL_TEMPERATURES)),
        uniform_shrinkage=float(rng.choice(FULL_UNIFORM_SHRINKAGES)),
        group_aggregation=str(rng.choice(FULL_GROUP_AGGREGATIONS)),
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _local_params(
    base: LearnedRankerParams, rng: np.random.Generator
) -> LearnedRankerParams:
    weights = {
        name: float(np.clip(value + rng.normal(0.0, 0.08), -2.0, 2.0))
        for name, value in base.weights.items()
    }
    weights["constraint_penalty"] = -abs(weights["constraint_penalty"])
    temperatures = list(FULL_TEMPERATURES)
    current = min(
        range(len(temperatures)),
        key=lambda index: abs(temperatures[index] - base.temperature),
    )
    next_index = int(
        np.clip(current + int(rng.choice([-1, 0, 1])), 0, len(temperatures) - 1)
    )
    shrinkages = list(FULL_UNIFORM_SHRINKAGES)
    current_shrinkage = min(
        range(len(shrinkages)),
        key=lambda index: abs(shrinkages[index] - base.uniform_shrinkage),
    )
    next_shrinkage = int(
        np.clip(
            current_shrinkage + int(rng.choice([-1, 0, 1])),
            0,
            len(shrinkages) - 1,
        )
    )
    return replace(
        base,
        weights=weights,
        temperature=temperatures[next_index],
        uniform_shrinkage=shrinkages[next_shrinkage],
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _profiled_objective(
    targets: Sequence[_PreparedTarget],
    params: LearnedRankerParams,
    objective_profile: str,
) -> float:
    # 保持旧测试/插件 monkeypatch 的二参数兼容；balanced 是原 v4 目标。
    if objective_profile == "balanced":
        return _objective(targets, params)
    return _objective(targets, params, objective_profile=objective_profile)


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
    feature_configs = search_config.feature_configs or (search_config.feature_config,)
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

    rng = np.random.default_rng(search_config.seed)
    candidates = [
        LearnedRankerParams(),
        *[_random_params(rng) for _ in range(search_config.random_trials)],
    ]
    candidates = [
        replace(
            params,
            direct_top_k=search_config.direct_objective_top_k,
            group_top_k=search_config.group_objective_top_k,
            position_pool_size=search_config.position_objective_pool_size,
        )
        for params in candidates
    ]
    coarse: list[LearnedSearchTrial] = []
    completed_configs: list[dict[str, object]] = []
    for config_number, feature_config in enumerate(feature_configs, start=1):
        prepared_search = prepared(feature_config, search_indices)
        config_trials = [
            LearnedSearchTrial(
                params,
                feature_config,
                _profiled_objective(
                    prepared_search, params, search_config.objective_profile
                ),
            )
            for params in candidates
        ]
        coarse.extend(config_trials)
        config_best = max(config_trials, key=lambda item: item.search_objective)
        completed_configs.append(
            {
                "configNumber": config_number,
                "configTotal": len(feature_configs),
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
    local_params = [
        _local_params(search_best.params, rng)
        for _ in range(search_config.local_trials)
    ]
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
    unvalidated_trials = tuple([*coarse, *local])
    selected = max(unvalidated_trials, key=lambda item: item.search_objective)
    best_search_targets = prepared(selected.feature_config, search_indices)
    best_validation_targets = prepared(selected.feature_config, validation_indices)
    validation_objective = _profiled_objective(
        best_validation_targets, selected.params, search_config.objective_profile
    )
    best = replace(selected, validation_objective=validation_objective)
    trials = tuple(best if trial is selected else trial for trial in unvalidated_trials)
    _write_progress_checkpoint(
        search_config.progress_checkpoint_path,
        {
            "status": "validation_complete",
            "lottery": rule.code,
            "searchTargets": len(search_indices),
            "validationTargets": len(validation_indices),
            "selectedBy": "search_objective_only",
            "searchObjective": selected.search_objective,
            "validationObjective": validation_objective,
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
        test_segment_used_for_selection=False,
        objective_profile=search_config.objective_profile,
        search_space_manifest=build_search_space_manifest(smoke=search_config.smoke),
        budget_curves={
            "search": build_development_budget_curves(best_search_targets, best.params),
            "validation": build_development_budget_curves(
                best_validation_targets, best.params
            ),
        },
    )
