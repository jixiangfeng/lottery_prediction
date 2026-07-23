# -*- coding: utf-8 -*-
"""快乐8 v2 探索性特征发现；仅处理已隔离的开发区 DataFrame。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.analysis.kl8_pick5_probability_v1 import (
    canonical_kl8_sha256,
    normalize_kl8_dataframe,
    normalize_sum20,
)

FREQUENCY_WINDOWS = (5, 10, 20, 40, 80, 160, 320)
LAG_PERIODS = (1, 2, 3, 5, 10)
EWMA_HALF_LIVES = (10, 20, 80, 300)

FREQUENCY_FEATURES = tuple(f"frequency{window}" for window in FREQUENCY_WINDOWS)
OMISSION_FEATURES = ("omissionRaw", "omissionLog", "omissionCapped")
DYNAMIC_FEATURES = (
    *(f"lag{lag}" for lag in LAG_PERIODS),
    "trendFrequency10Minus80",
    "trendFrequency20Minus160",
    *(f"ewma{half_life}" for half_life in EWMA_HALF_LIVES),
)
PREVIOUS_CONTEXT_FEATURES = (
    "inPrevious",
    "adjacentToPreviousIndicator",
    "adjacentToPreviousCount",
    "sameTailCount",
    "sameDecadeCount",
)
PAIR_CONTEXT_FEATURES = ("pairContextLift80",)
ALL_FEATURE_COLUMNS = (
    *FREQUENCY_FEATURES,
    *OMISSION_FEATURES,
    *DYNAMIC_FEATURES,
    *PREVIOUS_CONTEXT_FEATURES,
    *PAIR_CONTEXT_FEATURES,
)

CANDIDATE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "frequency": FREQUENCY_FEATURES,
    "frequency+omission": (*FREQUENCY_FEATURES, *OMISSION_FEATURES),
    "frequency+omission+lags_trends_ewma": (
        *FREQUENCY_FEATURES,
        *OMISSION_FEATURES,
        *DYNAMIC_FEATURES,
    ),
    "frequency+omission+lags_trends_ewma+previous_context": (
        *FREQUENCY_FEATURES,
        *OMISSION_FEATURES,
        *DYNAMIC_FEATURES,
        *PREVIOUS_CONTEXT_FEATURES,
    ),
    "full": ALL_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class Kl8FeatureDiscoveryConfig:
    """v2 固定分段与保守 LightGBM 配置。"""

    initial_train: int = 300
    search_periods: int = 714
    evaluation_periods: int = 500
    refit_interval: int = 50
    stability_blocks: int = 5
    seed: int = 20260723
    epsilon: float = 1e-6
    n_estimators: int = 60
    learning_rate: float = 0.04
    num_leaves: int = 7
    max_depth: int = 3
    min_child_samples: int = 100
    reg_alpha: float = 0.2
    reg_lambda: float = 1.0
    n_jobs: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "initial_train": self.initial_train,
            "search_periods": self.search_periods,
            "evaluation_periods": self.evaluation_periods,
            "stability_blocks": self.stability_blocks,
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "n_jobs": self.n_jobs,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                raise ValueError(f"{name}必须为整数")
        if (
            self.initial_train <= 0
            or self.search_periods <= 0
            or self.evaluation_periods <= 0
        ):
            raise ValueError(
                "initial_train、search_periods、evaluation_periods必须为正整数"
            )
        if self.refit_interval != 50:
            raise ValueError("refit_interval必须固定为50")
        if self.stability_blocks != 5:
            raise ValueError("stability_blocks必须固定为5")
        if self.n_estimators <= 0 or self.num_leaves < 2 or self.min_child_samples <= 0:
            raise ValueError("LightGBM模型规模参数无效")
        if self.max_depth == 0 or self.max_depth < -1:
            raise ValueError("max_depth必须为-1或正整数")
        if self.n_jobs == 0:
            raise ValueError("n_jobs不得为0")
        finite_fields: dict[str, float] = {
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }
        for name, numeric_value in finite_fields.items():
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name}必须为有限数")
        if not 0.0 < self.epsilon < 0.25 or self.learning_rate <= 0.0:
            raise ValueError("epsilon或learning_rate范围无效")
        if self.reg_alpha < 0.0 or self.reg_lambda < 0.0:
            raise ValueError("正则化参数不得为负数")

    @property
    def required_periods(self) -> int:
        """返回分段所需开发期数。"""

        return self.initial_train + self.search_periods + self.evaluation_periods

    def validate_history_length(self, history_length: int) -> None:
        """要求三段长度之和恰好等于开发历史长度。"""

        if history_length != self.required_periods:
            raise ValueError(
                f"分段总长度必须恰好等于开发历史：{self.required_periods}!={history_length}"
            )


def fixed_config_for_development_length(
    history_length: int, *, n_jobs: int
) -> Kl8FeatureDiscoveryConfig:
    """返回实际1514期开发区的唯一固定探索配置。"""

    config = Kl8FeatureDiscoveryConfig(n_jobs=n_jobs)
    config.validate_history_length(history_length)
    return config


def _outcome_matrix(development: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(development, pd.DataFrame):
        raise TypeError("v2只接受已隔离的开发DataFrame，不接受路径或Frozen数据源")
    normalized = normalize_kl8_dataframe(development)
    if normalized.empty:
        raise ValueError("开发DataFrame不得为空")
    outcomes = np.zeros((len(normalized), 80), dtype=np.float64)
    for period_index, numbers in enumerate(normalized["numbers"]):
        outcomes[period_index, np.asarray(numbers, dtype=np.int64) - 1] = 1.0
    return normalized, outcomes


def _pair_lift_against_previous(
    pair_counts: np.ndarray,
    marginal_counts: np.ndarray,
    previous: np.ndarray,
    window_length: int,
) -> np.ndarray:
    if window_length == 0 or not np.any(previous):
        return np.zeros(80, dtype=np.float64)
    previous_indexes = np.flatnonzero(previous)
    numerator = (pair_counts[:, previous_indexes] + 0.25) * (window_length + 1.0)
    denominator = (marginal_counts[:, None] + 0.5) * (
        marginal_counts[previous_indexes][None, :] + 0.5
    )
    lifts = numerator / denominator - 1.0
    candidate_indexes = np.arange(80)[:, None]
    valid = candidate_indexes != previous_indexes[None, :]
    totals = np.where(valid, lifts, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    return np.divide(totals, counts, out=np.zeros(80), where=counts > 0).astype(
        np.float64
    )


def build_prior_only_number_panel(development: pd.DataFrame) -> pd.DataFrame:
    """构建期号×号码面板；目标期特征只依赖目标期之前的开奖。"""

    normalized, outcomes = _outcome_matrix(development)
    periods = len(normalized)
    cumulative = np.vstack(
        [np.zeros((1, 80), dtype=np.float64), np.cumsum(outcomes, axis=0)]
    )
    last_seen = np.full(80, -1, dtype=np.int64)
    ewma = {
        half_life: np.full(80, 0.25, dtype=np.float64) for half_life in EWMA_HALF_LIVES
    }
    ewma_alpha = {
        half_life: 1.0 - math.exp(math.log(0.5) / half_life)
        for half_life in EWMA_HALF_LIVES
    }
    pair_counts = np.zeros((80, 80), dtype=np.float64)
    pair_marginals = np.zeros(80, dtype=np.float64)
    feature_arrays = {
        name: np.empty((periods, 80), dtype=np.float64) for name in ALL_FEATURE_COLUMNS
    }

    numbers = np.arange(1, 81, dtype=np.int64)
    tails = numbers % 10
    decades = (numbers - 1) // 10
    for period_index in range(periods):
        for window in FREQUENCY_WINDOWS:
            start = max(0, period_index - window)
            observed = period_index - start
            frequency = (
                (cumulative[period_index] - cumulative[start]) / observed
                if observed
                else np.full(80, 0.25, dtype=np.float64)
            )
            feature_arrays[f"frequency{window}"][period_index] = frequency

        omission = np.where(last_seen >= 0, period_index - last_seen - 1, period_index)
        omission_float = omission.astype(np.float64)
        feature_arrays["omissionRaw"][period_index] = omission_float
        feature_arrays["omissionLog"][period_index] = np.log1p(omission_float)
        feature_arrays["omissionCapped"][period_index] = np.minimum(
            omission_float, 320.0
        )
        for lag in LAG_PERIODS:
            feature_arrays[f"lag{lag}"][period_index] = (
                outcomes[period_index - lag]
                if period_index >= lag
                else np.zeros(80, dtype=np.float64)
            )
        feature_arrays["trendFrequency10Minus80"][period_index] = (
            feature_arrays["frequency10"][period_index]
            - feature_arrays["frequency80"][period_index]
        )
        feature_arrays["trendFrequency20Minus160"][period_index] = (
            feature_arrays["frequency20"][period_index]
            - feature_arrays["frequency160"][period_index]
        )
        for half_life in EWMA_HALF_LIVES:
            feature_arrays[f"ewma{half_life}"][period_index] = ewma[half_life]

        previous = (
            outcomes[period_index - 1]
            if period_index > 0
            else np.zeros(80, dtype=np.float64)
        )
        previous_numbers = np.flatnonzero(previous) + 1
        adjacent_counts = (
            np.isin(numbers[:, None] - previous_numbers[None, :], (-1, 1)).sum(axis=1)
            if len(previous_numbers)
            else np.zeros(80, dtype=np.int64)
        )
        feature_arrays["inPrevious"][period_index] = previous
        feature_arrays["adjacentToPreviousCount"][period_index] = adjacent_counts
        feature_arrays["adjacentToPreviousIndicator"][period_index] = (
            adjacent_counts > 0
        )
        feature_arrays["sameTailCount"][period_index] = (
            (tails[:, None] == (previous_numbers % 10)[None, :]).sum(axis=1)
            if len(previous_numbers)
            else np.zeros(80, dtype=np.int64)
        )
        feature_arrays["sameDecadeCount"][period_index] = (
            (decades[:, None] == ((previous_numbers - 1) // 10)[None, :]).sum(axis=1)
            if len(previous_numbers)
            else np.zeros(80, dtype=np.int64)
        )
        feature_arrays["pairContextLift80"][period_index] = _pair_lift_against_previous(
            pair_counts,
            pair_marginals,
            previous,
            min(period_index, 80),
        )

        current = outcomes[period_index]
        for half_life in EWMA_HALF_LIVES:
            alpha = ewma_alpha[half_life]
            ewma[half_life] = (1.0 - alpha) * ewma[half_life] + alpha * current
        present = np.flatnonzero(current)
        pair_counts += np.outer(current, current)
        pair_marginals += current
        if period_index >= 80:
            expired = outcomes[period_index - 80]
            pair_counts -= np.outer(expired, expired)
            pair_marginals -= expired
        last_seen[present] = period_index

    panel = pd.DataFrame(
        {
            "periodIndex": np.repeat(np.arange(periods, dtype=np.int64), 80),
            "issue": np.repeat(normalized["issue"].to_numpy(), 80),
            "number": np.tile(np.arange(1, 81, dtype=np.int64), periods),
            "target": outcomes.reshape(-1).astype(np.float64),
        }
    )
    for name in ALL_FEATURE_COLUMNS:
        panel[name] = feature_arrays[name].reshape(-1).astype(np.float64)
    if not np.isfinite(panel[list(ALL_FEATURE_COLUMNS)].to_numpy()).all():
        raise ValueError("v2特征面板包含非有限值")
    return panel


def _model(config: Kl8FeatureDiscoveryConfig) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=config.seed,
        bagging_seed=config.seed,
        feature_fraction_seed=config.seed,
        data_random_seed=config.seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )


def _period_scores(
    probabilities: np.ndarray, targets: np.ndarray, *, epsilon: float
) -> tuple[float, float, float]:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    log_loss = -float(
        np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
    )
    brier = float(np.mean(np.square(clipped - targets)))
    top_indexes = np.lexsort((np.arange(80), -clipped))[:20]
    hits = float(targets[top_indexes].sum())
    return log_loss, brier, hits


def _summarize_predictions(
    period_records: Sequence[dict[str, float]], *, stability_blocks: int
) -> dict[str, object]:
    if not period_records:
        raise ValueError("评估窗口不得为空")
    log_loss = float(np.mean([record["logLoss"] for record in period_records]))
    brier = float(np.mean([record["brier"] for record in period_records]))
    top20_hits = float(np.mean([record["top20Hits"] for record in period_records]))
    uniform_log_loss = float(
        np.mean([record["uniformLogLoss"] for record in period_records])
    )
    uniform_brier = float(
        np.mean([record["uniformBrier"] for record in period_records])
    )
    uniform_top20_hits = float(
        np.mean([record["uniformTop20Hits"] for record in period_records])
    )
    blocks = []
    for block_index, indexes in enumerate(
        np.array_split(np.arange(len(period_records)), stability_blocks), start=1
    ):
        records = [period_records[int(index)] for index in indexes]
        block_log_loss = float(np.mean([record["logLoss"] for record in records]))
        block_brier = float(np.mean([record["brier"] for record in records]))
        block_uniform_log_loss = float(
            np.mean([record["uniformLogLoss"] for record in records])
        )
        block_uniform_brier = float(
            np.mean([record["uniformBrier"] for record in records])
        )
        blocks.append(
            {
                "block": block_index,
                "periods": len(records),
                "deltaLogLoss": block_uniform_log_loss - block_log_loss,
                "deltaBrier": block_uniform_brier - block_brier,
            }
        )
    return {
        "periods": len(period_records),
        "logLoss": log_loss,
        "brier": brier,
        "top20MeanHits": top20_hits,
        "uniformLogLoss": uniform_log_loss,
        "uniformBrier": uniform_brier,
        "uniformTop20MeanHits": uniform_top20_hits,
        "deltaLogLoss": uniform_log_loss - log_loss,
        "deltaBrier": uniform_brier - brier,
        "deltaTop20MeanHits": top20_hits - uniform_top20_hits,
        "blocks": blocks,
    }


def _evaluate_window(
    panel: pd.DataFrame,
    *,
    start_period: int,
    end_period: int,
    features: Sequence[str] | None,
    config: Kl8FeatureDiscoveryConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    records: list[dict[str, float]] = []
    gain_totals = {feature: 0.0 for feature in features or ()}
    train_count = 0
    classifier: lgb.LGBMClassifier | None = None
    uniform = np.full(80, 0.25, dtype=np.float64)
    for period_index in range(start_period, end_period):
        if features is not None and (
            classifier is None
            or (period_index - start_period) % config.refit_interval == 0
        ):
            train_rows = panel["periodIndex"] < period_index
            classifier = _model(config)
            classifier.fit(
                panel.loc[train_rows, list(features)],
                panel.loc[train_rows, "target"],
            )
            gains = classifier.booster_.feature_importance(importance_type="gain")
            for feature, gain in zip(features, gains, strict=True):
                gain_totals[feature] += float(gain)
            train_count += 1
        target_rows = panel["periodIndex"] == period_index
        targets = panel.loc[target_rows, "target"].to_numpy(dtype=np.float64)
        if features is None:
            probabilities = uniform
        else:
            if classifier is None:
                raise RuntimeError("LightGBM未按计划完成首次拟合")
            raw_predictions = np.asarray(
                classifier.predict_proba(panel.loc[target_rows, list(features)]),
                dtype=np.float64,
            )
            raw = raw_predictions[:, 1]
            probabilities = normalize_sum20(raw, epsilon=config.epsilon)
        log_loss, brier, top20_hits = _period_scores(
            probabilities, targets, epsilon=config.epsilon
        )
        uniform_log_loss, uniform_brier, uniform_top20_hits = _period_scores(
            uniform, targets, epsilon=config.epsilon
        )
        records.append(
            {
                "logLoss": log_loss,
                "brier": brier,
                "top20Hits": top20_hits,
                "uniformLogLoss": uniform_log_loss,
                "uniformBrier": uniform_brier,
                "uniformTop20Hits": uniform_top20_hits,
            }
        )
    summary = _summarize_predictions(records, stability_blocks=config.stability_blocks)
    total_gain = float(sum(gain_totals.values()))
    summary.update(
        {
            "elapsedSeconds": time.perf_counter() - started,
            "trainCount": train_count,
            "featureGainImportance": [
                {
                    "feature": feature,
                    "gain": gain_totals[feature],
                    "gainShare": (
                        gain_totals[feature] / total_gain if total_gain > 0.0 else 0.0
                    ),
                }
                for feature in sorted(
                    gain_totals,
                    key=lambda name: (-gain_totals[name], name),
                )
            ],
        }
    )
    return summary


def _validate_candidate_sets(
    candidate_feature_sets: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    validated: dict[str, tuple[str, ...]] = {}
    known = set(ALL_FEATURE_COLUMNS)
    for name, features in candidate_feature_sets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("候选特征集名称必须为非空字符串")
        sequence = tuple(features)
        if not sequence or len(sequence) != len(set(sequence)):
            raise ValueError(f"候选特征集{name}必须非空且不得重复")
        unknown = set(sequence) - known
        if unknown:
            raise ValueError(f"候选特征集{name}包含未知特征：{sorted(unknown)}")
        validated[name] = sequence
    return validated


def _float_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}必须为数值")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label}必须为有限数")
    return output


def _int_value(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label}必须为整数")
    return cast(int, value)


def _eligible(metrics: Mapping[str, object]) -> bool:
    blocks = metrics["blocks"]
    if not isinstance(blocks, list) or len(blocks) != 5:
        return False
    return bool(
        _float_value(metrics["deltaLogLoss"], "deltaLogLoss") >= 0.0
        and _float_value(metrics["deltaBrier"], "deltaBrier") >= 0.0
        and all(
            isinstance(block, Mapping)
            and _float_value(block["deltaLogLoss"], "block.deltaLogLoss") >= 0.0
            and _float_value(block["deltaBrier"], "block.deltaBrier") >= 0.0
            for block in blocks
        )
    )


def _selection_key(result: Mapping[str, object]) -> tuple[float, float, int, str]:
    return (
        -_float_value(result["deltaLogLoss"], "deltaLogLoss"),
        -_float_value(result["deltaBrier"], "deltaBrier"),
        _int_value(result["featureCount"], "featureCount"),
        str(result["name"]),
    )


def _source_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (Path(__file__), root / "scripts" / "kl8_feature_discovery_v2.py")


def source_fingerprint() -> str:
    """返回v2核心与CLI源码指纹。"""

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in _source_paths():
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def run_kl8_feature_discovery_v2(
    development: pd.DataFrame,
    config: Kl8FeatureDiscoveryConfig,
    *,
    frozen_periods_excluded: int,
    frozen_boundary: Mapping[str, object],
    candidate_feature_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """仅在开发区执行Search选择，并对胜者做一次最终Evaluation。"""

    normalized, _ = _outcome_matrix(development)
    config.validate_history_length(len(normalized))
    if frozen_periods_excluded <= 0:
        raise ValueError("必须显式排除至少1期Frozen")
    candidates = _validate_candidate_sets(
        CANDIDATE_FEATURE_SETS
        if candidate_feature_sets is None
        else candidate_feature_sets
    )
    panel = build_prior_only_number_panel(normalized)
    search_start = config.initial_train
    search_end = search_start + config.search_periods
    search_results: list[dict[str, object]] = []
    for name, features in candidates.items():
        metrics = _evaluate_window(
            panel,
            start_period=search_start,
            end_period=search_end,
            features=features,
            config=config,
        )
        search_results.append(
            {
                "name": name,
                "featureCount": len(features),
                "features": list(features),
                "eligible": _eligible(metrics),
                **metrics,
            }
        )
    eligible = [result for result in search_results if result["eligible"]]
    if eligible:
        selected = sorted(eligible, key=_selection_key)[0]
        selected_name = str(selected["name"])
        selected_features: Sequence[str] | None = candidates[selected_name]
        selection_reason = (
            "eligible_max_delta_logloss_then_brier_then_fewer_features_then_name"
        )
    else:
        selected_name = "uniform"
        selected_features = None
        selection_reason = "no_eligible_feature_set"
    evaluation = _evaluate_window(
        panel,
        start_period=search_end,
        end_period=config.required_periods,
        features=selected_features,
        config=config,
    )
    first = normalized.iloc[0]
    last = normalized.iloc[-1]
    report: dict[str, object] = {
        "schemaVersion": "kl8_feature_discovery_v2",
        "evidenceStatus": "exploratory_feature_discovery_only",
        "frozenRead": False,
        "promotionPassed": False,
        "recommendationEnabled": False,
        "userVisibleCandidates": [],
        "selectedFeatureSet": selected_name,
        "selectionReason": selection_reason,
        "searchCandidates": search_results,
        "evaluation": evaluation,
        "segmentation": {
            "initialTrain": config.initial_train,
            "search": config.search_periods,
            "evaluation": config.evaluation_periods,
            "refitInterval": config.refit_interval,
        },
        "boundaries": {
            "development": {
                "periods": len(normalized),
                "firstIssue": str(first["issue"]),
                "lastIssue": str(last["issue"]),
                "firstDate": str(first["date"]),
                "lastDate": str(last["date"]),
            },
            "frozen": {
                "periodsExcluded": frozen_periods_excluded,
                "firstIssue": str(frozen_boundary["firstIssue"]),
                "lastIssue": str(frozen_boundary["lastIssue"]),
                "numbersRead": False,
            },
        },
        "dataSha256": canonical_kl8_sha256(normalized),
        "sourceFingerprint": source_fingerprint(),
        "config": asdict(config),
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


def _write_immutable_json(payload: object, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() == content:
            return path
        raise FileExistsError(f"报告已存在且为不同内容，拒绝覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"报告已存在且为不同内容，拒绝覆盖：{path}")
        return path
    finally:
        temporary.unlink(missing_ok=True)


def write_kl8_feature_discovery_report(
    report: Mapping[str, object], path: str | Path
) -> Path:
    """原子且不可变地写入探索性v2报告。"""

    return _write_immutable_json(dict(report), path)


__all__ = [
    "ALL_FEATURE_COLUMNS",
    "CANDIDATE_FEATURE_SETS",
    "Kl8FeatureDiscoveryConfig",
    "build_prior_only_number_panel",
    "fixed_config_for_development_length",
    "run_kl8_feature_discovery_v2",
    "source_fingerprint",
    "write_kl8_feature_discovery_report",
]
