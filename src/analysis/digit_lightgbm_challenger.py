# -*- coding: utf-8 -*-
"""LightGBM三位置多分类挑战模型及全部连续区块回测。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import binom

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_ranker import learned_ranker_source_fingerprint
from src.lotteries.base import LotteryRule

_UNIFORM_LOG_LOSS = math.log(1000.0)
_UNIFORM_BRIER = 0.999


@dataclass(frozen=True)
class LightGBMParams:
    name: str
    num_leaves: int
    max_depth: int
    min_child_samples: int
    n_estimators: int
    learning_rate: float = 0.03

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "numLeaves": self.num_leaves,
            "maxDepth": self.max_depth,
            "minChildSamples": self.min_child_samples,
            "nEstimators": self.n_estimators,
            "learningRate": self.learning_rate,
        }


_DEFAULT_GRID = (
    LightGBMParams("shallow", 7, 3, 200, 120),
    LightGBMParams("compact", 15, 4, 150, 160),
    LightGBMParams("slow", 7, 5, 250, 220, 0.02),
)


@dataclass(frozen=True)
class LightGBMChallengeConfig:
    windows: tuple[int, ...] = (20, 50, 150, 300, 500)
    lag_count: int = 10
    minimum_train_periods: int = 500
    inner_validation_periods: int = 500
    block_size: int = 500
    top_k: int = 50
    parameter_grid: tuple[LightGBMParams, ...] = _DEFAULT_GRID
    shrinkages: tuple[float, ...] = (0.25, 0.5, 0.75)
    random_seed: int = 20260720

    def __post_init__(self) -> None:
        if not self.windows or min(self.windows) <= 0:
            raise ValueError("windows必须是正整数")
        if self.lag_count <= 0 or self.minimum_train_periods <= 0:
            raise ValueError("lag_count和minimum_train_periods必须大于零")
        if self.inner_validation_periods <= 0 or self.block_size <= 0:
            raise ValueError("Validation和区块大小必须大于零")
        if not self.parameter_grid or not self.shrinkages:
            raise ValueError("参数网格和收缩候选不能为空")
        if any(value <= 0 or value > 1 for value in self.shrinkages):
            raise ValueError("LightGBM收缩系数必须在(0,1]内")


@dataclass(frozen=True)
class LightGBMFeatureTable:
    features: np.ndarray
    targets: np.ndarray
    target_indices: tuple[int, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class LightGBMBlockBacktestResult:
    lottery: str
    config: LightGBMChallengeConfig
    feature_count: int
    first_block_start: int
    blocks: tuple[dict[str, Any], ...]
    data_sha256: str
    source_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        periods = sum(int(block["periods"]) for block in self.blocks)
        hits = sum(int(block["top50Hits"]) for block in self.blocks)
        mean_log_loss = float(
            np.average(
                [float(block["meanLogLoss"]) for block in self.blocks],
                weights=[int(block["periods"]) for block in self.blocks],
            )
        )
        mean_brier = float(
            np.average(
                [float(block["meanBrier"]) for block in self.blocks],
                weights=[int(block["periods"]) for block in self.blocks],
            )
        )
        blocks_above = sum(
            float(block["top50HitRate"]) >= 0.05 for block in self.blocks
        )
        joint_blocks = sum(bool(block["jointGatePassed"]) for block in self.blocks)
        pooled_p = float(binom.sf(hits - 1, periods, self.config.top_k / 1000.0))
        stable_required = math.ceil(len(self.blocks) * 0.7)
        passed = (
            hits / periods > self.config.top_k / 1000.0
            and pooled_p < 0.05
            and mean_log_loss < _UNIFORM_LOG_LOSS
            and mean_brier < _UNIFORM_BRIER
            and blocks_above >= stable_required
            and joint_blocks >= 2
        )
        return {
            "modelVersion": "lightgbm_position_multiclass_v1",
            "evaluationKind": "lightgbm_position_multiclass_blocks",
            "evidenceStatus": "retrospective_challenger",
            "lottery": self.lottery,
            "selectionPolicy": "inner_validation_only",
            "blockSelectionAllowed": False,
            "topK": self.config.top_k,
            "blockSize": self.config.block_size,
            "firstBlockStart": self.first_block_start,
            "blocksEvaluated": len(self.blocks),
            "periodsEvaluated": periods,
            "featureCount": self.feature_count,
            "dataSha256": self.data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "config": {
                "windows": list(self.config.windows),
                "lagCount": self.config.lag_count,
                "minimumTrainPeriods": self.config.minimum_train_periods,
                "innerValidationPeriods": self.config.inner_validation_periods,
                "parameterGrid": [
                    item.to_dict() for item in self.config.parameter_grid
                ],
                "shrinkages": list(self.config.shrinkages),
            },
            "summary": {
                "top50Hits": hits,
                "top50HitRate": hits / periods,
                "top50PValue": pooled_p,
                "meanLogLoss": mean_log_loss,
                "uniformLogLoss": _UNIFORM_LOG_LOSS,
                "meanBrier": mean_brier,
                "uniformBrier": _UNIFORM_BRIER,
                "blocksAtOrAboveBaseline": blocks_above,
                "stableBlocksRequired": stable_required,
                "jointGateBlocks": joint_blocks,
            },
            "gatePassed": passed,
            "formalPredictionActivated": False,
            "blocks": list(self.blocks),
        }


def _canonical_history(history: pd.DataFrame, rule: LotteryRule) -> pd.DataFrame:
    return sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )


def _one_hot_lags(
    values: np.ndarray, lag_count: int
) -> tuple[list[np.ndarray], list[str]]:
    rows, positions = values.shape
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for position in range(positions):
        for lag in range(1, lag_count + 1):
            shifted = np.full(rows, -1, dtype=np.int16)
            shifted[lag:] = values[:-lag, position]
            for digit in range(10):
                arrays.append((shifted == digit).astype(np.float32))
                names.append(f"p{position}_lag{lag}_d{digit}")
    return arrays, names


def _rolling_frequencies(
    values: np.ndarray, windows: tuple[int, ...]
) -> tuple[list[np.ndarray], list[str]]:
    rows, positions = values.shape
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for position in range(positions):
        for digit in range(10):
            indicator = (values[:, position] == digit).astype(np.float32)
            cumulative = np.concatenate(([0.0], np.cumsum(indicator, dtype=float)))
            for window in windows:
                indices = np.arange(rows)
                starts = np.maximum(0, indices - window)
                counts = cumulative[indices] - cumulative[starts]
                denominators = np.maximum(1, indices - starts)
                arrays.append((counts / denominators).astype(np.float32))
                names.append(f"p{position}_freq_w{window}_d{digit}")
    return arrays, names


def _omissions(values: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    rows, positions = values.shape
    output = np.empty((rows, positions, 10), dtype=np.float32)
    gaps = np.zeros((positions, 10), dtype=np.float32)
    for index in range(rows):
        output[index] = gaps
        gaps += 1.0
        for position in range(positions):
            gaps[position, values[index, position]] = 0.0
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for position in range(positions):
        for digit in range(10):
            arrays.append(np.log1p(output[:, position, digit]))
            names.append(f"p{position}_omission_d{digit}")
    return arrays, names


def _shape_lags(values: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    sums = values.sum(axis=1)
    spans = values.max(axis=1) - values.min(axis=1)
    unique = np.array([len(set(row.tolist())) for row in values])
    shifted_sum = np.concatenate(([-1], sums[:-1]))
    shifted_span = np.concatenate(([-1], spans[:-1]))
    shifted_shape = np.concatenate(([-1], unique[:-1]))
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for value in range(28):
        arrays.append((shifted_sum == value).astype(np.float32))
        names.append(f"last_sum_{value}")
    for value in range(10):
        arrays.append((shifted_span == value).astype(np.float32))
        names.append(f"last_span_{value}")
    for value in (1, 2, 3):
        arrays.append((shifted_shape == value).astype(np.float32))
        names.append(f"last_unique_{value}")
    return arrays, names


def build_lightgbm_feature_table(
    history: pd.DataFrame,
    rule: LotteryRule,
    *,
    windows: tuple[int, ...] = (20, 50, 150, 300, 500),
    lag_count: int = 10,
) -> LightGBMFeatureTable:
    chronological = _canonical_history(history, rule)
    values = chronological[list(rule.number_columns)].to_numpy(dtype=np.int16)
    feature_start = max(max(windows), lag_count)
    if len(values) <= feature_start:
        raise ValueError("历史不足以构建LightGBM特征")
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for builder in (
        lambda: _one_hot_lags(values, lag_count),
        lambda: _rolling_frequencies(values, windows),
        lambda: _omissions(values),
        lambda: _shape_lags(values),
    ):
        built_arrays, built_names = builder()
        arrays.extend(built_arrays)
        names.extend(built_names)
    matrix = np.column_stack(arrays).astype(np.float32, copy=False)
    target_indices = tuple(range(feature_start, len(values)))
    return LightGBMFeatureTable(
        features=matrix[feature_start:],
        targets=values[feature_start:],
        target_indices=target_indices,
        feature_names=tuple(names),
    )


def joint_digit_probabilities(position_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(position_probabilities, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (3, 10):
        raise ValueError("位置概率必须是[n,3,10]")
    joint = (
        probabilities[:, 0, :, None, None]
        * probabilities[:, 1, None, :, None]
        * probabilities[:, 2, None, None, :]
    ).reshape(len(probabilities), 1000)
    totals = joint.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("联合概率和必须大于零")
    return joint / totals


def _fit_position_models(
    features: np.ndarray,
    targets: np.ndarray,
    params: LightGBMParams,
    seed: int,
) -> tuple[LGBMClassifier, ...]:
    models: list[LGBMClassifier] = []
    for position in range(3):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=10,
            num_leaves=params.num_leaves,
            max_depth=params.max_depth,
            min_child_samples=params.min_child_samples,
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            reg_alpha=1.0,
            reg_lambda=5.0,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed + position,
            n_jobs=4,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(features, targets[:, position])
        models.append(model)
    return tuple(models)


def _predict_position_probabilities(
    models: tuple[LGBMClassifier, ...], features: np.ndarray
) -> np.ndarray:
    output = np.empty((len(features), 3, 10), dtype=float)
    for position, model in enumerate(models):
        predicted = np.asarray(model.predict_proba(features), dtype=float)
        output[:, position, :] = 0.0
        for column, class_value in enumerate(model.classes_):
            output[:, position, int(class_value)] = predicted[:, column]
    output /= output.sum(axis=2, keepdims=True)
    return output


def _actual_indices(targets: np.ndarray) -> np.ndarray:
    return targets[:, 0] * 100 + targets[:, 1] * 10 + targets[:, 2]


def _metrics(
    probabilities: np.ndarray, actual: np.ndarray, top_k: int
) -> dict[str, Any]:
    row_indices = np.arange(len(actual))
    actual_probability = probabilities[row_indices, actual]
    log_losses = -np.log(np.clip(actual_probability, 1e-15, 1.0))
    brier = (
        np.sum(probabilities * probabilities, axis=1) + 1.0 - 2.0 * actual_probability
    )
    candidate_indices = np.arange(1000)
    hits = 0
    ranks: list[int] = []
    for row, candidate in zip(probabilities, actual):
        actual_value = row[candidate]
        rank = (
            1
            + int(np.sum(row > actual_value))
            + int(np.sum((row == actual_value) & (candidate_indices < candidate)))
        )
        ranks.append(rank)
        hits += rank <= top_k
    return {
        "periods": len(actual),
        "meanLogLoss": float(np.mean(log_losses)),
        "meanBrier": float(np.mean(brier)),
        "meanRank": float(np.mean(ranks)),
        "top50Hits": hits,
        "top50HitRate": hits / len(actual),
        "top50PValue": float(binom.sf(hits - 1, len(actual), top_k / 1000.0)),
    }


def _shrink(probabilities: np.ndarray, value: float) -> np.ndarray:
    return value * probabilities + (1.0 - value) / 1000.0


def run_lightgbm_block_backtest(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: LightGBMChallengeConfig = LightGBMChallengeConfig(),
) -> LightGBMBlockBacktestResult:
    chronological = _canonical_history(history, rule)
    table = build_lightgbm_feature_table(
        chronological, rule, windows=config.windows, lag_count=config.lag_count
    )
    feature_start = table.target_indices[0]
    first_block_start = (
        feature_start + config.minimum_train_periods + config.inner_validation_periods
    )
    block_count = (len(chronological) - first_block_start) // config.block_size
    if block_count <= 0:
        raise ValueError("历史不足以形成LightGBM外层测试块")
    target_array = np.asarray(table.target_indices)
    blocks: list[dict[str, Any]] = []
    for block_index in range(block_count):
        block_start = first_block_start + block_index * config.block_size
        block_end = block_start + config.block_size
        validation_start = block_start - config.inner_validation_periods
        train_mask = target_array < validation_start
        validation_mask = (target_array >= validation_start) & (
            target_array < block_start
        )
        block_mask = (target_array >= block_start) & (target_array < block_end)
        if int(train_mask.sum()) < config.minimum_train_periods:
            raise RuntimeError("内部训练样本不足")
        best_key: tuple[float, float, str, float] | None = None
        best_params: LightGBMParams | None = None
        best_shrinkage: float | None = None
        selection_candidates: list[dict[str, Any]] = []
        for params in config.parameter_grid:
            models = _fit_position_models(
                table.features[train_mask],
                table.targets[train_mask],
                params,
                config.random_seed,
            )
            joint = joint_digit_probabilities(
                _predict_position_probabilities(models, table.features[validation_mask])
            )
            actual = _actual_indices(table.targets[validation_mask])
            for shrinkage in config.shrinkages:
                metrics = _metrics(_shrink(joint, shrinkage), actual, config.top_k)
                selection_candidates.append(
                    {
                        "params": params.name,
                        "shrinkage": shrinkage,
                        "meanLogLoss": metrics["meanLogLoss"],
                        "meanBrier": metrics["meanBrier"],
                        "top50Hits": metrics["top50Hits"],
                    }
                )
                key = (
                    float(metrics["meanLogLoss"]),
                    float(metrics["meanBrier"]),
                    params.name,
                    shrinkage,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_params = params
                    best_shrinkage = shrinkage
        if best_params is None or best_shrinkage is None:
            raise RuntimeError("LightGBM内部选择失败")
        fit_mask = target_array < block_start
        final_models = _fit_position_models(
            table.features[fit_mask],
            table.targets[fit_mask],
            best_params,
            config.random_seed,
        )
        block_joint = joint_digit_probabilities(
            _predict_position_probabilities(final_models, table.features[block_mask])
        )
        metrics = _metrics(
            _shrink(block_joint, best_shrinkage),
            _actual_indices(table.targets[block_mask]),
            config.top_k,
        )
        metrics.update(
            {
                "blockIndex": block_index,
                "startIndex": block_start,
                "endIndex": block_end,
                "selectedParams": best_params.to_dict(),
                "selectedShrinkage": best_shrinkage,
                "innerCandidates": selection_candidates,
                "uniformLogLoss": _UNIFORM_LOG_LOSS,
                "uniformBrier": _UNIFORM_BRIER,
                "jointGatePassed": (
                    float(metrics["meanLogLoss"]) < _UNIFORM_LOG_LOSS
                    and float(metrics["meanBrier"]) < _UNIFORM_BRIER
                    and float(metrics["top50PValue"]) < 0.05
                ),
            }
        )
        blocks.append(metrics)
    data_hash = hashlib.sha256(
        chronological.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    return LightGBMBlockBacktestResult(
        lottery=rule.code,
        config=config,
        feature_count=table.features.shape[1],
        first_block_start=first_block_start,
        blocks=tuple(blocks),
        data_sha256=data_hash,
        source_fingerprint=learned_ranker_source_fingerprint(),
    )


def write_lightgbm_report(
    result: LightGBMBlockBacktestResult, path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["reportSha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return destination


__all__ = [
    "LightGBMBlockBacktestResult",
    "LightGBMChallengeConfig",
    "LightGBMFeatureTable",
    "LightGBMParams",
    "build_lightgbm_feature_table",
    "joint_digit_probabilities",
    "run_lightgbm_block_backtest",
    "write_lightgbm_report",
]
