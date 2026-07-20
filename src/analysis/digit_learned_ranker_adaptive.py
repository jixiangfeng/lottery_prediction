# -*- coding: utf-8 -*-
"""learned_ranker_v4开发区在线自适应逐期模拟。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    LearnedFeatureConfig,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import LearnedRankerParams, params_fingerprint
from src.analysis.digit_learned_ranker_search import (
    LearnedSearchConfig,
    LearnedSplit,
    search_learned_ranker_params,
)
from src.analysis.digit_learned_ranker_walk_forward import (
    LearnedWalkForwardPeriod,
    _period,
    run_learned_ranker_walk_forward,
)
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class AdaptiveResearchConfig:
    """在线自适应开发协议；所有边界均按升序历史行号定义。"""

    development_end: int
    outer_periods: int = 500
    retrain_interval: int = 10
    training_lookback: int = 500
    inner_validation_periods: int = 100
    inner_stride: int = 10
    min_train_size: int = 150
    random_trials: int = 4
    local_trials: int = 2
    seed: int = 20260717
    maximum_ece: float = 0.05
    direct_top_k: int = 50
    group_top_k: int = 10
    position_pool_size: int = 3
    checkpoint_dir: str | Path | None = None

    def __post_init__(self) -> None:
        positive = (
            self.development_end,
            self.outer_periods,
            self.retrain_interval,
            self.training_lookback,
            self.inner_validation_periods,
            self.inner_stride,
            self.min_train_size,
            self.random_trials,
        )
        if any(value <= 0 for value in positive) or self.local_trials < 0:
            raise ValueError("自适应周期、窗口、步长和搜索次数必须为正")
        if (
            self.training_lookback
            <= self.min_train_size + self.inner_validation_periods
        ):
            raise ValueError("训练回看窗口必须容纳内层Search和Validation")
        if not 0 <= self.maximum_ece <= 1:
            raise ValueError("ECE阈值必须位于0..1")


@dataclass(frozen=True)
class AdaptiveSelection:
    """一次参数重选及其仅使用过去数据的证据。"""

    block_start_index: int
    history_end_index: int
    inner_history_start_index: int
    selected_params: LearnedRankerParams
    feature_config: LearnedFeatureConfig
    search_objective: float
    validation_objective: float
    mean_log_loss: float
    mean_brier_score: float
    expected_calibration_error: float
    stable_blocks: int
    abstained: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockStartIndex": self.block_start_index,
            "historyEndIndex": self.history_end_index,
            "innerHistoryStartIndex": self.inner_history_start_index,
            "params": self.selected_params.to_dict(),
            "featureConfig": {
                "windows": list(self.feature_config.windows),
                "alpha": self.feature_config.alpha,
                "halfLife": self.feature_config.half_life,
                "omissionCap": self.feature_config.omission_cap,
                "windowWeights": dict(self.feature_config.window_weights or ()),
            },
            "searchObjective": self.search_objective,
            "validationObjective": self.validation_objective,
            "meanLogLoss": self.mean_log_loss,
            "meanBrierScore": self.mean_brier_score,
            "expectedCalibrationError": self.expected_calibration_error,
            "stableBlocks": self.stable_blocks,
            "abstained": self.abstained,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AdaptivePrediction:
    """单个目标期的事前参数状态和事后评估结果。"""

    target_index: int
    target_issue: str
    history_end_issue: str
    params_fingerprint: str
    uniform_shrinkage: float
    abstained: bool
    actual_text: str
    actual_rank: int
    actual_probability: float
    log_loss: float
    brier_score: float
    direct_hit: bool
    group_hit: bool
    position_hits: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "targetIndex": payload["target_index"],
            "targetIssue": payload["target_issue"],
            "historyEndIssue": payload["history_end_issue"],
            "paramsFingerprint": payload["params_fingerprint"],
            "uniformShrinkage": payload["uniform_shrinkage"],
            "abstained": payload["abstained"],
            "actualText": payload["actual_text"],
            "actualRank": payload["actual_rank"],
            "actualProbability": payload["actual_probability"],
            "logLoss": payload["log_loss"],
            "brierScore": payload["brier_score"],
            "directHit": payload["direct_hit"],
            "groupHit": payload["group_hit"],
            "positionHits": payload["position_hits"],
        }


@dataclass(frozen=True)
class AdaptiveResearchReport:
    rule_code: str
    config: AdaptiveResearchConfig
    outer_start_index: int
    outer_end_index: int
    frozen_test_read: bool
    selections: tuple[AdaptiveSelection, ...]
    predictions: tuple[AdaptivePrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        periods = len(self.predictions)
        active = [item for item in self.predictions if not item.abstained]
        return {
            "schemaVersion": 1,
            "modelVersion": "learned_ranker_v4_adaptive_research",
            "ruleCode": self.rule_code,
            "config": {
                **asdict(self.config),
                "checkpoint_dir": (
                    str(self.config.checkpoint_dir)
                    if self.config.checkpoint_dir is not None
                    else None
                ),
            },
            "outerStartIndex": self.outer_start_index,
            "outerEndIndex": self.outer_end_index,
            "frozenTestRead": self.frozen_test_read,
            "metrics": {
                "periods": periods,
                "activePeriods": len(active),
                "abstainedPeriods": periods - len(active),
                "abstentionRate": (periods - len(active)) / periods if periods else 0.0,
                "meanLogLoss": float(
                    np.mean([item.log_loss for item in self.predictions])
                ),
                "uniformLogLoss": math.log(1000),
                "meanBrierScore": float(
                    np.mean([item.brier_score for item in self.predictions])
                ),
                "uniformBrierScore": 0.999,
                "activeDirectHits": sum(item.direct_hit for item in active),
                "activeGroupHits": sum(item.group_hit for item in active),
                "activePositionHits": sum(item.position_hits for item in active),
            },
            "selections": [item.to_dict() for item in self.selections],
            "predictions": [item.to_dict() for item in self.predictions],
        }


def _selection_reasons(
    report: Any, result: Any, config: AdaptiveResearchConfig
) -> tuple[str, ...]:
    reasons = []
    if result.search_objective <= 0:
        reasons.append("Inner Search proper-scoring未优于均匀基线")
    if result.validation_objective <= 0:
        reasons.append("Inner Validation proper-scoring未优于均匀基线")
    if result.params.uniform_shrinkage <= 0:
        reasons.append("Search选择λ=0，模型退回均匀概率")
    if report.mean_log_loss >= report.uniform_log_loss:
        reasons.append("Inner Validation LogLoss未优于均匀基线")
    if report.mean_brier_score > report.uniform_brier_score:
        reasons.append("Inner Validation Brier劣于均匀基线")
    if report.expected_calibration_error > config.maximum_ece:
        reasons.append("Inner Validation ECE超过固定阈值")
    if report.stable_blocks < 2:
        reasons.append("Inner Validation稳定时间块不足2/3")
    return tuple(reasons)


def _select_for_block(
    chronological: pd.DataFrame,
    rule: LotteryRule,
    block_start: int,
    config: AdaptiveResearchConfig,
) -> AdaptiveSelection:
    inner_start = max(0, block_start - config.training_lookback)
    inner_history = chronological.iloc[inner_start:block_start].reset_index(drop=True)
    inner_end = len(inner_history)
    search_end = inner_end - config.inner_validation_periods
    split = LearnedSplit(search_end, inner_end, inner_end)
    checkpoint = None
    if config.checkpoint_dir is not None:
        checkpoint = Path(config.checkpoint_dir) / f"{rule.code}_{block_start}.json"
    result = search_learned_ranker_params(
        inner_history,
        rule,
        LearnedSearchConfig(
            split=split,
            min_train_size=config.min_train_size,
            random_trials=config.random_trials,
            local_trials=config.local_trials,
            evaluation_stride=config.inner_stride,
            seed=config.seed + block_start,
            feature_config=LearnedFeatureConfig(),
            feature_configs=(LearnedFeatureConfig(),),
            objective_profile="research_calibrated",
            direct_objective_top_k=config.direct_top_k,
            group_objective_top_k=config.group_top_k,
            position_objective_pool_size=config.position_pool_size,
            progress_checkpoint_path=checkpoint,
        ),
    )
    selected = replace(
        result.params,
        direct_top_k=config.direct_top_k,
        group_top_k=config.group_top_k,
        position_pool_size=config.position_pool_size,
    )
    validation_report = run_learned_ranker_walk_forward(
        inner_history,
        rule,
        selected,
        LearnedSplit(search_end, search_end, inner_end),
        feature_config=result.feature_config,
        test_segment_used_for_selection=False,
    )
    reasons = _selection_reasons(validation_report, result, config)
    return AdaptiveSelection(
        block_start_index=block_start,
        history_end_index=block_start - 1,
        inner_history_start_index=inner_start,
        selected_params=selected,
        feature_config=result.feature_config,
        search_objective=result.search_objective,
        validation_objective=result.validation_objective,
        mean_log_loss=validation_report.mean_log_loss,
        mean_brier_score=validation_report.mean_brier_score,
        expected_calibration_error=validation_report.expected_calibration_error,
        stable_blocks=validation_report.stable_blocks,
        abstained=bool(reasons),
        reasons=reasons,
    )


def _prediction(
    period: LearnedWalkForwardPeriod,
    params: LearnedRankerParams,
    abstained: bool,
    config: LearnedFeatureConfig,
) -> AdaptivePrediction:
    return AdaptivePrediction(
        target_index=period.target_index,
        target_issue=period.target_issue,
        history_end_issue=period.history_end_issue,
        params_fingerprint=params_fingerprint(params, config),
        uniform_shrinkage=params.uniform_shrinkage,
        abstained=abstained,
        actual_text=period.actual_text,
        actual_rank=period.actual_rank,
        actual_probability=period.actual_probability,
        log_loss=period.log_loss,
        brier_score=period.brier_score,
        direct_hit=period.direct_hit,
        group_hit=period.group_hit,
        position_hits=sum(
            rank <= params.position_pool_size for rank in period.position_ranks
        ),
    )


def run_adaptive_research(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: AdaptiveResearchConfig,
) -> AdaptiveResearchReport:
    """只在development_end之前逐期模拟；每个参数块只读取块起点之前历史。"""

    if rule.code not in {"fc3d", "pl3"} or rule.draw_count != 3:
        raise ValueError("自适应v4只支持fc3d/pl3")
    if "期数" not in history.columns:
        raise ValueError("自适应v4输入必须先标准化并包含期数列")
    ordered_issues = sort_digit_dataframe_by_issue(
        history.loc[:, ["期数"]], ascending=True
    )
    if config.development_end > len(ordered_issues):
        raise ValueError("development_end超过历史长度")
    development_issues = set(
        ordered_issues.iloc[: config.development_end]["期数"].astype(str)
    )
    raw_development = history.loc[
        history["期数"].astype(str).isin(development_issues)
    ].copy()
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(raw_development, rule), ascending=True
    )
    outer_start = config.development_end - config.outer_periods
    minimum_start = config.training_lookback
    if outer_start < minimum_start:
        raise ValueError("外层起点必须保留完整训练回看窗口")
    indices = tuple(range(outer_start, config.development_end))
    states = dict(
        zip(
            indices,
            iter_rolling_history_states(
                chronological, rule, indices, LearnedFeatureConfig()
            ),
        )
    )
    selections = []
    predictions = []
    for block_start in range(
        outer_start, config.development_end, config.retrain_interval
    ):
        selection = _select_for_block(chronological, rule, block_start, config)
        selections.append(selection)
        effective_params = (
            replace(selection.selected_params, uniform_shrinkage=0.0)
            if selection.abstained
            else selection.selected_params
        )
        block_end = min(block_start + config.retrain_interval, config.development_end)
        for index in range(block_start, block_end):
            period = _period(
                chronological,
                index,
                rule,
                effective_params,
                selection.feature_config,
                history_state=states[index],
            )
            predictions.append(
                _prediction(
                    period,
                    effective_params,
                    selection.abstained,
                    selection.feature_config,
                )
            )
    return AdaptiveResearchReport(
        rule_code=rule.code,
        config=config,
        outer_start_index=outer_start,
        outer_end_index=config.development_end,
        frozen_test_read=False,
        selections=tuple(selections),
        predictions=tuple(predictions),
    )


def write_adaptive_report(report: AdaptiveResearchReport, path: str | Path) -> Path:
    """原子写入在线自适应开发报告。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


__all__ = [
    "AdaptivePrediction",
    "AdaptiveResearchConfig",
    "AdaptiveResearchReport",
    "AdaptiveSelection",
    "run_adaptive_research",
    "write_adaptive_report",
]
