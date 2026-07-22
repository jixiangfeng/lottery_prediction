# -*- coding: utf-8 -*-
"""多个在线梯度变体共享逐期特征矩阵的流式执行器。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.analysis import digit_online_gradient as engine
from src.analysis.digit_daily_policy import select_daily_candidates
from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_learned_features import (
    BEHAVIORAL_FEATURE_NAMES,
    LearnedFeatureConfig,
    build_candidate_features,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import rank_candidate_indices
from src.analysis.digit_online_gradient import (
    OnlineGradientCandidate,
    OnlineGradientConfig,
    OnlineGradientPeriod,
    OnlineGradientReport,
    OnlineGradientSelection,
    online_gradient_step,
)
from src.analysis.digit_statistics import classify_digit_shape
from src.lotteries.base import LotteryRule

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass
class _VariantRuntime:
    config: OnlineGradientConfig
    candidates: list[OnlineGradientCandidate]
    learners: list[engine._CandidateState]
    selections: list[OnlineGradientSelection]
    periods: list[OnlineGradientPeriod]
    current_selection: OnlineGradientSelection | None = None


def _common_execution_key(config: OnlineGradientConfig) -> tuple[object, ...]:
    return (
        config.development_end,
        config.outer_periods,
        config.calibration_interval,
        config.search_lookback,
        config.validation_lookback,
        config.warmup_history,
        config.learning_rates,
        config.shrinkages,
    )


def _new_runtime(config: OnlineGradientConfig) -> _VariantRuntime:
    candidates = [
        OnlineGradientCandidate(learning_rate, shrinkage)
        for learning_rate in config.learning_rates
        for shrinkage in config.shrinkages
    ]
    learners = [
        engine._CandidateState(candidate, engine._initial_weights(config), [], [])
        for candidate in candidates
    ]
    return _VariantRuntime(config, candidates, learners, [], [])


def run_online_gradient_research_variants(
    history: pd.DataFrame,
    rule: LotteryRule,
    configs: Mapping[str, OnlineGradientConfig],
    feature_config: LearnedFeatureConfig = LearnedFeatureConfig(),
    *,
    frozen_test_read: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> dict[str, OnlineGradientReport]:
    """流式运行多个变体；每个目标期只构建一次完整候选特征。"""

    if not configs:
        raise ValueError("至少需要一个在线梯度变体")
    if progress_interval <= 0:
        raise ValueError("progress_interval必须为正")
    if rule.code not in {"fc3d", "pl3"} or rule.draw_count != 3:
        raise ValueError("在线梯度变体只支持fc3d/pl3")
    common_keys = {_common_execution_key(config) for config in configs.values()}
    if len(common_keys) != 1:
        raise ValueError("共享特征的变体必须使用相同时间边界和候选网格")

    first = next(iter(configs.values()))
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    if first.development_end > len(chronological):
        raise ValueError("development_end超过开发历史")
    outer_start = first.development_end - first.outer_periods
    calibration_history = first.search_lookback + first.validation_lookback
    audit_start = outer_start - calibration_history
    if audit_start < first.warmup_history:
        raise ValueError("开发历史不足以保留warmup和校准窗口")

    indices = tuple(range(audit_start, first.development_end))
    states = iter_rolling_history_states(chronological, rule, indices, feature_config)
    runtimes = {name: _new_runtime(config) for name, config in configs.items()}
    include_behavior = any(
        feature in BEHAVIORAL_FEATURE_NAMES
        for config in configs.values()
        for feature in config.feature_names
    )

    for target_index, history_state in zip(indices, states):
        features = build_candidate_features(
            history_state,
            rule,
            include_behavioral_context=include_behavior,
        )
        actual_row = chronological.iloc[target_index]
        actual_text = "".join(
            str(int(actual_row[column])) for column in rule.number_columns
        )
        actual_index = int(actual_text)

        for runtime in runtimes.values():
            config = runtime.config
            if target_index >= outer_start and (
                runtime.current_selection is None
                or (target_index - outer_start) % config.calibration_interval == 0
            ):
                runtime.current_selection = engine._select_candidate(
                    runtime.learners, target_index, config
                )
                runtime.selections.append(runtime.current_selection)

            matrix = features[list(config.feature_names)].to_numpy(dtype=float)
            steps = [
                online_gradient_step(
                    matrix,
                    actual_index,
                    learner.weights,
                    learner.candidate,
                    config,
                )
                for learner in runtime.learners
            ]
            if target_index >= outer_start:
                selection = runtime.current_selection
                if selection is None:
                    raise RuntimeError("外层预测缺少参数选择")
                selected_index = runtime.candidates.index(selection.candidate)
                selected_learner = runtime.learners[selected_index]
                selected_step = steps[selected_index]
                probabilities = selected_step.final_probabilities
                order = rank_candidate_indices(probabilities, engine._CANDIDATES)
                rank = int(np.flatnonzero(order == actual_index)[0]) + 1
                if config.daily_candidate_policy:
                    if history_state.latest_numbers is None:
                        raise RuntimeError("日常候选策略缺少上期开奖")
                    latest_exact = "".join(
                        str(value) for value in history_state.latest_numbers
                    )
                    selected_candidates = select_daily_candidates(
                        (engine._CANDIDATES[int(index)] for index in order),
                        latest_exact=latest_exact,
                        top_k=config.direct_top_k,
                        maximum_triples=config.maximum_top50_triples,
                    )
                    selected_indices = tuple(
                        int(value) for value in selected_candidates
                    )
                    candidate_policy_rank = (
                        selected_candidates.index(actual_text) + 1
                        if actual_text in selected_candidates
                        else None
                    )
                else:
                    selected_indices = tuple(
                        int(index) for index in order[: config.direct_top_k]
                    )
                    candidate_policy_rank = (
                        rank if rank <= config.direct_top_k else None
                    )
                boundary_index = selected_indices[-1]
                contributions = selected_learner.weights * (
                    matrix[actual_index] - matrix[boundary_index]
                )
                top50_shape_counts = {"组六": 0, "组三": 0, "豹子": 0}
                for candidate_index in selected_indices:
                    digits = tuple(
                        int(value) for value in engine._CANDIDATES[candidate_index]
                    )
                    top50_shape_counts[classify_digit_shape(digits)] += 1
                deployed_probability = (
                    1.0 / 1000.0
                    if selection.abstained
                    else float(probabilities[actual_index])
                )
                deployed_brier = (
                    engine._UNIFORM_BRIER
                    if selection.abstained
                    else selected_step.brier
                )
                runtime.periods.append(
                    OnlineGradientPeriod(
                        target_index=target_index,
                        target_issue=str(actual_row["期数"]),
                        history_end_issue=str(history_state.history_end_issue),
                        candidate_key=selection.candidate.key,
                        learning_rate=selection.candidate.learning_rate,
                        uniform_shrinkage=selection.candidate.uniform_shrinkage,
                        abstained=selection.abstained,
                        actual_text=actual_text,
                        research_actual_probability=float(probabilities[actual_index]),
                        deployed_actual_probability=deployed_probability,
                        research_rank=rank,
                        candidate_policy_rank=candidate_policy_rank,
                        research_direct_hit=candidate_policy_rank is not None,
                        research_log_loss=selected_step.log_loss,
                        deployed_log_loss=-math.log(deployed_probability),
                        research_brier=selected_step.brier,
                        deployed_brier=deployed_brier,
                        gradient_norm=float(np.linalg.norm(selected_step.gradient)),
                        gradients={
                            name: float(value)
                            for name, value in zip(
                                config.feature_names, selected_step.gradient
                            )
                        },
                        boundary_contributions={
                            name: float(value)
                            for name, value in zip(config.feature_names, contributions)
                        },
                        weights_before={
                            name: float(value)
                            for name, value in zip(
                                config.feature_names, selected_learner.weights
                            )
                        },
                        weights_after={
                            name: float(value)
                            for name, value in zip(
                                config.feature_names, selected_step.weights_after
                            )
                        },
                        top50_shape_counts=top50_shape_counts,
                    )
                )
            for learner, step in zip(runtime.learners, steps):
                learner.log_losses.append(step.log_loss)
                learner.brier_scores.append(step.brier)
                learner.weights = step.weights_after

        if target_index >= outer_start and progress_callback is not None:
            processed = target_index - outer_start + 1
            if processed % progress_interval == 0 or processed == first.outer_periods:
                progress_callback(
                    {
                        "processedOuterPeriods": processed,
                        "totalOuterPeriods": first.outer_periods,
                        "completedFixedBlocks": processed // progress_interval,
                        "totalFixedBlocks": math.ceil(
                            first.outer_periods / progress_interval
                        ),
                        "targetIssue": str(actual_row["期数"]),
                    }
                )

    evidence_status = (
        "independent_frozen_test"
        if frozen_test_read
        else "exploratory_reused_development"
    )
    return {
        name: OnlineGradientReport(
            lottery=rule.code,
            outer_start_index=outer_start,
            outer_end_index=runtime.config.development_end,
            frozen_test_read=frozen_test_read,
            evidence_status=evidence_status,
            config=runtime.config,
            selections=tuple(runtime.selections),
            periods=tuple(runtime.periods),
        )
        for name, runtime in runtimes.items()
    }


__all__ = ["ProgressCallback", "run_online_gradient_research_variants"]
