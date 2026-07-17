# -*- coding: utf-8 -*-
"""构建数字彩蒙特卡洛与机器学习外部投票分数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.digit_candidates import (
    ENSEMBLE_MODEL_NAMES,
    DigitCandidateConfig,
    DigitExternalModelScores,
)
from src.analysis.digit_ml_ranker import score_digit_ranker, train_digit_ranker
from src.analysis.digit_monte_carlo import simulate_digit_candidates
from src.analysis.digit_statistics import DigitStatisticsResult
from src.lotteries.base import LotteryRule


@dataclass(frozen=True)
class DigitAdvancedModelDiagnostics:
    """外部模型运行证据。"""

    monte_carlo_enabled: bool
    monte_carlo_simulations: int
    monte_carlo_accepted: int
    monte_carlo_pair_conditioned: bool
    monte_carlo_structure_conditioned: bool
    ml_enabled: bool
    ml_trained: bool
    ml_training_targets: int
    ml_training_samples: int

    @property
    def available_model_names(self) -> tuple[str, ...]:
        """返回当前代码可提供的全部模型槽位。"""

        return ENSEMBLE_MODEL_NAMES

    @property
    def active_model_names(self) -> tuple[str, ...]:
        """返回本次报告提供非中性结果或模型候选信号的模型。"""

        active = list(ENSEMBLE_MODEL_NAMES[:14])
        if self.monte_carlo_enabled and self.monte_carlo_accepted > 0:
            active.append("monteCarlo")
        if self.ml_enabled and self.ml_trained:
            active.append("mlRanker")
        return tuple(active)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monteCarloEnabled": self.monte_carlo_enabled,
            "monteCarloSimulations": self.monte_carlo_simulations,
            "monteCarloAccepted": self.monte_carlo_accepted,
            "monteCarloPairConditioned": self.monte_carlo_pair_conditioned,
            "monteCarloStructureConditioned": self.monte_carlo_structure_conditioned,
            "mlEnabled": self.ml_enabled,
            "mlTrained": self.ml_trained,
            "mlTrainingTargets": self.ml_training_targets,
            "mlTrainingSamples": self.ml_training_samples,
            "activeModelNames": list(self.active_model_names),
            "activeModelCount": len(self.active_model_names),
            "availableModelNames": list(self.available_model_names),
            "availableModelCount": len(self.available_model_names),
        }


def build_advanced_model_scores(
    history: pd.DataFrame,
    stats: DigitStatisticsResult,
    rule: LotteryRule,
    config: DigitCandidateConfig,
    *,
    enable_monte_carlo: bool = True,
    monte_carlo_simulations: int = 20_000,
    enable_ml: bool = True,
    ml_min_train_size: int = 30,
    ml_training_periods: int = 60,
    ml_negative_samples: int = 9,
    seed: int = 20260716,
) -> tuple[DigitExternalModelScores, DigitAdvancedModelDiagnostics]:
    """构建两个外部投票器；历史不足时机器学习分数自动为空。"""

    monte_carlo_scores: dict[str, float] = {}
    monte_carlo_accepted = 0
    monte_carlo_pair_conditioned = False
    monte_carlo_structure_conditioned = False
    if enable_monte_carlo:
        simulation = simulate_digit_candidates(
            stats,
            rule,
            config,
            simulations=monte_carlo_simulations,
            seed=seed,
        )
        monte_carlo_scores = simulation.scores
        monte_carlo_accepted = simulation.accepted
        monte_carlo_pair_conditioned = simulation.pair_conditioned
        monte_carlo_structure_conditioned = simulation.structure_conditioned
    ranker = None
    ml_scores: dict[str, float] = {}
    if enable_ml:
        ranker = train_digit_ranker(
            history,
            rule,
            config,
            min_train_size=ml_min_train_size,
            training_periods=ml_training_periods,
            negative_samples=ml_negative_samples,
            seed=seed,
        )
        ml_scores = score_digit_ranker(ranker, stats, rule, config)
    diagnostics = DigitAdvancedModelDiagnostics(
        monte_carlo_enabled=enable_monte_carlo,
        monte_carlo_simulations=monte_carlo_simulations if enable_monte_carlo else 0,
        monte_carlo_accepted=monte_carlo_accepted,
        monte_carlo_pair_conditioned=monte_carlo_pair_conditioned,
        monte_carlo_structure_conditioned=monte_carlo_structure_conditioned,
        ml_enabled=enable_ml,
        ml_trained=ranker is not None,
        ml_training_targets=ranker.training_targets if ranker is not None else 0,
        ml_training_samples=ranker.training_samples if ranker is not None else 0,
    )
    return (
        DigitExternalModelScores(monte_carlo_scores, ml_scores),
        diagnostics,
    )


__all__ = ["DigitAdvancedModelDiagnostics", "build_advanced_model_scores"]
