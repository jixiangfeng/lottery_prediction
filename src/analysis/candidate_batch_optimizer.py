# -*- coding: utf-8 -*-
"""候选批次二次优化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.analysis.candidate_portfolio_score import CandidatePortfolioScore


@dataclass(frozen=True)
class CandidateBatchOptimizationResult:
    """多批候选中最优批次选择结果。"""

    groups: Sequence[Any]
    best_score: CandidatePortfolioScore
    best_index: int
    trial_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.trial_count > 1,
            "trialCount": self.trial_count,
            "bestIndex": self.best_index,
            "bestScore": self.best_score.to_dict(),
        }


def select_best_candidate_batch(
    batches: Sequence[Sequence[Any]],
    score_fn: Callable[[Sequence[Any]], CandidatePortfolioScore],
) -> CandidateBatchOptimizationResult:
    """从多批候选中选择组合总评分最高的一批。"""

    if not batches:
        raise ValueError("候选批次不能为空")

    best_groups = batches[0]
    best_score = score_fn(best_groups)
    best_index = 1
    for index, groups in enumerate(batches[1:], 2):
        score = score_fn(groups)
        if score.final_score > best_score.final_score:
            best_groups = groups
            best_score = score
            best_index = index
    return CandidateBatchOptimizationResult(
        groups=best_groups,
        best_score=best_score,
        best_index=best_index,
        trial_count=len(batches),
    )
