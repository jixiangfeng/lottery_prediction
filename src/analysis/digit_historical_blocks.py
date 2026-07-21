# -*- coding: utf-8 -*-
"""固定规则下的全部连续500期历史滚动稳健性回测。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.analysis.digit_data import (
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_full_history_shadow import (
    FullHistoryShadowConfig,
    decay_shadow_weights,
)
from src.analysis.digit_learned_features import (
    FEATURE_NAMES,
    build_candidate_features,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import (
    learned_ranker_source_fingerprint,
    rank_candidate_indices,
)
from src.analysis.digit_online_gradient import (
    _CANDIDATES,
    _UNIFORM_BRIER,
    OnlineGradientCandidate,
    _CandidateState,
    _initial_weights,
    _select_candidate,
    online_gradient_step,
)
from src.lotteries.base import LotteryRule

_UNIFORM_LOG_LOSS = math.log(1000.0)


@dataclass(frozen=True)
class HistoricalBlockBacktestResult:
    lottery: str
    block_size: int
    first_block_start: int
    blocks: tuple[dict[str, object], ...]
    training_start: int
    training_end: int
    data_sha256: str
    source_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "modelVersion": "learned_ranker_v4",
            "evaluationKind": "historical_rolling_blocks",
            "evidenceStatus": "retrospective_robustness_only",
            "lottery": self.lottery,
            "blockSize": self.block_size,
            "firstBlockStart": self.first_block_start,
            "blocksEvaluated": len(self.blocks),
            "selectionPolicy": "all_complete_nonoverlapping_blocks",
            "blockSelectionAllowed": False,
            "trainingStart": self.training_start,
            "trainingEnd": self.training_end,
            "dataSha256": self.data_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "blocks": list(self.blocks),
            "formalPredictionActivated": False,
            "future500Required": True,
        }


def _sha256(history: pd.DataFrame) -> str:
    return hashlib.sha256(history.to_csv(index=False).encode("utf-8")).hexdigest()


def run_historical_block_backtest(
    history: pd.DataFrame,
    rule: LotteryRule,
    config: FullHistoryShadowConfig = FullHistoryShadowConfig(),
    *,
    block_size: int = 500,
) -> HistoricalBlockBacktestResult:
    if block_size <= 0:
        raise ValueError("block_size必须大于零")
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    required = config.search_lookback + config.validation_lookback
    first_block_start = config.warmup_history + required
    available = len(chronological) - first_block_start
    block_count = available // block_size
    if block_count <= 0:
        raise ValueError("历史不足以形成完整500期区块")
    online_config = config.online_config(len(chronological))
    candidates = [
        OnlineGradientCandidate(rate, shrinkage)
        for rate in config.learning_rates
        for shrinkage in config.shrinkages
    ]
    learners = [
        _CandidateState(candidate, _initial_weights(online_config), [], [])
        for candidate in candidates
    ]
    blocks: list[dict[str, Any]] = [
        {
            "blockIndex": index,
            "startIndex": first_block_start + index * block_size,
            "endIndex": first_block_start + (index + 1) * block_size,
            "periods": 0,
            "researchLogLoss": [],
            "deployedLogLoss": [],
            "researchBrier": [],
            "deployedBrier": [],
            "researchHits": 0,
            "activePeriods": 0,
            "abstainedPeriods": 0,
        }
        for index in range(block_count)
    ]
    selection = None
    last_selection_index: int | None = None
    states = iter_rolling_history_states(
        chronological,
        rule,
        range(config.warmup_history, len(chronological)),
        config.feature_config,
    )
    for target_index, history_state in zip(
        range(config.warmup_history, len(chronological)), states
    ):
        if len(learners[0].log_losses) >= required and (
            selection is None
            or last_selection_index is None
            or target_index - last_selection_index >= config.calibration_interval
        ):
            selection = _select_candidate(learners, target_index, online_config)
            last_selection_index = target_index
        matrix = build_candidate_features(history_state, rule)[
            list(FEATURE_NAMES)
        ].to_numpy(dtype=float)
        actual_row = chronological.iloc[target_index]
        actual_index = int(
            "".join(str(int(actual_row[column])) for column in rule.number_columns)
        )
        steps = [
            online_gradient_step(
                matrix, actual_index, learner.weights, learner.candidate, online_config
            )
            for learner in learners
        ]
        if target_index >= first_block_start:
            block_index = (target_index - first_block_start) // block_size
            if block_index < block_count:
                if selection is None:
                    raise RuntimeError("区块评估缺少此前历史选择")
                block = blocks[block_index]
                selected_index = candidates.index(selection.candidate)
                step = steps[selected_index]
                order = rank_candidate_indices(step.final_probabilities, _CANDIDATES)
                rank = int(np.flatnonzero(order == actual_index)[0]) + 1
                block["periods"] = int(block["periods"]) + 1
                block["researchLogLoss"].append(step.log_loss)
                block["researchBrier"].append(step.brier)
                if rank <= online_config.direct_top_k:
                    block["researchHits"] = int(block["researchHits"]) + 1
                if selection.abstained:
                    block["abstainedPeriods"] = int(block["abstainedPeriods"]) + 1
                    block["deployedLogLoss"].append(_UNIFORM_LOG_LOSS)
                    block["deployedBrier"].append(_UNIFORM_BRIER)
                else:
                    block["activePeriods"] = int(block["activePeriods"]) + 1
                    block["deployedLogLoss"].append(step.log_loss)
                    block["deployedBrier"].append(step.brier)
        for learner, step in zip(learners, steps):
            learner.weights = decay_shadow_weights(
                step.weights_after,
                learner.candidate,
                config.weight_half_life,
                online_config.zeroed_features,
            )
            learner.log_losses.append(step.log_loss)
            learner.brier_scores.append(step.brier)
            if len(learner.log_losses) > required:
                learner.log_losses.pop(0)
                learner.brier_scores.pop(0)
    summaries: list[dict[str, object]] = []
    for block in blocks:
        periods = int(block["periods"])
        if periods != block_size:
            raise RuntimeError("存在未完成区块，禁止写入回测报告")
        hits = int(block["researchHits"])
        summary = {
            "blockIndex": block["blockIndex"],
            "startIndex": block["startIndex"],
            "endIndex": block["endIndex"],
            "periods": periods,
            "researchMeanLogLoss": float(np.mean(block["researchLogLoss"])),
            "deployedMeanLogLoss": float(np.mean(block["deployedLogLoss"])),
            "researchMeanBrier": float(np.mean(block["researchBrier"])),
            "deployedMeanBrier": float(np.mean(block["deployedBrier"])),
            "uniformLogLoss": _UNIFORM_LOG_LOSS,
            "uniformBrier": _UNIFORM_BRIER,
            "researchTop50Hits": hits,
            "researchTop50HitRate": hits / periods,
            "researchTop50PValue": float(binom.sf(hits - 1, periods, 0.05)),
            "activePeriods": block["activePeriods"],
            "abstainedPeriods": block["abstainedPeriods"],
        }
        summaries.append(summary)
    return HistoricalBlockBacktestResult(
        lottery=rule.code,
        block_size=block_size,
        first_block_start=first_block_start,
        blocks=tuple(summaries),
        training_start=config.warmup_history,
        training_end=len(chronological),
        data_sha256=_sha256(chronological),
        source_fingerprint=learned_ranker_source_fingerprint(),
    )


def write_historical_block_report(
    result: HistoricalBlockBacktestResult, path: str | Path
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
    "HistoricalBlockBacktestResult",
    "run_historical_block_backtest",
    "write_historical_block_report",
]
