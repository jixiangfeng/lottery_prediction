#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取最新开奖并基于锁定影子状态生成下一期研究候选。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_daily_policy import select_daily_candidates  # noqa: E402
from src.analysis.digit_data import (  # noqa: E402
    load_digit_csv,
    normalize_digit_dataframe,
    sort_digit_dataframe_by_issue,
)
from src.analysis.digit_full_history_shadow import (  # noqa: E402
    FullHistoryShadowConfig,
    decay_shadow_weights,
    train_full_history_shadow,
)
from src.analysis.digit_history_fetcher import fetch_digit_history  # noqa: E402
from src.analysis.digit_learned_features import (  # noqa: E402
    FEATURE_NAMES,
    build_candidate_features,
    build_history_state,
    iter_rolling_history_states,
)
from src.analysis.digit_learned_ranker import rank_candidate_indices  # noqa: E402
from src.analysis.digit_online_gradient import (  # noqa: E402
    OnlineGradientCandidate,
    OnlineGradientSelection,
    _CandidateState,
    online_gradient_step,
)
from src.lotteries import get_lottery_rule  # noqa: E402

_CANDIDATES = tuple(f"{value:03d}" for value in range(1000))


def _merge_latest_draws(
    base_history: pd.DataFrame, lottery: str, periods: int, timeout: float, retries: int
) -> pd.DataFrame:
    rule = get_lottery_rule(lottery)
    fetched = fetch_digit_history(
        lottery, periods=periods, timeout=timeout, retries=retries
    )
    fetched_rows = pd.DataFrame(
        [
            {
                "期数": draw.issue,
                **{
                    column: draw.numbers[index]
                    for index, column in enumerate(rule.number_columns)
                },
            }
            for draw in fetched
        ]
    )
    merged = pd.concat([base_history, fetched_rows], ignore_index=True)
    merged["期数"] = merged["期数"].astype(str).str.strip()
    merged["__issue_order"] = merged["期数"].map(int)
    merged = merged.sort_values("__issue_order", kind="mergesort").drop_duplicates(
        "期数", keep="last"
    )
    return merged.drop(columns="__issue_order")


def _load_shadow_state(path: Path) -> tuple[
    dict[str, object],
    FullHistoryShadowConfig,
    list[_CandidateState],
    OnlineGradientSelection,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = FullHistoryShadowConfig()
    learners: list[_CandidateState] = []
    for state in payload["candidateStates"]:
        candidate_payload = state["candidate"]
        candidate = OnlineGradientCandidate(
            float(candidate_payload["learningRate"]),
            float(candidate_payload["uniformShrinkage"]),
        )
        weights = np.array(
            [float(state["weights"][name]) for name in FEATURE_NAMES], dtype=float
        )
        learners.append(
            _CandidateState(
                candidate,
                weights,
                list(state["rollingLogLoss"]),
                list(state["rollingBrier"]),
            )
        )
    selection_payload = payload["currentSelection"]
    selected_candidate = selection_payload["candidate"]
    selection = OnlineGradientSelection(
        block_start_index=int(selection_payload["blockStartIndex"]),
        candidate=OnlineGradientCandidate(
            float(selected_candidate["learningRate"]),
            float(selected_candidate["uniformShrinkage"]),
        ),
        search_mean_log_loss=float(selection_payload["searchMeanLogLoss"]),
        validation_mean_log_loss=float(selection_payload["validationMeanLogLoss"]),
        validation_mean_brier=float(selection_payload["validationMeanBrier"]),
        stable_blocks=int(selection_payload["stableBlocks"]),
        abstained=bool(selection_payload["abstained"]),
        reasons=tuple(str(reason) for reason in selection_payload["reasons"]),
    )
    return payload, config, learners, selection


def _predict_from_learners(
    chronological: pd.DataFrame,
    rule,
    config: FullHistoryShadowConfig,
    learners: list[_CandidateState],
    selection: OnlineGradientSelection,
) -> list[str]:
    online_config = config.online_config(len(chronological))
    next_state = build_history_state(chronological, rule, config.feature_config)
    next_matrix = build_candidate_features(next_state, rule)[
        list(FEATURE_NAMES)
    ].to_numpy(dtype=float)
    selected = next(
        learner
        for learner in learners
        if learner.candidate.key == selection.candidate.key
    )
    scores = next_matrix @ selected.weights / online_config.temperature
    shifted = scores - float(scores.max())
    model_probabilities = np.exp(shifted)
    model_probabilities /= float(model_probabilities.sum())
    final_probabilities = (
        selection.candidate.uniform_shrinkage * model_probabilities
        + (1.0 - selection.candidate.uniform_shrinkage) / 1000.0
    )
    order = rank_candidate_indices(final_probabilities, _CANDIDATES)
    ranked_candidates = (_CANDIDATES[int(index)] for index in order)
    latest_row = chronological.iloc[-1]
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
    return list(
        select_daily_candidates(
            ranked_candidates,
            latest_exact=latest_exact,
            top_k=50,
            maximum_triples=1,
        )
    )


def _incremental_predict(
    history: pd.DataFrame, lottery: str, shadow_state: Path
) -> dict[str, object]:
    rule = get_lottery_rule(lottery)
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    payload, config, learners, selection = _load_shadow_state(shadow_state)
    state_end = int(cast(int | str, payload["trainingEndIndex"]))
    if len(chronological) < state_end:
        raise ValueError("CSV历史短于影子状态，不能增量预测")

    online_config = config.online_config(len(chronological))
    update_indices = tuple(range(state_end, len(chronological)))
    for target_index, history_state in zip(
        update_indices,
        iter_rolling_history_states(
            chronological, rule, update_indices, config.feature_config
        ),
    ):
        matrix = build_candidate_features(history_state, rule)[
            list(FEATURE_NAMES)
        ].to_numpy(dtype=float)
        row = chronological.iloc[target_index]
        actual_index = int(
            "".join(str(int(row[column])) for column in rule.number_columns)
        )
        for learner in learners:
            step = online_gradient_step(
                matrix, actual_index, learner.weights, learner.candidate, online_config
            )
            learner.weights = decay_shadow_weights(
                step.weights_after,
                learner.candidate,
                config.weight_half_life,
                online_config.zeroed_features,
            )
            learner.log_losses.append(step.log_loss)
            learner.brier_scores.append(step.brier)
            required = config.search_lookback + config.validation_lookback
            if len(learner.log_losses) > required:
                learner.log_losses.pop(0)
                learner.brier_scores.pop(0)

    latest_row = chronological.iloc[-1]
    new_draws = []
    for index in update_indices:
        row = chronological.iloc[index]
        new_draws.append(
            {
                "issue": str(row["期数"]),
                "number": "".join(
                    str(int(row[column])) for column in rule.number_columns
                ),
            }
        )
    research_top50 = _predict_from_learners(
        chronological, rule, config, learners, selection
    )
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
    return {
        "lottery": lottery,
        "latestHistoryIssue": str(latest_row["期数"]),
        "newDrawsUsed": new_draws,
        "formalPredictionActivated": False,
        "shadowOnly": True,
        "selection": selection.to_dict(),
        "candidatePolicy": {
            "excludeLatestExact": True,
            "latestExact": latest_exact,
            "maximumTriples": 1,
            "selectedTriples": sum(
                len(set(candidate)) == 1 for candidate in research_top50
            ),
        },
        "researchTop50": research_top50,
    }


def _full_history_predict(history: pd.DataFrame, lottery: str) -> dict[str, object]:
    rule = get_lottery_rule(lottery)
    result = train_full_history_shadow(history, rule)
    payload = result.to_dict()
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    latest_row = chronological.iloc[-1]
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
    next_prediction = cast(dict[str, object], payload["nextPrediction"])
    candidates = list(cast(list[str], next_prediction["researchTop50"]))
    return {
        "lottery": lottery,
        "latestHistoryIssue": payload["latestHistoryIssue"],
        "newDrawsUsed": [],
        "formalPredictionActivated": payload["formalPredictionActivated"],
        "shadowOnly": next_prediction["shadowOnly"],
        "selection": payload["currentSelection"],
        "candidatePolicy": {
            "excludeLatestExact": True,
            "latestExact": latest_exact,
            "maximumTriples": 1,
            "selectedTriples": sum(
                len(set(candidate)) == 1 for candidate in candidates
            ),
        },
        "researchTop50": candidates,
    }


def _print_text(result: dict[str, object]) -> None:
    top50 = cast(list[str], result["researchTop50"])
    print(f"玩法：{result['lottery']}")
    print(f"最新已开奖期号：{result['latestHistoryIssue']}")
    new_draws = cast(list[dict[str, str]], result["newDrawsUsed"])
    if new_draws:
        used = ", ".join(f"{row['issue']}={row['number']}" for row in new_draws)
        print(f"本次增量使用：{used}")
    print(f"正式推荐：{result['formalPredictionActivated']}")
    print(f"仅影子研究：{result['shadowOnly']}")
    policy = cast(dict[str, object], result["candidatePolicy"])
    print(
        "候选策略：排除上期原号，豹子最多"
        f"{policy['maximumTriples']}个（本期保留{policy['selectedTriples']}个）"
    )
    print(f"Top10：{', '.join(top50[:10])}")
    print(f"Top50：{', '.join(top50)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", help="默认 data/<玩法>/official_history.csv")
    parser.add_argument("--periods", type=int, default=5, help="联网抓取最近期数")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-fetch", action="store_true", help="只使用本地CSV")
    parser.add_argument("--json", action="store_true", help="输出JSON")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv or f"data/{args.lottery}/official_history.csv")
    history = load_digit_csv(csv_path, get_lottery_rule(args.lottery))
    if not args.no_fetch:
        history = _merge_latest_draws(
            history, args.lottery, args.periods, args.timeout, args.retries
        )
    shadow_state = (
        ROOT / f"state/learned_ranker_v4/full_history_shadow_{args.lottery}.json"
    )
    if shadow_state.exists():
        result = _incremental_predict(history, args.lottery, shadow_state)
    else:
        result = _full_history_predict(history, args.lottery)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
