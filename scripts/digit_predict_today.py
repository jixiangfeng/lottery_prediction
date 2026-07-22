#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取最新开奖并基于锁定影子状态生成下一期研究候选。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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
    validate_locked_shadow_state,
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
from src.analysis.digit_prediction_narrative import (  # noqa: E402
    DeepSeekNarrativeConfig,
    deterministic_prediction_narrative,
    load_deepseek_narrative_config,
    request_deepseek_prediction_narrative,
)
from src.lotteries import get_lottery_rule  # noqa: E402

_CANDIDATES = tuple(f"{value:03d}" for value in range(1000))
_FEATURE_LABELS = {
    "position_frequency": "位置频率",
    "position_omission": "位置遗漏",
    "pair_frequency": "二位组合频率",
    "sum_distribution": "和值分布",
    "span_distribution": "跨度分布",
    "recent_trend": "近期趋势",
    "position_trend": "位置趋势",
    "pair_trend": "二位组合趋势",
    "shape_transition": "形态转移",
    "shape_recent_deviation": "近期形态偏离",
    "constraint_penalty": "软约束惩罚",
}


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


def _load_shadow_payload(payload: Mapping[str, Any]) -> tuple[
    dict[str, Any],
    FullHistoryShadowConfig,
    list[_CandidateState],
    OnlineGradientSelection,
]:
    document = dict(payload)
    config = FullHistoryShadowConfig()
    learners: list[_CandidateState] = []
    for state in document["candidateStates"]:
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
    selection_payload = document["currentSelection"]
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
    return document, config, learners, selection


def _load_shadow_state(path: Path, lottery: str) -> tuple[
    dict[str, Any],
    FullHistoryShadowConfig,
    list[_CandidateState],
    OnlineGradientSelection,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("影子状态根节点必须是JSON对象")
    return _load_shadow_payload(
        validate_locked_shadow_state(payload, expected_lottery=lottery)
    )


def _predict_from_learners(
    chronological: pd.DataFrame,
    rule,
    config: FullHistoryShadowConfig,
    learners: list[_CandidateState],
    selection: OnlineGradientSelection,
) -> list[dict[str, object]]:
    if selection.candidate.uniform_shrinkage <= 0:
        return []
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
    latest_row = chronological.iloc[-1]
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
    selected_numbers = select_daily_candidates(
        (_CANDIDATES[int(index)] for index in order),
        latest_exact=latest_exact,
        top_k=50,
        maximum_triples=1,
    )
    contributions = (
        next_matrix * selected.weights[np.newaxis, :] / online_config.temperature
    )
    candidates: list[dict[str, object]] = []
    for rank, candidate in enumerate(selected_numbers, 1):
        index = int(candidate)
        feature_contributions = sorted(
            (
                {
                    "feature": name,
                    "featureLabel": _FEATURE_LABELS[name],
                    "contribution": float(contributions[index, feature_index]),
                }
                for feature_index, name in enumerate(FEATURE_NAMES)
            ),
            key=lambda item: (-abs(float(item["contribution"])), str(item["feature"])),
        )[:3]
        candidates.append(
            {
                "rank": rank,
                "number": candidate,
                "score": float(scores[index]),
                "normalizedRankingWeight": float(final_probabilities[index]),
                "relativeToUniform": float(final_probabilities[index] * 1000.0),
                "topContributions": feature_contributions,
            }
        )
    return candidates


def _unique_reasons(*groups: list[str] | tuple[str, ...]) -> list[str]:
    reasons: list[str] = []
    for group in groups:
        for reason in group:
            if reason and reason not in reasons:
                reasons.append(reason)
    return reasons


def _build_prediction_result(
    *,
    lottery: str,
    latest_history_issue: str,
    new_draws: list[dict[str, str]],
    selection: OnlineGradientSelection,
    research_candidates: list[dict[str, object]],
    state_payload: Mapping[str, object],
    latest_exact: str,
) -> dict[str, object]:
    formal_prediction_activated = bool(
        state_payload.get("formalPredictionActivated", False)
    )
    prospective_payload = state_payload.get("prospectiveValidation", {})
    prospective = (
        dict(prospective_payload) if isinstance(prospective_payload, Mapping) else {}
    )
    observed_periods = int(prospective.get("observedPeriods", 0)) + len(new_draws)
    required_periods = int(prospective.get("requiredPeriods", 500))
    prospective.update(
        {
            "status": "collecting" if observed_periods < required_periods else "ready",
            "observedPeriods": observed_periods,
            "requiredPeriods": required_periods,
            "remainingPeriods": max(0, required_periods - observed_periods),
        }
    )
    abstained = bool(selection.abstained or not formal_prediction_activated)
    reasons = _unique_reasons(
        selection.reasons,
        (
            (
                "影子模型仍在收集前瞻验证数据",
                "正式预测策略尚未激活",
            )
            if not formal_prediction_activated
            else ()
        ),
    )
    research_top50 = [str(item["number"]) for item in research_candidates]
    result: dict[str, object] = {
        "lottery": lottery,
        "latestHistoryIssue": latest_history_issue,
        "newDrawsUsed": new_draws,
        "status": "abstained" if abstained else "active",
        "abstained": abstained,
        "abstentionReasons": reasons if abstained else [],
        "formalPredictionActivated": formal_prediction_activated,
        "shadowOnly": True,
        "signal": {
            "evidenceStatus": state_payload.get("evidenceStatus", "prospective_only"),
            "modelWeight": selection.candidate.uniform_shrinkage,
            "learningRate": selection.candidate.learning_rate,
            "validationMeanLogLoss": selection.validation_mean_log_loss,
            "validationMeanBrier": selection.validation_mean_brier,
            "stableBlocks": selection.stable_blocks,
            "selectionAbstained": selection.abstained,
        },
        "prospectiveValidation": prospective,
        "selection": selection.to_dict(),
        "candidatePolicy": {
            "excludeLatestExact": True,
            "latestExact": latest_exact,
            "maximumTriples": 1,
            "selectedTriples": sum(
                len(set(str(item["number"]))) == 1 for item in research_candidates
            ),
        },
        "userVisibleCandidates": [] if abstained else research_top50,
        "researchTop50": research_top50,
        "researchCandidates": research_candidates,
        "rankingDisclaimer": (
            "normalizedRankingWeight仅用于内部排序，不是真实开奖概率；"
            "researchCandidates仅用于审计。"
        ),
        "ai": {"requested": False, "status": "disabled"},
    }
    result["narrative"] = deterministic_prediction_narrative(result)
    result["narrativeSource"] = "deterministic"
    return result


def _incremental_predict(
    history: pd.DataFrame, lottery: str, shadow_state: Path
) -> dict[str, object]:
    rule = get_lottery_rule(lottery)
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    payload, config, learners, selection = _load_shadow_state(shadow_state, lottery)
    state_end = int(payload["trainingEndIndex"])
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
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
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
    return _build_prediction_result(
        lottery=lottery,
        latest_history_issue=str(latest_row["期数"]),
        new_draws=new_draws,
        selection=selection,
        research_candidates=_predict_from_learners(
            chronological, rule, config, learners, selection
        ),
        state_payload=payload,
        latest_exact=latest_exact,
    )


def _full_history_predict(history: pd.DataFrame, lottery: str) -> dict[str, object]:
    rule = get_lottery_rule(lottery)
    result = train_full_history_shadow(history, rule)
    payload = result.to_dict()
    _, config, learners, selection = _load_shadow_payload(payload)
    chronological = sort_digit_dataframe_by_issue(
        normalize_digit_dataframe(history, rule), ascending=True
    )
    latest_row = chronological.iloc[-1]
    latest_exact = "".join(
        str(int(latest_row[column])) for column in rule.number_columns
    )
    return _build_prediction_result(
        lottery=lottery,
        latest_history_issue=str(payload["latestHistoryIssue"]),
        new_draws=[],
        selection=selection,
        research_candidates=_predict_from_learners(
            chronological, rule, config, learners, selection
        ),
        state_payload=payload,
        latest_exact=latest_exact,
    )


def _print_text(result: dict[str, object], *, show_research: bool = False) -> None:
    print(f"玩法：{result['lottery']}")
    print(f"最新已开奖期号：{result['latestHistoryIssue']}")
    if result["newDrawsUsed"]:
        used = ", ".join(
            f"{row['issue']}={row['number']}" for row in result["newDrawsUsed"]
        )
        print(f"本次增量使用：{used}")
    print(f"状态：{'暂不提供正式推荐' if result['abstained'] else '正式策略已激活'}")
    policy = dict(result["candidatePolicy"])
    print(
        "候选策略：排除上期原号，豹子最多"
        f"{policy['maximumTriples']}个（本期保留{policy['selectedTriples']}个）"
    )
    for reason in result["abstentionReasons"]:
        print(f"- {reason}")
    print(f"说明：{result['narrative']}")
    if result["userVisibleCandidates"]:
        print(f"候选：{', '.join(result['userVisibleCandidates'])}")
    elif show_research:
        research_candidates = list(result.get("researchCandidates", []))[:10]
        if research_candidates:
            print("研究观察Top10（未通过准入，不是正式推荐）：")
            print("、".join(str(item["number"]) for item in research_candidates))
            print("前三名排序依据：")
            for item in research_candidates[:3]:
                contributions = "，".join(
                    f"{part['featureLabel']} {float(part['contribution']):+.3f}"
                    for part in item["topContributions"]
                )
                print(
                    f"- {item['number']}：相对均匀排序权重 "
                    f"{float(item['relativeToUniform']):.3f} 倍；{contributions}"
                )
        else:
            print("候选：无（λ=0表示没有可用号码排序）")
    else:
        signal = dict(result.get("signal", {}))
        if float(signal.get("modelWeight", 0.0)) <= 0:
            print("候选：无（λ=0表示没有可用号码排序）")
        else:
            print("候选：无（研究排序仅保留在 --json 审计输出中）")
    ai = dict(result.get("ai", {}))
    if ai.get("requested") and ai.get("status") == "failed":
        print(f"AI润色：失败，已回退到确定性说明（{ai.get('error', '未知错误')}）")


def _apply_ai_narrative(
    result: dict[str, object],
    *,
    config_path: str | Path,
    model: str | None,
    timeout: float | None,
) -> None:
    result["ai"] = {"requested": True, "status": "pending"}
    try:
        config: DeepSeekNarrativeConfig = load_deepseek_narrative_config(
            config_path,
            model_override=model,
            timeout_override=timeout,
        )
        narrative = request_deepseek_prediction_narrative(
            result,
            config,
        )
    except (OSError, RuntimeError, ValueError) as error:
        result["ai"] = {
            "requested": True,
            "status": "failed",
            "provider": "deepseek",
            "error": str(error),
        }
        return
    result["narrative"] = narrative
    result["narrativeSource"] = "deepseek"
    result["ai"] = {
        "requested": True,
        "status": "completed",
        "provider": "deepseek",
        "model": config.model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", help="默认 data/<玩法>/official_history.csv")
    parser.add_argument(
        "--shadow-state",
        help="影子状态路径；默认 state/learned_ranker_v4/full_history_shadow_<玩法>.json",
    )
    parser.add_argument("--periods", type=int, default=5, help="联网抓取最近期数")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-fetch", action="store_true", help="只使用本地CSV")
    parser.add_argument("--json", action="store_true", help="输出JSON")
    parser.add_argument("--ai", action="store_true", help="使用DeepSeek润色状态说明")
    parser.add_argument("--ai-config", default="config/ai.local.json")
    parser.add_argument("--ai-model", help="覆盖本机AI配置中的model")
    parser.add_argument("--ai-timeout", type=float)
    parser.add_argument(
        "--show-research",
        action="store_true",
        help="显式展示未通过准入的研究Top10及前三名贡献",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv or f"data/{args.lottery}/official_history.csv")
    history = load_digit_csv(csv_path, get_lottery_rule(args.lottery))
    if not args.no_fetch:
        history = _merge_latest_draws(
            history, args.lottery, args.periods, args.timeout, args.retries
        )
    shadow_state = (
        Path(args.shadow_state)
        if args.shadow_state
        else (ROOT / f"state/learned_ranker_v4/full_history_shadow_{args.lottery}.json")
    )
    if shadow_state.exists():
        result = _incremental_predict(history, args.lottery, shadow_state)
    else:
        result = _full_history_predict(history, args.lottery)
    if args.ai:
        _apply_ai_narrative(
            result,
            config_path=args.ai_config,
            model=args.ai_model,
            timeout=args.ai_timeout,
        )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_text(result, show_research=args.show_research)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
