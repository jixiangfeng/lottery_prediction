# -*- coding: utf-8 -*-
"""基于快乐8逐期前推回测的参数排序加权。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence


def parse_walk_forward_strategy_scores(payload: dict[str, Any]) -> dict[str, float]:
    """从 walk_forward_kl8.json 中提取策略综合分。"""

    output: dict[str, float] = {}
    for row in payload.get("summaries", []):
        strategy = str(row.get("strategy", "")).strip()
        if not strategy:
            continue
        try:
            output[strategy] = float(row.get("score"))
        except (TypeError, ValueError):
            continue
    return output


def _load_walk_forward_scores(path: Path | str | None) -> dict[str, float]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return parse_walk_forward_strategy_scores(payload)


def _with_score(result: Any, score: float) -> Any:
    try:
        return replace(result, score=score)
    except TypeError:
        return type(result)(result.config, score)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {name: 0.5 for name in scores}
    return {name: round((score - low) / (high - low), 6) for name, score in scores.items()}


def apply_walk_forward_parameter_weights(
    results: Sequence[Any],
    walk_forward_json_path: Path | str | None,
    *,
    alpha: float = 0.08,
) -> tuple[list[Any], dict[str, Any]]:
    """按逐期前推回测综合分对参数结果做保守加权。

    使用归一化分数而非原始 ROI，避免偶发高奖或全负收益把日报排序拉偏：
    adjusted_score = historical_score + alpha * (normalized_walk_forward_score - 0.5)
    """

    strategy_scores = _load_walk_forward_scores(walk_forward_json_path)
    if not strategy_scores:
        return list(results), {"enabled": False, "alpha": alpha, "strategyScores": {}}
    normalized_scores = _normalize_scores(strategy_scores)
    adjusted = []
    matched = 0
    for result in results:
        name = result.config.name
        normalized = normalized_scores.get(name)
        if normalized is None:
            adjusted.append(result)
            continue
        matched += 1
        bonus = alpha * (normalized - 0.5)
        adjusted.append(_with_score(result, round(float(result.score) + bonus, 6)))
    if matched == 0:
        return list(results), {
            "enabled": False,
            "alpha": alpha,
            "strategyScores": strategy_scores,
            "normalizedScores": normalized_scores,
        }
    return sorted(adjusted, key=lambda item: item.score, reverse=True), {
        "enabled": True,
        "alpha": alpha,
        "matchedCount": matched,
        "strategyScores": strategy_scores,
        "normalizedScores": normalized_scores,
        "bestStrategy": max(strategy_scores, key=lambda name: strategy_scores[name]),
    }
