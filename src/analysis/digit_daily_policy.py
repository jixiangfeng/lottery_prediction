# -*- coding: utf-8 -*-
"""数字彩日常候选的开奖前组合安全策略。"""

from __future__ import annotations

from collections.abc import Iterable


def _partition_daily_candidates(
    ranked_candidates: Iterable[str],
    *,
    latest_exact: str,
    maximum_triples: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(latest_exact) != 3 or not latest_exact.isdigit():
        raise ValueError("latest_exact必须是三位数字")
    if maximum_triples < 0:
        raise ValueError("maximum_triples不能为负")

    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    triple_count = 0
    for candidate in ranked_candidates:
        if len(candidate) != 3 or not candidate.isdigit():
            raise ValueError("候选必须是三位数字")
        if candidate in seen:
            continue
        seen.add(candidate)
        is_triple = len(set(candidate)) == 1
        if candidate == latest_exact or (is_triple and triple_count >= maximum_triples):
            rejected.append(candidate)
            continue
        accepted.append(candidate)
        triple_count += int(is_triple)
    return tuple(accepted), tuple(rejected)


def rank_daily_candidates(
    ranked_candidates: Iterable[str],
    *,
    latest_exact: str,
    maximum_triples: int = 1,
) -> tuple[str, ...]:
    """按日常策略重排完整候选，确保任意TopK前缀与日常输出一致。"""

    accepted, rejected = _partition_daily_candidates(
        ranked_candidates,
        latest_exact=latest_exact,
        maximum_triples=maximum_triples,
    )
    return (*accepted, *rejected)


def select_daily_candidates(
    ranked_candidates: Iterable[str],
    *,
    latest_exact: str,
    top_k: int = 50,
    maximum_triples: int = 1,
) -> tuple[str, ...]:
    """按模型顺序排除上期原号、限制豹子并补足固定TopK。"""
    if top_k <= 0 or maximum_triples < 0:
        raise ValueError("top_k必须大于零且maximum_triples不能为负")

    accepted, _ = _partition_daily_candidates(
        ranked_candidates,
        latest_exact=latest_exact,
        maximum_triples=maximum_triples,
    )
    if len(accepted) >= top_k:
        return accepted[:top_k]

    raise ValueError(f"过滤后不足{top_k}个候选")


__all__ = ["rank_daily_candidates", "select_daily_candidates"]
