# -*- coding: utf-8 -*-
"""数字彩日常候选的开奖前组合安全策略。"""

from __future__ import annotations

from collections.abc import Iterable


def select_daily_candidates(
    ranked_candidates: Iterable[str],
    *,
    latest_exact: str,
    top_k: int = 50,
    maximum_triples: int = 1,
) -> tuple[str, ...]:
    """按模型顺序排除上期原号、限制豹子并补足固定TopK。"""
    if len(latest_exact) != 3 or not latest_exact.isdigit():
        raise ValueError("latest_exact必须是三位数字")
    if top_k <= 0 or maximum_triples < 0:
        raise ValueError("top_k必须大于零且maximum_triples不能为负")

    selected: list[str] = []
    seen: set[str] = set()
    triple_count = 0
    for candidate in ranked_candidates:
        if len(candidate) != 3 or not candidate.isdigit():
            raise ValueError("候选必须是三位数字")
        if candidate == latest_exact or candidate in seen:
            continue
        is_triple = len(set(candidate)) == 1
        if is_triple and triple_count >= maximum_triples:
            continue
        selected.append(candidate)
        seen.add(candidate)
        triple_count += int(is_triple)
        if len(selected) == top_k:
            return tuple(selected)

    raise ValueError(f"过滤后不足{top_k}个候选")


__all__ = ["select_daily_candidates"]
