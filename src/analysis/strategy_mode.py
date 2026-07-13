# -*- coding: utf-8 -*-
"""日报主推荐策略模式选择。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence


def parse_best_parameter_from_live_summary(text: str) -> str | None:
    """从 live_summary.md 中解析“最佳参数”。"""

    match = re.search(r"最佳参数：`([^`]+)`", text)
    return match.group(1) if match else None


def _read_stable_parameter(live_summary_path: Path | str | None) -> str | None:
    if live_summary_path is None:
        return None
    path = Path(live_summary_path)
    if not path.exists():
        return None
    return parse_best_parameter_from_live_summary(path.read_text(encoding="utf-8"))


def _find_by_name(results: Sequence[Any], name: str) -> Any | None:
    for result in results:
        if result.config.name == name:
            return result
    return None


def select_parameter_result(
    results: Sequence[Any],
    *,
    mode: str = "auto",
    strategy: str | None = None,
    live_summary_path: Path | str | None = None,
) -> tuple[Any, str]:
    """根据 auto/manual/stable 模式选择参数搜索结果。

    返回 `(result, resolved_mode)`；当 stable 缺少可用实盘最佳参数时回退为 `auto_fallback`。
    """

    if not results:
        raise ValueError("parameter_search_results 不能为空")
    normalized_mode = (mode or "auto").lower()
    if normalized_mode == "auto":
        return results[0], "auto"
    if normalized_mode == "manual":
        if not strategy:
            raise ValueError("manual 模式必须指定 strategy")
        selected = _find_by_name(results, strategy)
        if selected is None:
            names = ", ".join(result.config.name for result in results)
            raise ValueError(f"未知策略参数：{strategy}，可选：{names}")
        return selected, "manual"
    if normalized_mode == "stable":
        stable_name = _read_stable_parameter(live_summary_path)
        if stable_name:
            selected = _find_by_name(results, stable_name)
            if selected is not None:
                return selected, "stable"
        return results[0], "auto_fallback"
    raise ValueError(f"未知策略模式：{mode}，可选 auto/manual/stable")
