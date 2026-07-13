# -*- coding: utf-8 -*-
"""基于真实复盘表现的参数权重调整。"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence


def parse_parameter_roi_from_live_summary(text: str) -> dict[str, float]:
    """从 live_summary.md 的“参数表现”表格中解析参数累计 ROI。"""

    output: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped or "参数" in stripped:
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 2:
            continue
        parameter, roi_text = parts[0], parts[1]
        match = re.search(r"([-+]?\d+(?:\.\d+)?)%", roi_text)
        if parameter and match:
            output[parameter] = round(float(match.group(1)) / 100.0, 6)
    return output


def _load_parameter_roi(live_summary_path: Path | str | None) -> dict[str, float]:
    if live_summary_path is None:
        return {}
    path = Path(live_summary_path)
    if not path.exists():
        return {}
    return parse_parameter_roi_from_live_summary(path.read_text(encoding="utf-8"))


def _with_score(result: Any, score: float) -> Any:
    try:
        return replace(result, score=score)
    except TypeError:
        # 测试或第三方对象不一定是 dataclass，退回原对象类型构造。
        return type(result)(result.config, score)


def apply_live_parameter_weights(
    results: Sequence[Any],
    live_summary_path: Path | str | None,
    *,
    alpha: float = 0.25,
) -> tuple[list[Any], dict[str, Any]]:
    """按实盘 ROI 给参数搜索分数加权并重新排序。

    adjusted_score = historical_score + alpha * live_roi
    无 live_summary 或无匹配参数时保持原排序。
    """

    parameter_roi = _load_parameter_roi(live_summary_path)
    if not parameter_roi:
        return list(results), {"enabled": False, "alpha": alpha, "parameterRoi": {}}

    adjusted = []
    matched = 0
    for result in results:
        name = result.config.name
        roi = parameter_roi.get(name)
        if roi is None:
            adjusted.append(result)
            continue
        matched += 1
        adjusted.append(_with_score(result, round(float(result.score) + alpha * roi, 6)))

    if matched == 0:
        return list(results), {"enabled": False, "alpha": alpha, "parameterRoi": parameter_roi}
    return sorted(adjusted, key=lambda item: item.score, reverse=True), {
        "enabled": True,
        "alpha": alpha,
        "parameterRoi": parameter_roi,
        "matchedCount": matched,
    }
