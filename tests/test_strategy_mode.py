# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.analysis.strategy_mode import parse_best_parameter_from_live_summary, select_parameter_result


@dataclass(frozen=True)
class DummyConfig:
    name: str


@dataclass(frozen=True)
class DummyResult:
    config: DummyConfig
    score: float


def _results():
    return [DummyResult(DummyConfig("auto_best"), 1.0), DummyResult(DummyConfig("manual_one"), 0.5)]


def test_select_parameter_result_auto_uses_first_result():
    selected, resolved_mode = select_parameter_result(_results(), mode="auto")

    assert selected.config.name == "auto_best"
    assert resolved_mode == "auto"


def test_select_parameter_result_manual_uses_named_strategy():
    selected, resolved_mode = select_parameter_result(_results(), mode="manual", strategy="manual_one")

    assert selected.config.name == "manual_one"
    assert resolved_mode == "manual"


def test_select_parameter_result_manual_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="未知策略参数"):
        select_parameter_result(_results(), mode="manual", strategy="missing")


def test_stable_uses_best_parameter_from_live_summary_when_available(tmp_path):
    live_summary = tmp_path / "live_summary.md"
    live_summary.write_text("- 最佳参数：`manual_one`\n", encoding="utf-8")

    selected, resolved_mode = select_parameter_result(_results(), mode="stable", live_summary_path=live_summary)

    assert selected.config.name == "manual_one"
    assert resolved_mode == "stable"


def test_stable_falls_back_to_auto_when_no_live_summary(tmp_path):
    selected, resolved_mode = select_parameter_result(_results(), mode="stable", live_summary_path=tmp_path / "missing.md")

    assert selected.config.name == "auto_best"
    assert resolved_mode == "auto_fallback"


def test_parse_best_parameter_from_live_summary():
    text = "# 快乐8实盘累计表现\n- 最佳参数：`omission_mix`\n"
    assert parse_best_parameter_from_live_summary(text) == "omission_mix"
