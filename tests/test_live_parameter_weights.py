# -*- coding: utf-8 -*-

from dataclasses import dataclass

from src.analysis.live_parameter_weights import apply_live_parameter_weights, parse_parameter_roi_from_live_summary


@dataclass(frozen=True)
class DummyConfig:
    name: str


@dataclass(frozen=True)
class DummyResult:
    config: DummyConfig
    score: float


def test_parse_parameter_roi_from_live_summary_table():
    text = """
# 快乐8实盘累计表现

## 参数表现

| 参数 | 累计收益率 |
|---|---:|
| omission_mix | 12.50% |
| hot_heavy | -20.00% |
"""

    roi = parse_parameter_roi_from_live_summary(text)

    assert roi == {"omission_mix": 0.125, "hot_heavy": -0.2}


def test_apply_live_parameter_weights_boosts_better_live_parameter(tmp_path):
    live_summary = tmp_path / "live_summary.md"
    live_summary.write_text(
        "| 参数 | 累计收益率 |\n|---|---:|\n| stable_one | 40.00% |\n| historical_best | -10.00% |\n",
        encoding="utf-8",
    )
    results = [DummyResult(DummyConfig("historical_best"), 1.0), DummyResult(DummyConfig("stable_one"), 0.95)]

    adjusted, meta = apply_live_parameter_weights(results, live_summary, alpha=0.25)

    assert adjusted[0].config.name == "stable_one"
    assert meta["enabled"] is True
    assert meta["parameterRoi"]["stable_one"] == 0.4


def test_apply_live_parameter_weights_falls_back_when_no_live_summary(tmp_path):
    results = [DummyResult(DummyConfig("historical_best"), 1.0), DummyResult(DummyConfig("stable_one"), 0.95)]

    adjusted, meta = apply_live_parameter_weights(results, tmp_path / "missing.md")

    assert [item.config.name for item in adjusted] == ["historical_best", "stable_one"]
    assert meta["enabled"] is False
