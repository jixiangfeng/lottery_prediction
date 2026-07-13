# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass

from src.analysis.walk_forward_parameter_weights import (
    apply_walk_forward_parameter_weights,
    parse_walk_forward_strategy_scores,
)


@dataclass(frozen=True)
class DummyConfig:
    name: str


@dataclass(frozen=True)
class DummyResult:
    config: DummyConfig
    score: float


def test_parse_walk_forward_strategy_scores_from_json_payload():
    payload = {
        "summaries": [
            {"strategy": "omission_mix", "score": -0.68},
            {"strategy": "hot_omission", "score": -0.82},
            {"strategy": "broken", "score": None},
        ]
    }

    assert parse_walk_forward_strategy_scores(payload) == {"omission_mix": -0.68, "hot_omission": -0.82}


def test_apply_walk_forward_parameter_weights_promotes_stable_forward_strategy(tmp_path):
    path = tmp_path / "walk_forward_kl8.json"
    path.write_text(
        json.dumps(
            {
                "summaries": [
                    {"strategy": "stable_forward", "score": 0.9},
                    {"strategy": "historical_best", "score": 0.1},
                ]
            }
        ),
        encoding="utf-8",
    )
    results = [DummyResult(DummyConfig("historical_best"), 1.0), DummyResult(DummyConfig("stable_forward"), 0.97)]

    adjusted, meta = apply_walk_forward_parameter_weights(results, path, alpha=0.08)

    assert adjusted[0].config.name == "stable_forward"
    assert meta["enabled"] is True
    assert meta["bestStrategy"] == "stable_forward"
    assert meta["matchedCount"] == 2


def test_apply_walk_forward_parameter_weights_falls_back_without_report(tmp_path):
    results = [DummyResult(DummyConfig("historical_best"), 1.0), DummyResult(DummyConfig("stable_forward"), 0.97)]

    adjusted, meta = apply_walk_forward_parameter_weights(results, tmp_path / "missing.json")

    assert [item.config.name for item in adjusted] == ["historical_best", "stable_forward"]
    assert meta["enabled"] is False
