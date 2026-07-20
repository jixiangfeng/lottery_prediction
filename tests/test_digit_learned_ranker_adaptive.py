# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import digit_learned_ranker_adaptive as adaptive
from src.analysis.digit_learned_features import LearnedFeatureConfig
from src.analysis.digit_learned_ranker import (
    LearnedRankerParams,
    generate_learned_ranker_daily,
    probabilities_from_scores,
    save_params,
)
from src.lotteries import get_lottery_rule


def _history(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "期数": f"{2026000 + index}",
                "日期": f"2026-01-{(index % 28) + 1:02d}",
                "百位": index % 10,
                "十位": (index * 3) % 10,
                "个位": (index * 7) % 10,
            }
            for index in range(1, periods + 1)
        ]
    )


def test_uniform_shrinkage_is_exact_and_validated():
    scores = np.linspace(-3.0, 3.0, 1000)
    probabilities = probabilities_from_scores(
        scores, temperature=0.2, uniform_shrinkage=0.0
    )
    assert np.allclose(probabilities, np.full(1000, 0.001))
    assert math.isclose(float(probabilities.sum()), 1.0)
    with pytest.raises(ValueError, match="0..1"):
        LearnedRankerParams(uniform_shrinkage=1.1)
    with pytest.raises(ValueError, match="0..1"):
        probabilities_from_scores(scores, temperature=1.0, uniform_shrinkage=-0.1)


def test_params_round_trip_preserves_uniform_shrinkage():
    params = LearnedRankerParams(uniform_shrinkage=0.25)
    assert LearnedRankerParams.from_dict(params.to_dict()) == params
    legacy = params.to_dict()
    legacy.pop("uniform_shrinkage")
    assert LearnedRankerParams.from_dict(legacy).uniform_shrinkage == 1.0


def test_adaptive_research_predicts_every_period_and_retrains_by_block(monkeypatch):
    calls = []

    def fake_select(chronological, rule, block_start, config):
        calls.append((block_start, len(chronological)))
        return adaptive.AdaptiveSelection(
            block_start_index=block_start,
            history_end_index=block_start - 1,
            inner_history_start_index=block_start - config.training_lookback,
            selected_params=LearnedRankerParams(
                uniform_shrinkage=0.75,
                direct_top_k=config.direct_top_k,
                group_top_k=config.group_top_k,
                position_pool_size=config.position_pool_size,
            ),
            feature_config=LearnedFeatureConfig(),
            search_objective=-0.01,
            validation_objective=-0.01,
            mean_log_loss=math.log(1000),
            mean_brier_score=0.999,
            expected_calibration_error=0.0,
            stable_blocks=0,
            abstained=True,
            reasons=("无信号",),
        )

    monkeypatch.setattr(adaptive, "_select_for_block", fake_select)
    report = adaptive.run_adaptive_research(
        _history(230),
        get_lottery_rule("fc3d"),
        adaptive.AdaptiveResearchConfig(
            development_end=220,
            outer_periods=20,
            retrain_interval=10,
            training_lookback=180,
            inner_validation_periods=20,
            inner_stride=10,
            min_train_size=150,
            random_trials=2,
            local_trials=0,
        ),
    )

    assert calls == [(200, 220), (210, 220)]
    assert [item.target_index for item in report.predictions] == list(range(200, 220))
    assert all(item.abstained for item in report.predictions)
    assert all(item.uniform_shrinkage == 0.0 for item in report.predictions)
    assert all(
        math.isclose(item.actual_probability, 0.001) for item in report.predictions
    )
    assert report.frozen_test_read is False
    payload = report.to_dict()
    assert payload["metrics"]["periods"] == 20
    assert payload["metrics"]["abstentionRate"] == 1.0
    assert payload["outerEndIndex"] == 220

    calls.clear()
    frozen_mutated = _history(230)
    frozen_columns = ["百位", "十位", "个位"]
    frozen_mutated[frozen_columns] = frozen_mutated[frozen_columns].astype(object)
    frozen_mutated.loc[220:, frozen_columns] = "UNREAD"
    repeated = adaptive.run_adaptive_research(
        frozen_mutated,
        get_lottery_rule("fc3d"),
        adaptive.AdaptiveResearchConfig(
            development_end=220,
            outer_periods=20,
            retrain_interval=10,
            training_lookback=180,
            inner_validation_periods=20,
            inner_stride=10,
            min_train_size=150,
            random_trials=2,
            local_trials=0,
        ),
    )
    assert calls == [(200, 220), (210, 220)]
    assert repeated.to_dict() == report.to_dict()


def test_daily_abstains_when_uniform_shrinkage_is_zero(tmp_path: Path):
    csv_path = tmp_path / "fc3d.csv"
    params_path = tmp_path / "params.json"
    _history(180).sort_values("期数", ascending=False).to_csv(
        csv_path, index=False, encoding="utf-8"
    )
    save_params(LearnedRankerParams(uniform_shrinkage=0.0), params_path)

    _, json_path, _ = generate_learned_ranker_daily(
        "fc3d", csv_path, params_path, output_dir=tmp_path / "reports"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["abstained"] is True
    assert payload["noSignal"] is True
    assert payload["uniformShrinkage"] == 0.0
    assert payload["activation"]["activeDirect"] is False
    assert payload["activation"]["activeGroup"] is False
    assert payload["activation"]["activePosition"] is False
    assert payload["plan"]["mainRecommendation"]["directCandidates"] == []
