# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_candidates import DigitCandidateConfig
from src.analysis.digit_ml_ranker import score_digit_ranker, train_digit_ranker
from src.analysis.digit_statistics import analyze_digit_history
from src.lotteries import get_lottery_rule


def _history(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 5,
                "十位": (index * 3 + 1) % 10,
                "个位": (index * 7 + 2) % 10,
            }
            for index in range(rows)
        ]
    )


def test_lightweight_ranker_trains_on_point_in_time_features_and_scores_space():
    rule = get_lottery_rule("fc3d")
    history = _history()
    config = DigitCandidateConfig(count=10, ranking_mode="ensemble")

    ranker = train_digit_ranker(
        history,
        rule,
        config,
        min_train_size=10,
        training_periods=8,
        negative_samples=3,
        seed=9,
    )
    scores = score_digit_ranker(
        ranker,
        analyze_digit_history(history, rule),
        rule,
        config,
    )

    assert 0 < ranker.training_targets <= 8
    assert ranker.training_samples == ranker.training_targets * 4
    assert scores
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert len(set(round(value, 6) for value in scores.values())) > 1


def test_lightweight_ranker_returns_none_when_history_is_insufficient():
    rule = get_lottery_rule("fc3d")
    ranker = train_digit_ranker(
        _history(8),
        rule,
        DigitCandidateConfig(),
        min_train_size=10,
        training_periods=5,
        negative_samples=2,
    )

    assert ranker is None
