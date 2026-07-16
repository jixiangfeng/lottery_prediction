# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_candidates import DigitCandidateConfig
from src.analysis.digit_monte_carlo import simulate_digit_candidates
from src.analysis.digit_statistics import analyze_digit_history
from src.lotteries import get_lottery_rule


def test_monte_carlo_simulation_is_reproducible_and_respects_filters():
    rule = get_lottery_rule("fc3d")
    history = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 4,
                "十位": (index + 1) % 6,
                "个位": (index + 2) % 10,
            }
            for index in range(40)
        ]
    )
    config = DigitCandidateConfig(count=10, ranking_mode="ensemble")
    stats = analyze_digit_history(history, rule)

    first = simulate_digit_candidates(stats, rule, config, simulations=2000, seed=7)
    second = simulate_digit_candidates(stats, rule, config, simulations=2000, seed=7)

    assert first == second
    assert first.simulations == 2000
    assert first.accepted > 0
    assert first.scores
    assert all(0.0 < score <= 1.0 for score in first.scores.values())
    assert "000" not in first.scores


def test_monte_carlo_uses_pair_conditionals_instead_of_independent_positions():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": index % 10,
                "十位": index % 10,
                "个位": (index + 3) % 10,
            }
            for index in range(120)
        ]
    )
    config = DigitCandidateConfig(
        count=10,
        exclude_latest=False,
        sum_min=0,
        sum_max=27,
        span_min=0,
        span_max=9,
        allowed_shapes=("豹子", "组三", "组六"),
    )
    stats = analyze_digit_history(df, rule)

    result = simulate_digit_candidates(stats, rule, config, simulations=10000, seed=9)
    same_first_pair = sum(
        score for text, score in result.scores.items() if text[0] == text[1]
    )

    assert result.pair_conditioned is True
    assert result.structure_conditioned is True
    assert same_first_pair > 0.45
