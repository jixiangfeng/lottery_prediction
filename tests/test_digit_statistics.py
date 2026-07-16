# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_statistics import (
    analyze_digit_history,
    classify_digit_shape,
    current_digit_omission,
    digit_consecutive_count,
    digit_mirror_count,
    digit_prime_composite_label,
    digit_sum_tail,
)
from src.lotteries import get_lottery_rule


def test_classify_digit_shape_for_three_digit_lottery():
    assert classify_digit_shape([1, 1, 1]) == "豹子"
    assert classify_digit_shape([1, 1, 2]) == "组三"
    assert classify_digit_shape([1, 2, 3]) == "组六"


def test_classify_digit_shape_for_five_digit_lottery():
    assert classify_digit_shape([1, 1, 1, 1, 1]) == "五同"
    assert classify_digit_shape([1, 1, 1, 2, 2]) == "三二"
    assert classify_digit_shape([1, 1, 2, 2, 3]) == "二二一"
    assert classify_digit_shape([1, 2, 3, 4, 5]) == "全不同"


def test_extended_digit_features_are_deterministic():
    assert digit_prime_composite_label([2, 4, 1]) == "质1合1其他1"
    assert digit_consecutive_count([1, 2, 3]) == 2
    assert digit_consecutive_count([1, 1, 2]) == 1
    assert digit_mirror_count([1, 8, 0, 9, 5]) == 2
    assert digit_sum_tail([8, 9, 7]) == 4


def test_analyze_digit_history_returns_position_sum_span_and_shape_stats():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026002", "百位": 1, "十位": 1, "个位": 2},
            {"期数": "2026001", "百位": 9, "十位": 9, "个位": 9},
        ]
    )

    result = analyze_digit_history(df, rule)

    assert result.draw_count == 3
    assert result.position_frequency["百位"][1] == 2
    assert result.position_frequency["十位"][9] == 1
    assert result.sum_distribution[6] == 1
    assert result.sum_distribution[4] == 1
    assert result.span_distribution[0] == 1
    assert result.shape_distribution["豹子"] == 1
    assert result.shape_distribution["组三"] == 1
    assert result.shape_distribution["组六"] == 1
    assert result.parity_distribution["奇2偶1"] == 2
    assert result.big_small_distribution["大0小3"] == 2


def test_current_digit_omission_is_position_aware():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026002", "百位": 4, "十位": 5, "个位": 6},
            {"期数": "2026001", "百位": 1, "十位": 8, "个位": 9},
        ]
    )

    omission = current_digit_omission(df, rule)

    assert omission["百位"][1] == 0
    assert omission["百位"][4] == 1
    assert omission["百位"][0] == 3
    assert omission["个位"][3] == 0
    assert omission["个位"][6] == 1


def test_analyze_digit_history_exposes_multi_window_omission():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": str(2026001 + index), "百位": index % 3, "十位": 4, "个位": 9}
            for index in range(12)
        ]
    )

    result = analyze_digit_history(df, rule, frequency_windows=(5, 10))
    payload = result.to_dict()

    assert result.omission_windows[5]["百位"][9] == 5
    assert result.omission_windows[10]["百位"][9] == 10
    assert result.omission_windows[5]["十位"][4] == 0
    assert payload["omissionWindows"]["10"]["百位"]["9"] == 10


def test_analyze_digit_history_adds_multi_window_smoothed_position_probabilities():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": 1,
                "十位": index % 10,
                "个位": (index + 3) % 10,
            }
            for index in range(30)
        ]
    )

    result = analyze_digit_history(df, rule)

    assert {30, 50, 100, 300} <= set(result.position_frequency_windows)
    assert result.position_frequency_windows[30]["百位"][1] == 30
    assert result.position_probabilities[30]["百位"][0] > 0
    assert abs(sum(result.position_probabilities[30]["百位"].values()) - 1.0) < 1e-9
    payload = result.to_dict()
    assert "positionFrequencyWindows" in payload
    assert "positionProbabilities" in payload


def test_analyze_digit_history_adds_smoothed_pair_and_structure_probabilities():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {
                "期数": str(2026001 + index),
                "百位": 1,
                "十位": index % 2,
                "个位": (index + 3) % 10,
            }
            for index in range(12)
        ]
    )

    result = analyze_digit_history(df, rule, frequency_windows=(5, 20))

    assert set(result.pair_probabilities[5]) == {"0-1", "0-2", "1-2"}
    assert len(result.pair_probabilities[5]["0-1"]) == 100
    assert result.pair_probabilities[5]["0-1"][(9, 9)] > 0
    assert abs(sum(result.pair_probabilities[5]["0-1"].values()) - 1.0) < 1e-9
    assert abs(sum(result.shape_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.sum_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.span_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.parity_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.big_small_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.prime_composite_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.consecutive_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.mirror_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.sum_tail_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.latest_distance_probabilities[5].values()) - 1.0) < 1e-9
    assert abs(sum(result.repeat_latest_probabilities[5].values()) - 1.0) < 1e-9

    payload = result.to_dict()
    assert payload["pairProbabilities"]["5"]["0-1"]["9,9"] > 0
    assert "shapeProbabilities" in payload
    assert "sumProbabilities" in payload
    assert "spanProbabilities" in payload
    assert "parityProbabilities" in payload
    assert "bigSmallProbabilities" in payload
    assert "primeCompositeProbabilities" in payload
    assert "consecutiveProbabilities" in payload
    assert "mirrorProbabilities" in payload
    assert "sumTailProbabilities" in payload
    assert "latestDistanceProbabilities" in payload
    assert "repeatLatestProbabilities" in payload


def test_analyze_digit_history_orders_non_padded_issues_numerically():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "9", "百位": 9, "十位": 0, "个位": 0},
            {"期数": "10", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "8", "百位": 8, "十位": 0, "个位": 0},
        ]
    )

    result = analyze_digit_history(df, rule)

    assert result.latest_issue == "10"
    assert result.latest_numbers == [1, 2, 3]
    assert result.current_omission["百位"][1] == 0
