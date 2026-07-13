# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_statistics import analyze_digit_history, classify_digit_shape, current_digit_omission
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
