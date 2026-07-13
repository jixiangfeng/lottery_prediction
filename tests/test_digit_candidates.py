# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_candidates import DigitCandidateConfig, generate_digit_candidates
from src.analysis.digit_statistics import analyze_digit_history, classify_digit_shape
from src.lotteries import get_lottery_rule


def test_generate_digit_candidates_for_fc3d_respects_filters_and_shape():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026005", "百位": 1, "十位": 2, "个位": 3},
            {"期数": "2026004", "百位": 1, "十位": 1, "个位": 2},
            {"期数": "2026003", "百位": 4, "十位": 5, "个位": 6},
            {"期数": "2026002", "百位": 7, "十位": 8, "个位": 9},
            {"期数": "2026001", "百位": 0, "十位": 0, "个位": 0},
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(count=8, sum_min=6, sum_max=18, span_min=1, span_max=8, allowed_shapes=("组六", "组三"))

    result = generate_digit_candidates(stats, rule, config=config, seed=7)

    assert len(result.candidates) == 8
    assert result.rule_code == "fc3d"
    for candidate in result.candidates:
        assert 6 <= candidate.sum_value <= 18
        assert 1 <= candidate.span <= 8
        assert candidate.shape in {"组六", "组三"}
        assert candidate.shape == classify_digit_shape(candidate.numbers)
        assert len(candidate.text) == 3


def test_default_fc3d_candidates_exclude_latest_and_baozi():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "百位": 8, "十位": 1, "个位": 8},
            {"期数": "2026002", "百位": 1, "十位": 1, "个位": 1},
            {"期数": "2026001", "百位": 2, "十位": 3, "个位": 4},
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(stats, rule, config=DigitCandidateConfig(count=8), seed=11)

    assert all(candidate.text != "818" for candidate in result.candidates)
    assert all(candidate.shape != "豹子" for candidate in result.candidates)
    assert sum(1 for candidate in result.candidates if candidate.shape == "组三") <= 3
    assert all(6 <= candidate.sum_value <= 21 for candidate in result.candidates)
    assert all(2 <= candidate.span <= 9 for candidate in result.candidates)


def test_generate_digit_candidates_for_pl5_keeps_leading_zero_text():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "万位": 0, "千位": 1, "百位": 2, "十位": 3, "个位": 4},
            {"期数": "2026002", "万位": 9, "千位": 9, "百位": 8, "十位": 8, "个位": 7},
            {"期数": "2026001", "万位": 5, "千位": 6, "百位": 7, "十位": 8, "个位": 9},
        ]
    )
    stats = analyze_digit_history(df, rule)
    config = DigitCandidateConfig(count=6, sum_min=5, sum_max=35, allowed_shapes=("全不同", "二二一", "二一一一"), exclude_latest=False)

    result = generate_digit_candidates(stats, rule, config=config, seed=3)

    assert len(result.candidates) == 6
    assert result.rule_code == "pl5"
    assert all(len(candidate.text) == 5 for candidate in result.candidates)
    assert any(candidate.text.startswith("0") for candidate in result.candidates)


def test_default_pl5_candidates_exclude_latest_and_heavy_repeats():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame(
        [
            {"期数": "2026003", "万位": 1, "千位": 2, "百位": 4, "十位": 4, "个位": 4},
            {"期数": "2026002", "万位": 9, "千位": 9, "百位": 8, "十位": 8, "个位": 7},
            {"期数": "2026001", "万位": 0, "千位": 1, "百位": 2, "十位": 3, "个位": 4},
        ]
    )
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(stats, rule, config=DigitCandidateConfig(count=8), seed=13)

    assert all(candidate.text != "12444" for candidate in result.candidates)
    assert all(candidate.shape in {"全不同", "二一一一", "二二一"} for candidate in result.candidates)
    assert all(10 <= candidate.sum_value <= 35 for candidate in result.candidates)
    assert all(3 <= candidate.span <= 9 for candidate in result.candidates)


def test_digit_candidate_result_to_dict_is_report_friendly():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([{"期数": "2026001", "百位": 1, "十位": 2, "个位": 3}])
    stats = analyze_digit_history(df, rule)

    result = generate_digit_candidates(stats, rule, config=DigitCandidateConfig(count=2), seed=1)
    payload = result.to_dict()

    assert payload["ruleCode"] == "fc3d"
    assert len(payload["candidates"]) == 2
    assert {"text", "numbers", "sum", "span", "shape", "score"} <= set(payload["candidates"][0])
