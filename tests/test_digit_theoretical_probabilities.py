# -*- coding: utf-8 -*-
"""数字彩理论概率数学基线测试。"""

from __future__ import annotations

import pytest

from src.analysis.digit_statistics import (
    digit_theoretical_probability_cache_info,
    get_digit_theoretical_probabilities,
)
from src.lotteries import get_lottery_rule
from src.lotteries.base import BallSpec, LotteryRule


def test_fc3d_theoretical_shape_probabilities_are_exact():
    baseline = get_digit_theoretical_probabilities(get_lottery_rule("fc3d"))

    assert baseline["shape"] == {"豹子": 0.01, "组三": 0.27, "组六": 0.72}
    assert baseline["sampleSpaceSize"] == 1000
    assert baseline["baselineType"] == "exact_mathematical_enumeration"
    assert baseline["isPrediction"] is False


def test_theoretical_probability_domains_sum_to_one_for_fc3d_and_pl5():
    for lottery in ("fc3d", "pl5"):
        baseline = get_digit_theoretical_probabilities(get_lottery_rule(lottery))
        for feature in ("shape", "sum", "span", "parity", "bigSmall"):
            assert abs(sum(baseline[feature].values()) - 1.0) < 1e-12


def test_theoretical_probability_cache_is_reused_and_result_is_defensive_copy():
    rule = get_lottery_rule("fc3d")
    before = digit_theoretical_probability_cache_info()

    first = get_digit_theoretical_probabilities(rule)
    middle = digit_theoretical_probability_cache_info()
    first["shape"]["豹子"] = 1.0
    second = get_digit_theoretical_probabilities(rule)
    after = digit_theoretical_probability_cache_info()

    assert second["shape"]["豹子"] == 0.01
    assert middle.misses >= before.misses
    assert after.hits == middle.hits + 1


def test_theoretical_probabilities_reject_rules_without_legal_combinations():
    rule = LotteryRule(
        code="empty-space",
        display_name="无合法组合测试玩法",
        category="digit",
        source_name="test",
        draw_count=2,
        default_pick_count=2,
        ball_specs=(BallSpec("第一位", 0, 0), BallSpec("第二位", 0, 0)),
        allow_repeated=False,
        prize_mode="test",
    )

    with pytest.raises(ValueError, match="无合法组合"):
        get_digit_theoretical_probabilities(rule)
