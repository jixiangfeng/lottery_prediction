# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.digit_backtest import backtest_digit_candidates, build_digit_backtest_markdown
from src.analysis.digit_candidates import DigitCandidate, DigitCandidateResult, DigitCandidateConfig
from src.lotteries import get_lottery_rule


def _candidate(text: str) -> DigitCandidate:
    numbers = [int(ch) for ch in text]
    return DigitCandidate(numbers=numbers, text=text, sum_value=sum(numbers), span=max(numbers)-min(numbers), shape="", score=1.0)


def test_fc3d_backtest_counts_direct_and_group_hits():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([
        {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
        {"期数": "2026002", "百位": 1, "十位": 1, "个位": 2},
        {"期数": "2026001", "百位": 9, "十位": 9, "个位": 9},
    ])
    result = DigitCandidateResult("fc3d", "福彩3D", [_candidate("123"), _candidate("321"), _candidate("112")], DigitCandidateConfig(count=3))

    summary = backtest_digit_candidates(df, rule, result)

    assert summary.draw_count == 3
    assert summary.candidate_count == 3
    assert summary.direct_hits == 2
    assert summary.group_hits == 3
    assert summary.direct_hit_rate == 2 / 9
    assert summary.group_hit_rate == 3 / 9
    assert summary.rows[0].direct_hit_texts == ["123"]
    assert set(summary.rows[0].group_hit_texts) == {"123", "321"}


def test_pl5_backtest_only_counts_direct_hits():
    rule = get_lottery_rule("pl5")
    df = pd.DataFrame([
        {"期数": "2026002", "万位": 0, "千位": 1, "百位": 2, "十位": 3, "个位": 4},
        {"期数": "2026001", "万位": 9, "千位": 9, "百位": 8, "十位": 8, "个位": 7},
    ])
    result = DigitCandidateResult("pl5", "排列五", [_candidate("01234"), _candidate("43210"), _candidate("99887")], DigitCandidateConfig(count=3))

    summary = backtest_digit_candidates(df, rule, result)

    assert summary.direct_hits == 2
    assert summary.group_hits is None
    assert summary.direct_hit_rate == 2 / 6
    assert summary.rows[0].direct_hit_texts == ["01234"]


def test_build_digit_backtest_markdown_contains_summary():
    rule = get_lottery_rule("fc3d")
    df = pd.DataFrame([{"期数": "2026001", "百位": 1, "十位": 2, "个位": 3}])
    result = DigitCandidateResult("fc3d", "福彩3D", [_candidate("123")], DigitCandidateConfig(count=1))

    markdown = build_digit_backtest_markdown(backtest_digit_candidates(df, rule, result))

    assert "数字彩候选回测" in markdown
    assert "直选命中" in markdown
    assert "组选命中" in markdown
