# -*- coding: utf-8 -*-

import pandas as pd

from src.analysis.backtest import (
    KL8_SELECT10_PRIZE_TABLE,
    BacktestSummary,
    build_backtest_markdown,
    run_fixed_candidate_backtest,
)


def _history_for_backtest() -> pd.DataFrame:
    rows = [
        {"期数": 2024004, **{f"红球_{i+1}": n for i, n in enumerate(range(1, 21))}},
        {"期数": 2024003, **{f"红球_{i+1}": n for i, n in enumerate(range(6, 26))}},
        {"期数": 2024002, **{f"红球_{i+1}": n for i, n in enumerate(range(21, 41))}},
        {"期数": 2024001, **{f"红球_{i+1}": n for i, n in enumerate(range(41, 61))}},
    ]
    return pd.DataFrame(rows)


def test_run_fixed_candidate_backtest_counts_hits_cost_and_prize():
    groups = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    summary = run_fixed_candidate_backtest(_history_for_backtest(), groups, ticket_price=2)

    assert isinstance(summary, BacktestSummary)
    assert summary.draw_count == 4
    assert summary.group_count == 1
    assert summary.total_bets == 4
    assert summary.total_cost == 8
    assert summary.hit_distribution[10] == 1
    assert summary.hit_distribution[5] == 1
    assert summary.hit_distribution[0] == 2
    assert summary.total_prize == KL8_SELECT10_PRIZE_TABLE[10] + KL8_SELECT10_PRIZE_TABLE[5] + KL8_SELECT10_PRIZE_TABLE[0] * 2
    assert summary.roi == round((summary.total_prize - summary.total_cost) / summary.total_cost, 4)
    # 快乐8选十中0也有返奖，因此这组样例没有连续“未中奖”。
    assert summary.max_miss_streak == 0


def test_build_backtest_markdown_contains_distribution_and_roi():
    summary = run_fixed_candidate_backtest(
        _history_for_backtest(),
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        ticket_price=2,
    )

    markdown = build_backtest_markdown(summary)

    assert "## 最近历史固定候选回测" in markdown
    assert "投入" in markdown
    assert "返奖" in markdown
    assert "收益率" in markdown
    assert "中10" in markdown
    assert "最大连续未中奖" in markdown
