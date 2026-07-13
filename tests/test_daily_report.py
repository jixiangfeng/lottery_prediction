# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd

from src.analysis.backtest import run_fixed_candidate_backtest
from src.analysis.parameter_search import search_parameter_grid
from src.analysis.daily_report import (
    CandidateGroup,
    build_markdown_report,
    compute_basic_stats,
    generate_candidate_groups,
    select_best_parameter_candidates,
    write_daily_report,
)


def _sample_history() -> pd.DataFrame:
    rows = []
    base = list(range(1, 21))
    for idx in range(12):
        # 让 1-10 保持较热，同时每期略有轮转，便于测试统计/候选合法性
        nums = sorted(((n + idx - 1) % 80) + 1 for n in base)
        rows.append({"期数": 2024012 - idx, **{f"红球_{i+1}": nums[i] for i in range(20)}})
    return pd.DataFrame(rows)


def test_compute_basic_stats_contains_frequency_omission_and_latest_draw():
    stats = compute_basic_stats(_sample_history(), windows=(5, 10))

    assert stats.latest_issue == "2024012"
    assert len(stats.latest_numbers) == 20
    assert 5 in stats.frequency_by_window
    assert 10 in stats.frequency_by_window
    assert len(stats.current_omission) == 80
    assert len(stats.zone_distribution) == 8
    assert len(stats.tail_distribution) == 10
    assert stats.hot_numbers[0] in range(1, 81)
    assert stats.cold_numbers[0] in range(1, 81)


def test_generate_candidate_groups_are_unique_structured_and_scored():
    stats = compute_basic_stats(_sample_history(), windows=(5, 10))
    groups = generate_candidate_groups(stats, count=6, group_size=10, seed=2024013)

    assert len(groups) == 6
    assert all(isinstance(group, CandidateGroup) for group in groups)
    assert len({tuple(group.numbers) for group in groups}) == 6
    for group in groups:
        assert len(group.numbers) == 10
        assert group.numbers == sorted(group.numbers)
        assert all(1 <= number <= 80 for number in group.numbers)
        assert 0 <= group.odd_count <= 10
        assert 0 <= group.big_count <= 10
        assert group.score > 0


def test_build_and_write_markdown_report(tmp_path):
    df = _sample_history()
    stats = compute_basic_stats(df, windows=(5, 10))
    groups = generate_candidate_groups(stats, count=3, group_size=10, seed=2024013)
    backtest_summary = run_fixed_candidate_backtest(df, [group.numbers for group in groups], window=10)
    parameter_search_results = search_parameter_grid(
        df,
        stats,
        count=3,
        group_size=10,
        seed=2024013,
        window_size=5,
        step=5,
        max_windows=2,
    )

    markdown = build_markdown_report(
        stats,
        groups,
        title="快乐8测试报告",
        backtest_summary=backtest_summary,
        parameter_search_results=parameter_search_results,
    )
    assert "# 快乐8测试报告" in markdown
    assert "最新期号" in markdown
    assert "候选组选十" in markdown
    assert "最近历史固定候选回测" in markdown
    assert "收益率" in markdown
    assert "参数自动搜索" in markdown
    assert "回测提示" in markdown
    assert "2024012" in markdown

    output = write_daily_report(markdown, tmp_path, stats.latest_issue)
    assert output.exists()
    assert output.name.endswith("2024012.md")
    assert output.read_text(encoding="utf-8") == markdown


def test_select_best_parameter_candidates_promotes_search_winner():
    df = _sample_history()
    stats = compute_basic_stats(df, windows=(5, 10))
    parameter_search_results = search_parameter_grid(
        df,
        stats,
        count=3,
        group_size=10,
        seed=2024013,
        window_size=5,
        step=5,
        max_windows=2,
    )

    groups, parameter_name = select_best_parameter_candidates(stats, parameter_search_results)

    assert parameter_name == parameter_search_results[0].config.name
    assert [group.numbers for group in groups] == parameter_search_results[0].groups
    assert all(isinstance(group, CandidateGroup) for group in groups)

    markdown = build_markdown_report(
        stats,
        groups,
        title="快乐8测试报告",
        parameter_name=parameter_name,
        parameter_search_results=parameter_search_results,
    )
    assert f"主推荐参数：`{parameter_name}`" in markdown
