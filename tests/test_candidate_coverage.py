# -*- coding: utf-8 -*-

from src.analysis.candidate_coverage import analyze_candidate_coverage, build_candidate_coverage_markdown


def test_analyze_candidate_coverage_counts_unique_and_overlap():
    groups = [
        [1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8],
        [1, 8, 9, 10, 11],
    ]

    result = analyze_candidate_coverage(groups)

    assert result.group_count == 3
    assert result.total_unique_numbers == 11
    assert result.average_overlap == 1.33
    assert result.max_overlap == 2
    assert result.most_common_numbers[:3] == [(1, 2), (4, 2), (5, 2)]


def test_analyze_candidate_coverage_reports_zone_and_tail_coverage():
    groups = [[1, 10, 11, 20, 21, 30, 31, 40, 71, 80]]

    result = analyze_candidate_coverage(groups)

    assert result.zone_coverage_count == 5
    assert result.tail_coverage_count == 2
    assert result.zone_distribution == [2, 2, 2, 2, 0, 0, 0, 2]


def test_build_candidate_coverage_markdown_contains_key_metrics():
    result = analyze_candidate_coverage([[1, 2, 3], [3, 4, 5]])

    markdown = build_candidate_coverage_markdown(result)

    assert "## 候选组覆盖率 / 相似度" in markdown
    assert "覆盖号码总数" in markdown
    assert "最大组间重合" in markdown
