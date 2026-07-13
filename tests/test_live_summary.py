# -*- coding: utf-8 -*-

from src.analysis.pick_tracking import GroupEvaluation, PickEvaluation
from src.analysis.live_summary import (
    LiveSummary,
    build_live_summary_markdown,
    compute_live_summary,
    write_live_summary,
)


def _evaluation(issue: str, parameter: str, total_prize: int, hits: list[int]) -> PickEvaluation:
    return PickEvaluation(
        target_issue=issue,
        source_issue=str(int(issue) - 1),
        parameter_name=parameter,
        draw_numbers=list(range(1, 21)),
        group_results=[
            GroupEvaluation(
                index=idx + 1,
                numbers=list(range(idx + 1, idx + 11)),
                hit_numbers=list(range(1, hit + 1)),
                hit_count=hit,
                prize=0,
            )
            for idx, hit in enumerate(hits)
        ],
        total_cost=20,
        total_prize=total_prize,
        roi=round((total_prize - 20) / 20, 4),
    )


def test_compute_live_summary_aggregates_cost_prize_roi_and_hits():
    evaluations = [
        _evaluation("2024001", "omission_mix", 10, [1, 2, 5, 0, 3, 4, 5, 6, 0, 2]),
        _evaluation("2024002", "hot_heavy", 30, [2, 3, 4, 5, 6, 7, 1, 0, 2, 3]),
        _evaluation("2024003", "omission_mix", 0, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
    ]

    summary = compute_live_summary(evaluations)

    assert isinstance(summary, LiveSummary)
    assert summary.period_count == 3
    assert summary.total_cost == 60
    assert summary.total_prize == 40
    assert summary.roi == round((40 - 60) / 60, 4)
    assert summary.average_hit > 0
    assert summary.hit5_plus_count == 6
    assert summary.best_parameter == "hot_heavy"
    assert summary.max_losing_streak == 1


def test_build_and_write_live_summary_markdown(tmp_path):
    summary = compute_live_summary([
        _evaluation("2024001", "omission_mix", 10, [1, 2, 5, 0, 3, 4, 5, 6, 0, 2]),
        _evaluation("2024002", "hot_heavy", 30, [2, 3, 4, 5, 6, 7, 1, 0, 2, 3]),
    ])

    markdown = build_live_summary_markdown(summary)

    assert "# 快乐8实盘累计表现" in markdown
    assert "累计收益率" in markdown
    assert "最佳参数" in markdown
    assert "最近期号" in markdown

    output = write_live_summary(summary, tmp_path)
    assert output.name == "live_summary.md"
    assert output.read_text(encoding="utf-8") == markdown
