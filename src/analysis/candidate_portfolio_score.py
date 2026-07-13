# -*- coding: utf-8 -*-
"""候选组合多目标评分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.analysis.candidate_coverage import CandidateCoverageResult


@dataclass(frozen=True)
class CandidatePortfolioScore:
    """10 组候选整体质量评分。"""

    group_count: int
    final_score: float
    grade: str
    average_group_score: float
    group_quality_score: float
    coverage_score: float
    overlap_score: float
    max_overlap_score: float
    zone_score: float
    tail_score: float
    shape_balance_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "groupCount": self.group_count,
            "finalScore": self.final_score,
            "grade": self.grade,
            "averageGroupScore": self.average_group_score,
            "groupQualityScore": self.group_quality_score,
            "coverageScore": self.coverage_score,
            "overlapScore": self.overlap_score,
            "maxOverlapScore": self.max_overlap_score,
            "zoneScore": self.zone_score,
            "tailScore": self.tail_score,
            "shapeBalanceScore": self.shape_balance_score,
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _grade(score: float) -> str:
    if score >= 80:
        return "优秀"
    if score >= 65:
        return "良好"
    if score >= 50:
        return "一般"
    return "偏弱"


def _numbers(group: Any) -> list[int]:
    return [int(value) for value in getattr(group, "numbers", group)]


def _shape_balance(groups: Sequence[Any], group_size: int) -> float:
    """单组奇偶/大小形态均衡度，极端 9:1、8:2 会被明显降权。"""

    if not groups or group_size <= 0:
        return 0.0
    target = group_size / 2
    scores: list[float] = []
    for group in groups:
        numbers = _numbers(group)
        odd_count = sum(1 for number in numbers if number % 2 == 1)
        big_count = sum(1 for number in numbers if number > 40)
        odd_score = _clamp(1.0 - abs(odd_count - target) / target)
        big_score = _clamp(1.0 - abs(big_count - target) / target)
        # 两个维度取低值，避免“奇偶极端但大小正常”的组合被高估。
        scores.append(min(odd_score, big_score))
    return round(sum(scores) / len(scores), 4)


def score_candidate_portfolio(
    groups: Sequence[Any],
    coverage: CandidateCoverageResult,
    *,
    number_range_size: int = 80,
    group_size: int = 10,
) -> CandidatePortfolioScore:
    """综合单组质量、覆盖率、重合度、区间/尾数覆盖生成整体评分。"""

    group_count = len(groups)
    if group_count == 0:
        return CandidatePortfolioScore(0, 0.0, "偏弱", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    scores = [float(getattr(group, "score", 0.0)) for group in groups]
    average_group_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    # CandidateGroup.score 通常是统计权重和，量级可能大于 100；这里只把 >=100 视为满分。
    group_quality_score = _clamp(average_group_score / 100.0)

    coverage_score = _clamp(coverage.total_unique_numbers / number_range_size) if number_range_size else 0.0
    overlap_score = _clamp(1.0 - coverage.average_overlap / group_size) if group_size else 0.0
    max_overlap_score = _clamp(1.0 - coverage.max_overlap / group_size) if group_size else 0.0
    zone_score = _clamp(coverage.zone_coverage_count / 8.0)
    tail_score = _clamp(coverage.tail_coverage_count / 10.0)
    shape_balance_score = _shape_balance(groups, group_size)

    final = (
        0.20 * group_quality_score
        + 0.22 * coverage_score
        + 0.18 * overlap_score
        + 0.10 * max_overlap_score
        + 0.10 * zone_score
        + 0.10 * tail_score
        + 0.10 * shape_balance_score
    ) * 100

    final_score = round(final, 2)
    return CandidatePortfolioScore(
        group_count=group_count,
        final_score=final_score,
        grade=_grade(final_score),
        average_group_score=average_group_score,
        group_quality_score=round(group_quality_score, 4),
        coverage_score=round(coverage_score, 4),
        overlap_score=round(overlap_score, 4),
        max_overlap_score=round(max_overlap_score, 4),
        zone_score=round(zone_score, 4),
        tail_score=round(tail_score, 4),
        shape_balance_score=round(shape_balance_score, 4),
    )


def build_portfolio_score_markdown(score: CandidatePortfolioScore) -> str:
    """生成候选组合整体评分 Markdown。"""

    return "\n".join(
        [
            "## 候选组合总评分",
            "",
            f"- 最终评分：`{score.final_score}` / 100（{score.grade}）",
            f"- 单组质量：`{score.group_quality_score}`",
            f"- 覆盖率：`{score.coverage_score}`",
            f"- 平均低重合：`{score.overlap_score}`",
            f"- 最大低重合：`{score.max_overlap_score}`",
            f"- 区间覆盖：`{score.zone_score}`",
            f"- 尾数覆盖：`{score.tail_score}`",
            f"- 单组形态均衡：`{score.shape_balance_score}`",
        ]
    )
