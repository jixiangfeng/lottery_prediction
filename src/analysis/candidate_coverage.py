# -*- coding: utf-8 -*-
"""候选组覆盖率与相似度分析。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class CandidateCoverageResult:
    """候选组覆盖率与相似度指标。"""

    group_count: int
    total_unique_numbers: int
    average_overlap: float
    max_overlap: int
    most_common_numbers: list[tuple[int, int]]
    zone_coverage_count: int
    tail_coverage_count: int
    zone_distribution: list[int]
    tail_distribution: list[int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["most_common_numbers"] = [
            {"number": int(number), "count": int(count)} for number, count in self.most_common_numbers
        ]
        return {
            "groupCount": self.group_count,
            "totalUniqueNumbers": self.total_unique_numbers,
            "averageOverlap": self.average_overlap,
            "maxOverlap": self.max_overlap,
            "mostCommonNumbers": payload["most_common_numbers"],
            "zoneCoverageCount": self.zone_coverage_count,
            "tailCoverageCount": self.tail_coverage_count,
            "zoneDistribution": self.zone_distribution,
            "tailDistribution": self.tail_distribution,
        }


def _numbers(group: Any) -> list[int]:
    raw = getattr(group, "numbers", group)
    return sorted(int(number) for number in raw)


def _zone_distribution(numbers: Iterable[int]) -> list[int]:
    buckets = [0] * 8
    for number in numbers:
        buckets[(int(number) - 1) // 10] += 1
    return buckets


def _tail_distribution(numbers: Iterable[int]) -> list[int]:
    buckets = [0] * 10
    for number in numbers:
        buckets[int(number) % 10] += 1
    return buckets


def analyze_candidate_coverage(groups: Sequence[Any]) -> CandidateCoverageResult:
    """分析多组候选之间的覆盖号码数量、重合度、区间和尾数覆盖。"""

    normalized = [_numbers(group) for group in groups]
    counter: Counter[int] = Counter()
    for group in normalized:
        counter.update(group)

    overlaps = [len(set(left) & set(right)) for left, right in combinations(normalized, 2)]
    average_overlap = round(sum(overlaps) / len(overlaps), 2) if overlaps else 0.0
    max_overlap = max(overlaps, default=0)
    unique_numbers = sorted(counter)
    zone_distribution = _zone_distribution(unique_numbers)
    tail_distribution = _tail_distribution(unique_numbers)

    return CandidateCoverageResult(
        group_count=len(normalized),
        total_unique_numbers=len(unique_numbers),
        average_overlap=average_overlap,
        max_overlap=max_overlap,
        most_common_numbers=sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:10],
        zone_coverage_count=sum(1 for value in zone_distribution if value > 0),
        tail_coverage_count=sum(1 for value in tail_distribution if value > 0),
        zone_distribution=zone_distribution,
        tail_distribution=tail_distribution,
    )


def build_candidate_coverage_markdown(result: CandidateCoverageResult) -> str:
    """生成候选组覆盖率与相似度 Markdown。"""

    common = "；".join(f"{number:02d}({count}次)" for number, count in result.most_common_numbers[:8]) or "无"
    return "\n".join(
        [
            "## 候选组覆盖率 / 相似度",
            "",
            f"- 候选组数量：`{result.group_count}`",
            f"- 覆盖号码总数：`{result.total_unique_numbers}`",
            f"- 平均组间重合：`{result.average_overlap}`",
            f"- 最大组间重合：`{result.max_overlap}`",
            f"- 区间覆盖：`{result.zone_coverage_count}/8`，分布：`{'-'.join(str(v) for v in result.zone_distribution)}`",
            f"- 尾数覆盖：`{result.tail_coverage_count}/10`，分布：`{'-'.join(str(v) for v in result.tail_distribution)}`",
            f"- 高频候选号：{common}",
            "",
        ]
    )
