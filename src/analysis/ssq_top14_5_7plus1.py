from __future__ import annotations

from collections.abc import Sequence

from src.analysis.ssq_d8_b35_support import ranked_balls


def build_top14_five_7plus1(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> dict[str, object]:
    """从D8 Top14构造5张7红+1蓝；仅为研究影子，不生成35注。"""
    if len(red_probabilities) != 33 or len(blue_probabilities) != 16:
        raise ValueError("概率维度非法")
    ranked = ranked_balls(red_probabilities, 33)
    core = tuple(ranked[:8])
    boundary = tuple(ranked[8:13])
    blue = tuple(ranked_balls(blue_probabilities, 16)[:5])
    tickets = tuple(
        (tuple(sorted((*core[:-2], edge))), blue[i]) for i, edge in enumerate(boundary)
    )
    return {
        "schemaVersion": "ssq_top14_5_7plus1_v1",
        "researchOnly": True,
        "predictionClaim": False,
        "redCore": list(core),
        "boundaryTop5": list(boundary),
        "blueTop5": list(blue),
        "removedCoreRed": list(core[-2:]),
        "tickets": [{"red": list(r), "blue": b} for r, b in tickets],
        "costYuan": 70,
        "formalRecommendationStatus": "uniform_abstain",
    }
