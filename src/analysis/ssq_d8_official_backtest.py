# -*- coding: utf-8 -*-
"""D8 的严格前序官方逐票奖级回测；只读，不触碰任何前瞻链。"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Sequence, cast

from src.analysis.ssq_d8_b35_support import build_diversified_portfolio_v2
from src.analysis.ssq_ensemble_v1 import FixedEnsembleState
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_prizegrades import SSQPrizeGradeRecord
from src.analysis.ssq_settlement import SSQDraw as SettlementDraw
from src.analysis.ssq_settlement import SSQTicket, settle_portfolio
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    build_small_compound_8red1blue_v1,
)

WARMUP_DRAWS = 120
TICKET_PRICE = Decimal("2")
GRADE_NAMES = ("一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖")


def _official_prizes(record: SSQPrizeGradeRecord) -> dict[str, int]:
    grades = [grade.grade for grade in record.prizegrades]
    if len(grades) != 6 or set(grades) != set(range(1, 7)):
        raise ValueError(f"双色球期号{record.issue}官方奖级不完整")
    return {GRADE_NAMES[grade.grade - 1]: grade.amount for grade in record.prizegrades}


def evaluate_d8_official_backtest(
    draws: Sequence[SSQDraw], prize_records: Sequence[SSQPrizeGradeRecord]
) -> dict[str, object]:
    """严格先构造再逐票按同期官方奖级结算 D8；任一奖级缺失即失败关闭。"""
    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) <= WARMUP_DRAWS:
        raise ValueError("D8官方结算至少需要120期预热加1期")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("D8官方结算历史包含重复期号")
    prizes_by_issue = {record.issue: record for record in prize_records}
    evaluated = ordered[WARMUP_DRAWS:]
    missing = [draw.issue for draw in evaluated if draw.issue not in prizes_by_issue]
    if missing:
        raise ValueError(f"D8官方结算缺少官方奖级：{', '.join(missing[:5])}")

    state = FixedEnsembleState()
    for draw in ordered[:WARMUP_DRAWS]:
        state.score_then_update(draw)
    gross = Decimal("0")
    net = Decimal("0")
    cost = Decimal("0")
    running_profit = Decimal("0")
    peak_profit = Decimal("0")
    maximum_drawdown = Decimal("0")
    breakdown = {tier: 0 for tier in GRADE_NAMES}
    per_issue: list[dict[str, object]] = []
    for draw in evaluated:
        red, blue, _ = state.predict()
        reference = build_diversified_portfolio_v2(red, blue)
        d8 = build_small_compound_8red1blue_v1(red, blue, reference)
        raw_tickets = d8["expandedTickets"]
        if not isinstance(raw_tickets, list):
            raise ValueError("D8展开票结构非法")
        tickets = tuple(
            SSQTicket(
                tuple(cast(list[int], cast(Mapping[str, object], item)["red"])),
                cast(int, cast(Mapping[str, object], item)["blue"]),
            )
            for item in raw_tickets
            if isinstance(item, Mapping)
        )
        if len(tickets) != 28:
            raise ValueError("D8展开票数量非法")
        settlement = settle_portfolio(
            tickets,
            SettlementDraw(draw.red, draw.blue),
            _official_prizes(prizes_by_issue[draw.issue]),
            ticket_price_yuan=TICKET_PRICE,
        )
        gross += settlement.gross
        net += settlement.net
        cost += settlement.cost
        running_profit += settlement.total_profit
        peak_profit = max(peak_profit, running_profit)
        maximum_drawdown = max(maximum_drawdown, peak_profit - running_profit)
        for tier, count in settlement.prize_breakdown.items():
            breakdown[tier] += count
        per_issue.append(
            {
                "issue": draw.issue,
                "date": draw.draw_date,
                "tickets": len(tickets),
                "costYuan": f"{settlement.cost:.2f}",
                "grossPrizeYuan": f"{settlement.gross:.2f}",
                "netPrizeYuan": f"{settlement.net:.2f}",
                "prizeTierDistribution": settlement.prize_breakdown,
                "officialPrizeRawHash": prizes_by_issue[draw.issue].raw_hash,
            }
        )
        state.score_then_update(draw)
    return {
        "schemaVersion": "ssq_d8_official_backtest_v1",
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": "uniform_abstain",
        "history": {
            "warmupPeriods": WARMUP_DRAWS,
            "evaluatedPeriods": len(evaluated),
            "firstEvaluatedIssue": evaluated[0].issue,
            "lastEvaluatedIssue": evaluated[-1].issue,
        },
        "summary": {
            "ticketsPerIssue": 28,
            "totalCostYuan": f"{cost:.2f}",
            "grossPrizeYuan": f"{gross:.2f}",
            "netPrizeYuan": f"{net:.2f}",
            "netProfitYuan": f"{(net - cost):.2f}",
            "returnPerYuan": f"{(net / cost):.4f}" if cost else None,
            "roi": f"{((net - cost) / cost):.4f}" if cost else None,
            "maximumDrawdownYuan": f"{maximum_drawdown:.2f}",
            "prizeTierDistribution": breakdown,
        },
        "perIssue": per_issue,
    }


__all__ = ["evaluate_d8_official_backtest"]
