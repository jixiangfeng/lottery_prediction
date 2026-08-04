# -*- coding: utf-8 -*-
"""D8+7 的严格前序官方逐票奖级回测；只读，不触碰任何前瞻链。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import cast

from src.analysis.ssq_d8_plus7_singles import TICKET_COUNT, build_d8_plus7_singles
from src.analysis.ssq_ensemble_v1 import FixedEnsembleState
from src.analysis.ssq_history import SSQDraw
from src.analysis.ssq_prizegrades import SSQPrizeGradeRecord
from src.analysis.ssq_settlement import SSQDraw as SettlementDraw
from src.analysis.ssq_settlement import SSQTicket, settle_portfolio

WARMUP_DRAWS = 120
TICKET_PRICE = Decimal("2")
GRADE_NAMES = ("一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖")


def _official_prizes(record: SSQPrizeGradeRecord) -> dict[str, int]:
    grades = [grade.grade for grade in record.prizegrades]
    if len(grades) != 6 or set(grades) != set(range(1, 7)):
        raise ValueError(f"双色球期号{record.issue}官方奖级不完整")
    return {GRADE_NAMES[grade.grade - 1]: grade.amount for grade in record.prizegrades}


def _tickets(portfolio: Mapping[str, object]) -> tuple[SSQTicket, ...]:
    raw_tickets = portfolio.get("expandedTickets")
    if not isinstance(raw_tickets, list):
        raise ValueError("D8+7展开票结构非法")
    tickets: list[SSQTicket] = []
    for item in raw_tickets:
        if not isinstance(item, Mapping):
            raise ValueError("D8+7展开票项非法")
        red, blue = item.get("red"), item.get("blue")
        if not isinstance(red, list) or not isinstance(blue, int):
            raise ValueError("D8+7展开票字段非法")
        tickets.append(SSQTicket(tuple(cast(list[int], red)), blue))
    if len(tickets) != TICKET_COUNT:
        raise ValueError("D8+7展开票数量非法")
    return tuple(tickets)


def evaluate_d8_plus7_official_backtest(
    draws: Sequence[SSQDraw], prize_records: Sequence[SSQPrizeGradeRecord]
) -> dict[str, object]:
    """严格先构造35张 D8+7 票，再按同期官方奖级逐票结算。

    每个评估期仅使用此前 120 期历史更新的状态预测；构造、结算和状态更新按期
    串行执行。缺少或不完整的官方奖级立即失败，不产生填补或推断结果。
    """

    ordered = sorted(draws, key=lambda draw: int(draw.issue))
    if len(ordered) <= WARMUP_DRAWS:
        raise ValueError("D8+7官方结算至少需要120期预热加1期")
    if len({draw.issue for draw in ordered}) != len(ordered):
        raise ValueError("D8+7官方结算历史包含重复期号")

    prizes_by_issue: dict[str, SSQPrizeGradeRecord] = {}
    for record in prize_records:
        if record.issue in prizes_by_issue:
            raise ValueError(f"D8+7官方结算官方奖级包含重复期号：{record.issue}")
        prizes_by_issue[record.issue] = record
    evaluated = ordered[WARMUP_DRAWS:]
    missing = [draw.issue for draw in evaluated if draw.issue not in prizes_by_issue]
    if missing:
        raise ValueError(f"D8+7官方结算缺少官方奖级：{', '.join(missing[:5])}")

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
        tickets = _tickets(build_d8_plus7_singles(red, blue))
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
        "schemaVersion": "ssq_d8_plus7_official_backtest_v1",
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
            "ticketsPerIssue": TICKET_COUNT,
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


__all__ = ["evaluate_d8_plus7_official_backtest"]
