from __future__ import annotations

from decimal import Decimal

import pytest

from src.analysis.ssq_settlement import (
    SSQDraw,
    SSQTicket,
    TaxRule,
    settle_portfolio,
    settle_ticket,
)

DRAW = SSQDraw(red=(1, 2, 3, 4, 5, 6), blue=7)
PRIZES = {
    "一等奖": 5_000_000,
    "二等奖": 200_000,
    "三等奖": 3_000,
    "四等奖": 200,
    "五等奖": 10,
    "六等奖": 5,
}


@pytest.mark.parametrize(
    ("red", "blue", "red_hits", "blue_hit", "tier"),
    [
        ((1, 2, 3, 4, 5, 6), 7, 6, True, "一等奖"),
        ((1, 2, 3, 4, 5, 6), 8, 6, False, "二等奖"),
        ((1, 2, 3, 4, 5, 8), 7, 5, True, "三等奖"),
        ((1, 2, 3, 4, 5, 8), 8, 5, False, "四等奖"),
        ((1, 2, 3, 4, 8, 9), 7, 4, True, "四等奖"),
        ((1, 2, 3, 4, 8, 9), 8, 4, False, "五等奖"),
        ((1, 2, 3, 8, 9, 10), 7, 3, True, "五等奖"),
        ((1, 2, 8, 9, 10, 11), 7, 2, True, "六等奖"),
        ((1, 8, 9, 10, 11, 12), 7, 1, True, "六等奖"),
        ((8, 9, 10, 11, 12, 13), 7, 0, True, "六等奖"),
        ((1, 2, 3, 8, 9, 10), 8, 3, False, None),
    ],
)
def test_settle_ticket_applies_every_official_ssq_prize_rule(
    red: tuple[int, ...], blue: int, red_hits: int, blue_hit: bool, tier: str | None
) -> None:
    result = settle_ticket(SSQTicket(red=red, blue=blue), DRAW, PRIZES)

    assert result.red_hits == red_hits
    assert result.blue_hit is blue_hit
    assert result.prize_tier == tier
    expected_gross = Decimal(str(PRIZES[tier])) if tier else Decimal("0")
    assert result.gross_prize == expected_gross


def test_settle_ticket_applies_tax_to_entire_single_ticket_prize_above_threshold() -> (
    None
):
    result = settle_ticket(
        SSQTicket(red=(1, 2, 3, 4, 5, 6), blue=7),
        DRAW,
        PRIZES,
        tax_rule=TaxRule(threshold_yuan=10_000, rate=Decimal("0.20")),
    )

    assert result.gross_prize == Decimal("5000000")
    assert result.tax == Decimal("1000000.00")
    assert result.net_prize == Decimal("4000000.00")
    assert result.tax_rule == "单票单次奖金超过10000元，按奖金全额的20%计税"


def test_settle_portfolio_settles_each_ticket_not_portfolio_maximum_red_hit() -> None:
    portfolio = settle_portfolio(
        [
            SSQTicket(red=(1, 2, 3, 4, 5, 6), blue=8),  # 二等奖
            SSQTicket(red=(8, 9, 10, 11, 12, 13), blue=7),  # 六等奖
            SSQTicket(red=(1, 2, 3, 8, 9, 10), blue=8),  # 未中奖
        ],
        DRAW,
        PRIZES,
    )

    assert [ticket.prize_tier for ticket in portfolio.tickets] == [
        "二等奖",
        "六等奖",
        None,
    ]
    assert portfolio.cost == Decimal("6")
    assert portfolio.gross == Decimal("200005")
    # 二等奖单票奖金超过默认10,000元阈值，按全额20%计税。
    assert portfolio.net == Decimal("160005.00")
    assert portfolio.roi == Decimal("26666.50")
    assert portfolio.cumulative_drawdown == Decimal("2.00")
    assert portfolio.prize_breakdown == {"二等奖": 1, "六等奖": 1}


def test_settle_portfolio_reports_running_peak_to_trough_drawdown() -> None:
    portfolio = settle_portfolio(
        [
            SSQTicket(red=(1, 2, 3, 4, 8, 9), blue=8),  # 5th: net cashflow +8
            SSQTicket(red=(8, 9, 10, 11, 12, 13), blue=8),  # loss: cashflow -2
            SSQTicket(red=(8, 9, 10, 11, 12, 14), blue=8),  # loss: cashflow -2
        ],
        DRAW,
        PRIZES,
    )

    assert portfolio.cumulative_drawdown == Decimal("4")
    assert portfolio.total_profit == Decimal("4")


def test_settlement_rejects_invalid_tickets_missing_prizes_and_invalid_tax_rules() -> (
    None
):
    with pytest.raises(ValueError, match="红球必须严格升序"):
        settle_ticket(SSQTicket(red=(2, 1, 3, 4, 5, 6), blue=7), DRAW, PRIZES)
    with pytest.raises(ValueError, match="缺少官方奖级金额"):
        settle_ticket(SSQTicket(red=(1, 2, 3, 4, 5, 6), blue=7), DRAW, {})
    with pytest.raises(ValueError, match="税率必须在0到1之间"):
        TaxRule(rate=Decimal("1.01"))
