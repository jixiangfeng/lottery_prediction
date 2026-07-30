# -*- coding: utf-8 -*-
"""双色球单式票逐票判奖与组合票收益结算。

本模块只在内存中计算：不读取或写入前瞻 state，也不生成预测号码。奖级金额由调用者
传入对应期次的官方单注奖金；组合票绝不以其中的最大红球命中数替代逐票判奖。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

from src.lotteries.ssq import SSQ_RULE

Money = Decimal
PRIZE_TIERS = ("一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖")


@dataclass(frozen=True)
class SSQTicket:
    """一张双色球单式票，红球必须为六个严格升序的号码。"""

    red: tuple[int, ...]
    blue: int

    def normalized(self) -> "SSQTicket":
        """校验并返回符合双色球规则的票面。"""

        red, blue = SSQ_RULE.validate_draw(self.red, self.blue)
        return SSQTicket(red=red, blue=blue)


@dataclass(frozen=True)
class SSQDraw:
    """一期双色球开奖号码。"""

    red: tuple[int, ...]
    blue: int

    def normalized(self) -> "SSQDraw":
        """校验并返回符合双色球规则的开奖球号。"""

        red, blue = SSQ_RULE.validate_draw(self.red, self.blue)
        return SSQDraw(red=red, blue=blue)


@dataclass(frozen=True)
class TaxRule:
    """单票奖金个税规则。

    默认规则为：单票单次奖金不超过 10,000 元免税；超过时，按该票奖金全额的
    20% 计税。可用 ``threshold_yuan`` 和 ``rate`` 显式覆盖，适配不同口径。
    """

    threshold_yuan: Money = Decimal("10000")
    rate: Money = Decimal("0.20")

    def __post_init__(self) -> None:
        threshold = _money(self.threshold_yuan, "免税阈值")
        rate = _money(self.rate, "税率")
        if threshold < 0:
            raise ValueError("免税阈值不得小于0")
        if rate < 0 or rate > 1:
            raise ValueError("税率必须在0到1之间")
        object.__setattr__(self, "threshold_yuan", threshold)
        object.__setattr__(self, "rate", rate)

    def describe(self) -> str:
        """返回结算结果中可审计的中文税务口径。"""

        return (
            f"单票单次奖金超过{_format_money(self.threshold_yuan)}元，"
            f"按奖金全额的{_format_percent(self.rate)}计税"
        )


@dataclass(frozen=True)
class TicketSettlement:
    """一张票的逐票判奖结果；``net_prize`` 为税后奖金，未扣购票成本。"""

    ticket: SSQTicket
    red_hits: int
    blue_hit: bool
    prize_tier: str | None
    gross_prize: Money
    tax: Money
    net_prize: Money
    tax_rule: str


@dataclass(frozen=True)
class PortfolioSettlement:
    """组合票的逐票结果与汇总；``net`` 为税后奖金，``total_profit`` 扣除成本。"""

    tickets: tuple[TicketSettlement, ...]
    cost: Money
    gross: Money
    net: Money
    roi: Money
    total_profit: Money
    cumulative_drawdown: Money
    prize_breakdown: dict[str, int]


def settle_ticket(
    ticket: SSQTicket,
    draw: SSQDraw,
    official_prizes: Mapping[str, object],
    *,
    tax_rule: TaxRule | None = None,
) -> TicketSettlement:
    """按双色球官方奖级规则结算一张单式票。

    ``official_prizes`` 必须包含一等奖至六等奖的当期单注奖金（元）。税务按每张票
    独立计算，默认使用 :class:`TaxRule` 的 10,000 元/20% 全额计税口径。
    """

    normalized_ticket = ticket.normalized()
    normalized_draw = draw.normalized()
    prizes = _normalize_prizes(official_prizes)
    applied_tax_rule = tax_rule or TaxRule()

    red_hits = len(set(normalized_ticket.red).intersection(normalized_draw.red))
    blue_hit = normalized_ticket.blue == normalized_draw.blue
    prize_tier = _prize_tier(red_hits, blue_hit)
    gross_prize = prizes[prize_tier] if prize_tier is not None else Decimal("0")
    tax = (
        gross_prize * applied_tax_rule.rate
        if gross_prize > applied_tax_rule.threshold_yuan
        else Decimal("0")
    )
    return TicketSettlement(
        ticket=normalized_ticket,
        red_hits=red_hits,
        blue_hit=blue_hit,
        prize_tier=prize_tier,
        gross_prize=gross_prize,
        tax=tax,
        net_prize=gross_prize - tax,
        tax_rule=applied_tax_rule.describe(),
    )


def settle_portfolio(
    tickets: Sequence[SSQTicket],
    draw: SSQDraw,
    official_prizes: Mapping[str, object],
    *,
    ticket_price_yuan: object = 2,
    tax_rule: TaxRule | None = None,
) -> PortfolioSettlement:
    """逐张结算组合票并汇总成本、奖金、ROI、回撤和奖级分布。

    ``roi`` 使用 ``(税后奖金 - 成本) / 成本``；空组合票的 ROI 为 0。
    ``cumulative_drawdown`` 是按传入票面顺序、每张票先支付成本再获得该票税后奖金的
    累计损益曲线的最大峰谷回撤，因而能揭示单张逐票路径，不能由最大红球命中替代。
    """

    price = _money(ticket_price_yuan, "单票价格")
    if price <= 0:
        raise ValueError("单票价格必须大于0")
    settled = tuple(
        settle_ticket(ticket, draw, official_prizes, tax_rule=tax_rule)
        for ticket in tickets
    )
    cost = price * len(settled)
    gross = sum((item.gross_prize for item in settled), Decimal("0"))
    net = sum((item.net_prize for item in settled), Decimal("0"))
    total_profit = net - cost
    roi = total_profit / cost if cost else Decimal("0")

    breakdown: dict[str, int] = {}
    running_profit = Decimal("0")
    peak_profit = Decimal("0")
    maximum_drawdown = Decimal("0")
    for item in settled:
        running_profit += item.net_prize - price
        peak_profit = max(peak_profit, running_profit)
        maximum_drawdown = max(maximum_drawdown, peak_profit - running_profit)
        if item.prize_tier is not None:
            breakdown[item.prize_tier] = breakdown.get(item.prize_tier, 0) + 1

    return PortfolioSettlement(
        tickets=settled,
        cost=cost,
        gross=gross,
        net=net,
        roi=roi,
        total_profit=total_profit,
        cumulative_drawdown=maximum_drawdown,
        prize_breakdown=breakdown,
    )


def _prize_tier(red_hits: int, blue_hit: bool) -> str | None:
    """返回官方双色球单式票奖级。"""

    if red_hits == 6:
        return "一等奖" if blue_hit else "二等奖"
    if red_hits == 5:
        return "三等奖" if blue_hit else "四等奖"
    if red_hits == 4:
        return "四等奖" if blue_hit else "五等奖"
    if red_hits == 3 and blue_hit:
        return "五等奖"
    if blue_hit and red_hits in (0, 1, 2):
        return "六等奖"
    return None


def _normalize_prizes(official_prizes: Mapping[str, object]) -> dict[str, Money]:
    missing = [tier for tier in PRIZE_TIERS if tier not in official_prizes]
    if missing:
        raise ValueError(f"缺少官方奖级金额: {', '.join(missing)}")
    prizes = {
        tier: _money(official_prizes[tier], f"{tier}奖金") for tier in PRIZE_TIERS
    }
    if any(amount < 0 for amount in prizes.values()):
        raise ValueError("官方奖级金额不得小于0")
    return prizes


def _money(value: object, name: str) -> Money:
    if isinstance(value, bool):
        raise ValueError(f"{name}必须是有限非负金额")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name}必须是有限非负金额") from error
    if not amount.is_finite():
        raise ValueError(f"{name}必须是有限非负金额")
    return amount


def _format_money(value: Money) -> str:
    return format(value, "f").rstrip("0").rstrip(".") if value % 1 else str(int(value))


def _format_percent(value: Money) -> str:
    return f"{_format_money(value * 100)}%"


__all__ = [
    "PortfolioSettlement",
    "SSQDraw",
    "SSQTicket",
    "TaxRule",
    "TicketSettlement",
    "settle_portfolio",
    "settle_ticket",
]
