# -*- coding: utf-8 -*-
"""多彩种规则基础定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BallSpec:
    """单个号码位/号码池定义。"""

    name: str
    min_number: int
    max_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minNumber": self.min_number,
            "maxNumber": self.max_number,
        }


@dataclass(frozen=True)
class LotteryRule:
    """单个彩票玩法的规则元数据。"""

    code: str
    display_name: str
    category: str
    source_name: str
    draw_count: int
    default_pick_count: int
    ball_specs: tuple[BallSpec, ...]
    allow_repeated: bool
    prize_mode: str
    description: str = ""

    @property
    def number_columns(self) -> list[str]:
        return [spec.name for spec in self.ball_specs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "displayName": self.display_name,
            "category": self.category,
            "sourceName": self.source_name,
            "drawCount": self.draw_count,
            "defaultPickCount": self.default_pick_count,
            "numberColumns": self.number_columns,
            "allowRepeated": self.allow_repeated,
            "prizeMode": self.prize_mode,
            "description": self.description,
            "ballSpecs": [spec.to_dict() for spec in self.ball_specs],
        }


def validate_numbers(rule: LotteryRule, numbers: Sequence[int]) -> list[int]:
    """按玩法规则校验一期开奖结果/候选号码。"""

    normalized = [int(number) for number in numbers]
    if len(normalized) != rule.draw_count:
        raise ValueError(
            f"{rule.display_name}号码数量必须为 {rule.draw_count}，收到 {len(normalized)}"
        )
    if not rule.allow_repeated and len(set(normalized)) != len(normalized):
        raise ValueError(f"{rule.display_name}不允许重复号码")
    for index, number in enumerate(normalized):
        spec = (
            rule.ball_specs[index]
            if len(rule.ball_specs) == rule.draw_count
            else rule.ball_specs[0]
        )
        if number < spec.min_number or number > spec.max_number:
            raise ValueError(
                f"{rule.display_name}{spec.name}号码范围必须在 {spec.min_number}-{spec.max_number}，收到 {number}"
            )
    return normalized
