# -*- coding: utf-8 -*-
"""超级大乐透（DLT）独立规则定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DLTRule:
    """大乐透固定的前区五球、后区两球规则。"""

    code: str = "dlt"
    game_no: str = "85"
    display_name: str = "超级大乐透"
    front_min: int = 1
    front_max: int = 35
    front_count: int = 5
    back_min: int = 1
    back_max: int = 12
    back_count: int = 2

    def validate_draw(
        self, front: Sequence[int], back: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """严格校验并返回升序前区与后区号码。"""

        front_numbers = tuple(int(number) for number in front)
        back_numbers = tuple(int(number) for number in back)
        if len(front_numbers) != self.front_count:
            raise ValueError(f"大乐透前区数量必须为 {self.front_count}")
        if len(set(front_numbers)) != self.front_count:
            raise ValueError("大乐透前区不得重复")
        if tuple(sorted(front_numbers)) != front_numbers:
            raise ValueError("大乐透前区必须严格升序")
        if any(
            number < self.front_min or number > self.front_max
            for number in front_numbers
        ):
            raise ValueError("大乐透前区范围必须为 01-35")
        if len(back_numbers) != self.back_count:
            raise ValueError(f"大乐透后区数量必须为 {self.back_count}")
        if len(set(back_numbers)) != self.back_count:
            raise ValueError("大乐透后区不得重复")
        if tuple(sorted(back_numbers)) != back_numbers:
            raise ValueError("大乐透后区必须严格升序")
        if any(
            number < self.back_min or number > self.back_max for number in back_numbers
        ):
            raise ValueError("大乐透后区范围必须为 01-12")
        return front_numbers, back_numbers

    def format_zone(self, numbers: Sequence[int]) -> str:
        """将已校验号码格式化为两位数、空格分隔文本。"""

        return " ".join(f"{int(number):02d}" for number in numbers)


DLT_RULE = DLTRule()

__all__ = ["DLT_RULE", "DLTRule"]
