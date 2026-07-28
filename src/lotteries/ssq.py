# -*- coding: utf-8 -*-
"""双色球（SSQ）独立规则定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SSQRule:
    """双色球固定规则，不复用数字位彩票的数据通道。"""

    code: str = "ssq"
    display_name: str = "双色球"
    red_min: int = 1
    red_max: int = 33
    red_count: int = 6
    blue_min: int = 1
    blue_max: int = 16
    blue_count: int = 1

    def validate_draw(
        self, red: Sequence[int], blue: int
    ) -> tuple[tuple[int, ...], int]:
        """校验并返回规范的六个升序红球和一个蓝球。"""

        red_numbers = tuple(int(number) for number in red)
        blue_number = int(blue)
        if len(red_numbers) != self.red_count:
            raise ValueError(f"双色球红球数量必须为 {self.red_count}")
        if len(set(red_numbers)) != self.red_count:
            raise ValueError("双色球红球不得重复")
        if tuple(sorted(red_numbers)) != red_numbers:
            raise ValueError("双色球红球必须严格升序")
        if any(
            number < self.red_min or number > self.red_max for number in red_numbers
        ):
            raise ValueError("双色球红球范围必须为 01-33")
        if blue_number < self.blue_min or blue_number > self.blue_max:
            raise ValueError("双色球蓝球范围必须为 01-16")
        return red_numbers, blue_number

    def format_red(self, red: Sequence[int]) -> str:
        """把红球格式化为规范的空格分隔两位文本。"""

        red_numbers, _ = self.validate_draw(red, self.blue_min)
        return " ".join(f"{number:02d}" for number in red_numbers)

    def format_blue(self, blue: int) -> str:
        """把蓝球格式化为两位文本。"""

        blue_number = int(blue)
        if blue_number < self.blue_min or blue_number > self.blue_max:
            raise ValueError("双色球蓝球范围必须为 01-16")
        return f"{blue_number:02d}"


SSQ_RULE = SSQRule()


__all__ = ["SSQ_RULE", "SSQRule"]
