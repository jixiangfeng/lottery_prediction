# -*- coding: utf-8 -*-
"""三种数字彩理论概率最小示例。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.digit_statistics import (  # noqa: E402
    get_digit_theoretical_probabilities,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def print_theoretical_summary() -> None:
    """打印福彩3D、排列三、排列五的精确数学基线。

    示例：调用 print_theoretical_summary()。
    """

    for code in ("fc3d", "pl3", "pl5"):
        rule = get_lottery_rule(code)
        probabilities = get_digit_theoretical_probabilities(rule)
        shapes = "，".join(
            f"{name} {probability:.2%}"
            for name, probability in probabilities["shape"].items()
        )
        print(
            f"{rule.display_name}: 样本空间 {probabilities['sampleSpaceSize']}，形态 {shapes}"
        )


if __name__ == "__main__":
    print_theoretical_summary()
