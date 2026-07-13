# -*- coding: utf-8 -*-
"""数字彩分析报告命令入口。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_report import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
