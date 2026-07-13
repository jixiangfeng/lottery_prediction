# -*- coding: utf-8 -*-
"""快乐8每日报告命令入口。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.daily_report import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
