#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行behavioral_context_v1近期行为A/B挑战模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_behavioral_context import (  # noqa: E402
    BehavioralContextConfig,
    run_behavioral_context_challenge,
    write_behavioral_context_report,
)
from src.analysis.digit_data import load_digit_csv  # noqa: E402
from src.analysis.digit_online_gradient import OnlineGradientConfig  # noqa: E402
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--outer-periods", type=int, default=500)
    parser.add_argument("--paired-permutations", type=int, default=9999)
    args = parser.parse_args(argv)

    rule = get_lottery_rule(args.lottery)
    history = load_digit_csv(args.csv, rule)
    online = OnlineGradientConfig(
        development_end=len(history),
        outer_periods=args.outer_periods,
    )
    print(
        "effective: "
        f"lottery={args.lottery} directTopK={online.direct_top_k} "
        f"outerPeriods={online.outer_periods} behaviorL2=10 "
        "groups=A/B frozenTestRead=false currentDailyModelReplaced=false",
        flush=True,
    )
    report = run_behavioral_context_challenge(
        history,
        rule,
        BehavioralContextConfig(
            online=online,
            paired_permutations=args.paired_permutations,
        ),
    )
    destination = write_behavioral_context_report(report, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
