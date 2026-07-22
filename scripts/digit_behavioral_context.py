#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行behavioral_context_v4极简近期、同位重合与零豹子A/B/C挑战模型。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_behavioral_context import (  # noqa: E402
    BEHAVIORAL_V4_FEATURE_NAMES,
    BehavioralContextConfig,
    run_behavioral_context_challenge,
    write_behavioral_context_report,
)
from src.analysis.digit_data import load_digit_development_csv  # noqa: E402
from src.analysis.digit_learned_features import BEHAVIORAL_FEATURE_NAMES  # noqa: E402
from src.analysis.digit_online_gradient import OnlineGradientConfig  # noqa: E402
from src.lotteries import get_lottery_rule  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--outer-periods", type=int)
    scope.add_argument(
        "--all-development-blocks",
        action="store_true",
        help="扫描Frozen以前可用的全部完整500期块",
    )
    parser.add_argument("--frozen-test-periods", type=int, default=500)
    parser.add_argument("--paired-permutations", type=int, default=9999)
    parser.add_argument(
        "--behavior-features",
        nargs="+",
        choices=BEHAVIORAL_FEATURE_NAMES,
        default=BEHAVIORAL_V4_FEATURE_NAMES,
        help="参与挑战的行为特征；默认使用v4两项极简特征，挑战组同时排除全部豹子",
    )
    args = parser.parse_args(argv)
    if len(set(args.behavior_features)) != len(args.behavior_features):
        parser.error("behavior-features不得重复")

    rule = get_lottery_rule(args.lottery)
    history, full_periods = load_digit_development_csv(
        args.csv,
        rule,
        frozen_test_periods=args.frozen_test_periods,
    )
    first_eligible_index = 150 + 300 + 100
    maximum_outer_periods = len(history) - first_eligible_index
    if args.all_development_blocks:
        outer_periods = maximum_outer_periods // 500 * 500
        evaluation_end = first_eligible_index + outer_periods
    else:
        outer_periods = args.outer_periods or 500
        evaluation_end = len(history)
    if outer_periods <= 0 or outer_periods > maximum_outer_periods:
        parser.error(
            "开发历史不足：outer-periods必须位于1到" f"{max(0, maximum_outer_periods)}"
        )
    online = OnlineGradientConfig(
        development_end=evaluation_end,
        outer_periods=outer_periods,
    )
    print(
        "effective: "
        f"lottery={args.lottery} directTopK={online.direct_top_k} "
        f"outerPeriods={online.outer_periods} behaviorL2=2 fullRows={full_periods} "
        f"developmentRows={len(history)} evaluationEnd={evaluation_end} "
        f"frozenExcluded={args.frozen_test_periods} "
        f"behaviorFeatures={','.join(args.behavior_features)} "
        "baselineMaxTriples=1 challengerMaxTriples=0 "
        "dailyPolicy=true groups=A/B/C primary=C frozenTestRead=false "
        "currentDailyModelReplaced=false",
        flush=True,
    )
    started = time.monotonic()

    def show_progress(payload: dict[str, object]) -> None:
        processed = int(payload["processedOuterPeriods"])
        total = int(payload["totalOuterPeriods"])
        elapsed = max(time.monotonic() - started, 1e-9)
        remaining = (total - processed) * elapsed / max(processed, 1)
        print(
            "progress: "
            f"lottery={args.lottery} processed={processed}/{total} "
            f"blocks={payload['completedFixedBlocks']}/{payload['totalFixedBlocks']} "
            f"targetIssue={payload['targetIssue']} elapsedSeconds={elapsed:.1f} "
            f"etaSeconds={remaining:.1f}",
            flush=True,
        )

    report = run_behavioral_context_challenge(
        history,
        rule,
        BehavioralContextConfig(
            online=online,
            behavioral_feature_names=tuple(args.behavior_features),
            paired_permutations=args.paired_permutations,
        ),
        progress_callback=show_progress,
    )
    report["dataBoundary"] = {
        "fullPeriods": full_periods,
        "developmentPeriods": len(history),
        "frozenPeriodsExcluded": args.frozen_test_periods,
        "evaluationEndIndex": evaluation_end,
        "trailingDevelopmentPeriodsExcluded": len(history) - evaluation_end,
        "outerPeriods": outer_periods,
        "allDevelopmentBlocks": args.all_development_blocks,
    }
    destination = write_behavioral_context_report(report, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
