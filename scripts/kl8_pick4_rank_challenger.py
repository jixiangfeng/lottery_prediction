#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行快乐8Pick4固定排名挑战器。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick4_rank_challenger import (  # noqa: E402
    Kl8Pick4RankConfig,
    run_pick4_rank_challenger,
    write_pick4_rank_report,
)
from src.analysis.kl8_pick5_probability_v1 import (  # noqa: E402
    load_kl8_development_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="快乐8Pick4固定LambdaRank@4挑战器；开发复用证据，不读取Frozen"
    )
    parser.add_argument("--csv", default="data/kl8/kl8.csv")
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument(
        "--output", default="reports/development/kl8_pick4_rank_challenger_v2.json"
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    development, metadata = load_kl8_development_csv(
        args.csv, frozen_periods=args.frozen_periods
    )
    if args.smoke:
        if len(development) != 40:
            parser.error("smoke固定要求40期开发数据")
        config = Kl8Pick4RankConfig(
            initial_train=20,
            evaluation_periods=20,
            refit_interval=10,
            n_estimators=8,
            num_leaves=5,
            max_depth=3,
            min_child_samples=10,
            n_jobs=args.n_jobs,
        )
    else:
        config = Kl8Pick4RankConfig(n_jobs=args.n_jobs)
        config.validate_history_length(len(development))
    boundary_value = metadata["frozenBoundary"]
    if not isinstance(boundary_value, Mapping):
        parser.error("Pick4挑战必须存在Frozen边界")
    frozen_boundary = cast(Mapping[str, object], boundary_value)
    print(
        f"developmentPeriods={len(development)} frozenExcluded={args.frozen_periods} "
        "frozenRead=false evidenceStatus=exploratory_post_failure_reused_development",
        flush=True,
    )
    report = run_pick4_rank_challenger(
        development,
        config,
        frozen_periods_excluded=args.frozen_periods,
        frozen_boundary=frozen_boundary,
    )
    destination = write_pick4_rank_report(report, args.output)
    print(
        json.dumps(
            {
                "developmentGatePassed": report["developmentGatePassed"],
                "promotionPassed": False,
                "recommendationEnabled": False,
                "output": str(destination),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
