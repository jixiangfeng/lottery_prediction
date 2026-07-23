#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行快乐8 v2 隔离探索性特征发现。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_feature_discovery_v2 import (  # noqa: E402
    fixed_config_for_development_length,
    run_kl8_feature_discovery_v2,
    write_kl8_feature_discovery_report,
)
from src.analysis.kl8_pick5_probability_v1 import (  # noqa: E402
    load_kl8_development_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="快乐8 v2 探索性特征发现（不解析Frozen号码）"
    )
    parser.add_argument("--csv", default="data/kl8/kl8.csv")
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument(
        "--output", default="reports/development/kl8_feature_discovery_v2.json"
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    development, metadata = load_kl8_development_csv(
        args.csv, frozen_periods=args.frozen_periods
    )
    config = fixed_config_for_development_length(len(development), n_jobs=args.n_jobs)
    frozen_boundary = metadata["frozenBoundary"]
    if not isinstance(frozen_boundary, dict):
        parser.error("v2探索必须存在Frozen首末期边界")
    print(
        f"developmentPeriods={len(development)} frozenExcluded={args.frozen_periods} "
        "frozenRead=false evidenceStatus=exploratory_feature_discovery_only",
        flush=True,
    )
    report = run_kl8_feature_discovery_v2(
        development,
        config,
        frozen_periods_excluded=args.frozen_periods,
        frozen_boundary=frozen_boundary,
    )
    destination = write_kl8_feature_discovery_report(report, args.output)
    print(
        f"selectedFeatureSet={report['selectedFeatureSet']} promotionPassed=false "
        f"recommendationEnabled=false report={destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
