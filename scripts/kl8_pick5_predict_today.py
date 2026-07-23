#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在无独立Validation/Frozen准入时安全返回空正式候选。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick5_probability_v1 import (  # noqa: E402
    RESEARCH_ONLY_WORDING,
    Kl8Pick5Config,
    load_kl8_development_csv,
    run_kl8_development,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="快乐8选5安全预测边界；绝不自动抓取或改写状态"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument("--audit-research-candidates", action="store_true")
    args = parser.parse_args(argv)
    development, metadata = load_kl8_development_csv(
        args.csv, frozen_periods=args.frozen_periods
    )
    research: list[list[int]] = []
    if args.audit_research_candidates:
        config = Kl8Pick5Config()
        if len(development) >= config.required_periods:
            report = run_kl8_development(
                development.tail(config.required_periods).reset_index(drop=True),
                config,
                frozen_periods_excluded=args.frozen_periods,
                audit_research_candidates=True,
            )
            research = report.research_candidates
    boundary_value = metadata["frozenBoundary"]
    boundary = (
        cast(Mapping[str, object], boundary_value)
        if boundary_value is not None
        else None
    )
    research_target = boundary["firstIssue"] if research and boundary else None
    payload = {
        "schemaVersion": "kl8_pick5_prediction_boundary_v1",
        "historyPeriods": metadata["fullPeriods"],
        "developmentCutoffIssue": development.iloc[-1]["issue"],
        "latestKnownIssue": boundary["lastIssue"] if boundary else None,
        "automaticFetch": False,
        "stateOverwritten": False,
        "validationOpened": False,
        "promotionPassed": False,
        "recommendationEnabled": False,
        "formalRecommendation": None,
        "userVisibleCandidates": [],
        "researchCandidates": research,
        "researchTargetIssue": research_target,
        "researchTargetKind": "locked_frozen_start_audit" if research else None,
        "researchCandidateNotice": (
            f"{RESEARCH_ONLY_WORDING}；目标为锁定Frozen首期，不是今日推荐"
            if research
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
