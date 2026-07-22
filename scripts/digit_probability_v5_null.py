#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行probability_v5全流程均匀随机模拟；默认正式模式要求协议和开发报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_development_csv  # noqa: E402
from src.analysis.digit_probability_v5 import (  # noqa: E402
    ProbabilityV5DevelopmentConfig,
    load_and_verify_probability_v5_protocol,
    prepare_probability_v5_development_history,
    probability_v5_smoke_config,
    run_probability_v5_development,
)
from src.analysis.digit_probability_v5_null import (  # noqa: E402
    run_probability_v5_null_simulation,
    write_probability_v5_null_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="probability_v5全流程均匀随机模拟")
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--reference-report")
    parser.add_argument("--frozen-test-periods", type=int, default=500)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="执行少量确定性模拟验证代码和并行链，不形成正式证据",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    rule = get_lottery_rule(args.lottery)
    development, full_periods = load_digit_development_csv(
        args.csv,
        rule,
        frozen_test_periods=args.frozen_test_periods,
    )
    development_available = len(development)
    if args.smoke:
        if args.protocol or args.reference_report:
            parser.error("随机模拟smoke不得绑定正式协议或开发报告")
        config = probability_v5_smoke_config()
        development = prepare_probability_v5_development_history(
            development, rule, config
        )
        reference = run_probability_v5_development(
            development,
            rule,
            config,
            frozen_periods_excluded=args.frozen_test_periods,
            include_period_details=False,
        ).to_dict()
        iterations = args.iterations if args.iterations is not None else 2
        protocol_sha256 = None
        formal = False
    else:
        if not args.protocol or not args.reference_report:
            parser.error("正式随机模拟必须提供--protocol和--reference-report")
        if not args.checkpoint_dir:
            parser.error("正式随机模拟必须提供--checkpoint-dir")
        config = ProbabilityV5DevelopmentConfig()
        development = prepare_probability_v5_development_history(
            development, rule, config
        )
        protocol = load_and_verify_probability_v5_protocol(
            args.protocol,
            development,
            rule,
            config,
            frozen_periods_excluded=args.frozen_test_periods,
        )
        protocol_sha256 = str(protocol["protocolSha256"])
        raw_reference = json.loads(
            Path(args.reference_report).read_text(encoding="utf-8")
        )
        if not isinstance(raw_reference, dict):
            raise ValueError("开发报告格式无效")
        reference = raw_reference
        iterations = (
            args.iterations
            if args.iterations is not None
            else config.required_null_simulations
        )
        formal = True
    print(
        f"mode={'formal' if formal else 'smoke'} fullPeriods={full_periods} "
        f"developmentAvailable={development_available} "
        f"developmentPeriods={len(development)} frozenExcluded={args.frozen_test_periods} "
        f"iterations={iterations} workers={args.workers} frozenRead=false",
        flush=True,
    )
    progress_step = max(1, iterations // 20)
    report = run_probability_v5_null_simulation(
        reference,
        lottery=args.lottery,
        config=config,
        history_periods=len(development),
        frozen_periods_excluded=args.frozen_test_periods,
        iterations=iterations,
        workers=args.workers,
        formal=formal,
        protocol_sha256=protocol_sha256,
        checkpoint_dir=args.checkpoint_dir,
        progress_callback=lambda processed, total: (
            print(f"simulations={processed}/{total}", flush=True)
            if processed % progress_step == 0 or processed == total
            else None
        ),
    )
    destination = write_probability_v5_null_report(report, args.output)
    print(
        "nullSimulationPassed="
        f"{str(report.summary['nullSimulationPassed']).lower()} "
        "promotionPassed=false recommendationEnabled=false",
        flush=True,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
