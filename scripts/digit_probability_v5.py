#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行probability_v5隔离开发挑战器，不读取模型状态或Frozen号码。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.digit_data import load_digit_development_csv  # noqa: E402
from src.analysis.digit_probability_v5 import (  # noqa: E402
    ProbabilityV5DevelopmentConfig,
    build_probability_v5_protocol,
    prepare_probability_v5_development_history,
    probability_v5_smoke_config,
    run_probability_v5_development,
    run_registered_probability_v5_development,
    write_probability_v5_protocol,
    write_probability_v5_report,
)
from src.lotteries import get_lottery_rule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="probability_v5隔离开发评估（不读取Frozen、不写模型状态）"
    )
    parser.add_argument("--lottery", required=True, choices=("fc3d", "pl3"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    parser.add_argument("--protocol")
    parser.add_argument(
        "--register-protocol",
        action="store_true",
        help="只写一次登记完整开发协议，不执行开发评估",
    )
    parser.add_argument("--frozen-test-periods", type=int, default=500)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="仅用Frozen之前50期验证执行链，不产生可用于晋级的证据",
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
    config = (
        probability_v5_smoke_config()
        if args.smoke
        else ProbabilityV5DevelopmentConfig()
    )
    development_available = len(development)
    development = prepare_probability_v5_development_history(development, rule, config)
    if args.register_protocol:
        if args.smoke:
            parser.error("登记协议时不得使用--smoke")
        if not args.protocol:
            parser.error("登记协议必须提供--protocol")
        protocol = build_probability_v5_protocol(
            development,
            rule,
            config,
            frozen_periods_excluded=args.frozen_test_periods,
        )
        destination = write_probability_v5_protocol(protocol, args.protocol)
        print(
            f"protocolSha256={protocol['protocolSha256']} frozenRead=false",
            flush=True,
        )
        print(destination)
        return 0
    if not args.output:
        parser.error("开发评估必须提供--output")
    if args.smoke:
        if args.protocol:
            parser.error("smoke不得绑定正式开发协议")
    else:
        if not args.protocol:
            parser.error("完整开发评估必须提供已登记的--protocol")
    print(
        f"mode=development smoke={str(args.smoke).lower()} "
        f"frozenExcluded={args.frozen_test_periods} frozenRead=false "
        f"fullPeriods={full_periods} developmentAvailable={development_available} "
        f"developmentUsed={len(development)}",
        flush=True,
    )
    runner = (
        run_probability_v5_development
        if args.smoke
        else lambda history, lottery_rule, development_config, **kwargs: (
            run_registered_probability_v5_development(
                args.protocol,
                history,
                lottery_rule,
                development_config,
                **kwargs,
            )
        )
    )
    report = runner(
        development,
        rule,
        config,
        frozen_periods_excluded=args.frozen_test_periods,
        progress_callback=lambda processed, total, issue: print(
            f"processed={processed}/{total} issue={issue}", flush=True
        ),
    )
    destination = write_probability_v5_report(report, args.output)
    payload = report.to_dict()
    print(
        f"selectedTemperature={report.selected_temperature} "
        f"developmentSignalsPassed={str(payload['developmentSignalsPassed']).lower()} "
        "promotionPassed=false recommendationEnabled=false",
        flush=True,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
