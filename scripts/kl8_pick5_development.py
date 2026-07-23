#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行快乐8选5开发挑战器或登记只读协议。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick5_probability_v1 import (  # noqa: E402
    Kl8Pick5Config,
    assert_canonical_formal_config,
    build_kl8_protocol,
    load_kl8_development_csv,
    run_kl8_development,
    run_registered_kl8_development,
    write_kl8_protocol,
    write_kl8_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="快乐8选5严格预序开发挑战器")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    parser.add_argument("--protocol")
    parser.add_argument("--register-protocol", action="store_true")
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-research-candidates", action="store_true")
    args = parser.parse_args(argv)
    config = Kl8Pick5Config.smoke() if args.smoke else Kl8Pick5Config()
    if not args.smoke:
        assert_canonical_formal_config(config)
        if args.frozen_periods != config.frozen_periods:
            parser.error("正式登记/开发的--frozen-periods必须恰好为500")
    development, metadata = load_kl8_development_csv(
        args.csv, frozen_periods=args.frozen_periods
    )
    if len(development) < config.required_periods:
        parser.error(f"开发区至少需要{config.required_periods}期")
    locked = development.tail(config.required_periods).reset_index(drop=True)
    frozen_boundary = metadata["frozenBoundary"]
    if not isinstance(frozen_boundary, dict):
        parser.error("完整开发必须存在Frozen首末期元数据")
    print(
        f"developmentPeriods={len(locked)} frozenExcluded={args.frozen_periods} "
        "frozenRead=false validationOpened=false",
        flush=True,
    )
    if args.register_protocol:
        if args.smoke or not args.protocol or args.output:
            parser.error("登记协议必须使用完整配置、--protocol且不得同时输出报告")
        destination = write_kl8_protocol(
            build_kl8_protocol(
                locked,
                config,
                frozen_periods_excluded=args.frozen_periods,
                frozen_boundary=frozen_boundary,
            ),
            args.protocol,
        )
        print(destination)
        return 0
    if not args.output:
        parser.error("开发运行必须提供--output")
    if args.smoke:
        if args.protocol:
            parser.error("通用smoke不得声明或绑定已登记协议")
        report = run_kl8_development(
            locked,
            config,
            frozen_periods_excluded=args.frozen_periods,
            audit_research_candidates=args.audit_research_candidates,
        )
    else:
        if not args.protocol:
            parser.error("完整已登记开发必须提供只读--protocol路径")
        report = run_registered_kl8_development(
            args.protocol,
            locked,
            config,
            frozen_periods_excluded=args.frozen_periods,
            frozen_boundary=frozen_boundary,
            audit_research_candidates=args.audit_research_candidates,
        )
    destination = write_kl8_report(report, args.output)
    payload = report.to_dict()
    print(
        f"developmentSignalsPassed={str(payload['developmentSignalsPassed']).lower()} "
        f"userVisibleCandidates=0 fullPeriods={metadata['fullPeriods']} report={destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
