#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行快乐8选5全流程随机零假设模拟。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick5_null import (  # noqa: E402
    FORMAL_MIN_ITERATIONS,
    run_formal_kl8_null,
    run_kl8_null_smoke,
    write_kl8_null_report,
)
from src.analysis.kl8_pick5_probability_v1 import (  # noqa: E402
    Kl8Pick5Config,
    assert_canonical_formal_config,
    load_kl8_development_csv,
    run_kl8_development,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="快乐8选5全流程均匀随机模拟")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--reference-report")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--frozen-periods", type=int, default=500)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = Kl8Pick5Config.smoke() if args.smoke else Kl8Pick5Config()
    if not args.smoke:
        assert_canonical_formal_config(config)
        if args.frozen_periods != config.frozen_periods:
            parser.error("正式null的--frozen-periods必须恰好为500")
    iterations = (
        args.iterations
        if args.iterations is not None
        else (
            2
            if args.smoke
            else max(FORMAL_MIN_ITERATIONS, config.required_null_iterations)
        )
    )
    if not args.smoke and iterations < max(
        FORMAL_MIN_ITERATIONS, config.required_null_iterations
    ):
        parser.error("正式随机模拟至少5000次")
    development, metadata = load_kl8_development_csv(
        args.csv, frozen_periods=args.frozen_periods
    )
    locked = development.tail(config.required_periods).reset_index(drop=True)
    if args.smoke:
        if args.protocol or args.reference_report:
            parser.error("smoke不得绑定正式协议或报告")
        reference = run_kl8_development(
            locked, config, frozen_periods_excluded=args.frozen_periods
        ).to_dict()
        report = run_kl8_null_smoke(
            reference,
            config=config,
            history_periods=len(locked),
            frozen_periods_excluded=args.frozen_periods,
            iterations=iterations,
            workers=args.workers,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        if not args.protocol or not args.reference_report:
            parser.error("正式模式必须提供只读--protocol和--reference-report")
        frozen_boundary = metadata["frozenBoundary"]
        if not isinstance(frozen_boundary, dict):
            parser.error("正式null必须存在Frozen首末期元数据")
        report = run_formal_kl8_null(
            args.protocol,
            args.reference_report,
            locked,
            config=config,
            frozen_periods_excluded=args.frozen_periods,
            frozen_boundary=frozen_boundary,
            iterations=iterations,
            workers=args.workers,
            checkpoint_dir=args.checkpoint_dir,
        )
    write_kl8_null_report(report, args.output)
    print(
        f"iterations={iterations} evidenceStatus={report.to_dict()['evidenceStatus']} promotionPassed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
