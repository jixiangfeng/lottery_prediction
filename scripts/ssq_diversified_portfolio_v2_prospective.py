#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球分散组合 v2 的仅未来 HMAC 前瞻链命令行。"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.ssq_diversified_portfolio_v2_prospective import (  # noqa: E402
    DEFAULT_CANONICAL_CSV,
    DEFAULT_ENSEMBLE_REPORT,
    DEFAULT_STATE_DIR,
    create_snapshot,
    prospective_status,
    register_prospective,
    update_prospective,
)

KEY_ENV = "SSQ_DIVERSIFIED_PORTFOLIO_V2_HMAC_KEY_FILE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, include_report: bool) -> None:
        command.add_argument("--csv", default=str(DEFAULT_CANONICAL_CSV))
        command.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
        command.add_argument(
            "--hmac-key-file",
            help=f"外部HMAC密钥文件；也可设置环境变量{KEY_ENV}",
        )
        if include_report:
            command.add_argument(
                "--ensemble-report", default=str(DEFAULT_ENSEMBLE_REPORT)
            )

    common(
        subparsers.add_parser("register", help="只登记唯一未来边界，不生成快照"),
        include_report=True,
    )
    common(
        subparsers.add_parser("snapshot", help="生成唯一0000预开奖快照"),
        include_report=True,
    )
    common(
        subparsers.add_parser("update", help="恰好结算一期并生成下一快照"),
        include_report=False,
    )
    common(
        subparsers.add_parser("status", help="验签全链并汇总A/B/C状态"),
        include_report=False,
    )
    return parser


def _key_path(cli_value: str | None, state_dir: str) -> Path:
    value = cli_value or os.environ.get(KEY_ENV)
    if not value:
        raise ValueError(f"必须显式提供--hmac-key-file或环境变量{KEY_ENV}")
    path = Path(value).expanduser().resolve()
    state = Path(state_dir).expanduser().resolve()
    if path == state or state in path.parents:
        raise ValueError("HMAC密钥必须位于固定state目录外")
    return path


def _load_key(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"HMAC密钥文件不存在：{path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("HMAC密钥文件权限必须为0600或更严格")
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("HMAC密钥至少32字节")
    return key


def main(argv: list[str] | None = None) -> int:
    """执行固定命令；不生成密钥，也不输出密钥内容。"""

    args = _parser().parse_args(argv)
    try:
        key = _load_key(_key_path(args.hmac_key_file, args.state_dir))
        if args.command == "register":
            result = register_prospective(
                args.csv,
                args.ensemble_report,
                args.state_dir,
                hmac_key=key,
            )
        elif args.command == "snapshot":
            result = create_snapshot(
                args.csv,
                args.ensemble_report,
                args.state_dir,
                hmac_key=key,
            )
        elif args.command == "update":
            result = update_prospective(
                args.csv,
                args.state_dir,
                hmac_key=key,
            )
        else:
            result = prospective_status(
                args.state_dir,
                canonical_csv=args.csv,
                hmac_key=key,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"错误：{error}\n建议：检查登记边界、唯一目标期、外部0600密钥与链完整性。",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
