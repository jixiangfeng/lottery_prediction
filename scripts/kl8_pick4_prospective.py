#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快乐8 Pick4 正式前瞻证据链 v2 命令行入口。"""

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

from src.analysis.kl8_pick4_prospective import (  # noqa: E402
    prospective_status,
    register_prospective,
    update_prospective,
)

DEFAULT_STATE = "state/kl8_pick4_prospective_v2"
DEFAULT_KEY = "~/.hermes/secrets/kl8_pick4_prospective_v2.key"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="快乐8 Pick4 正式前瞻证据链 v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser, *, include_csv: bool) -> None:
        if include_csv:
            subparser.add_argument("--csv", default="data/kl8/kl8.csv")
        subparser.add_argument("--state-dir", default=DEFAULT_STATE)
        subparser.add_argument("--hmac-key-file", default=DEFAULT_KEY)

    register = subparsers.add_parser("register", help="事务登记正式v2第0版")
    add_common(register, include_csv=True)
    register.add_argument("--raw-jsonl", default="data/kl8/raw/history.jsonl")
    register.add_argument(
        "--review-report",
        default="reports/development/kl8_pick4_rank_challenger_v2_official_20260723.json",
    )
    register.add_argument("--consume-historical-frozen", action="store_true")
    register.add_argument("--generate-hmac-key", action="store_true")

    update = subparsers.add_parser("update", help="消费精确目标期并事务发布下一版")
    add_common(update, include_csv=True)
    update.add_argument("--anchor-receipt-file", required=True)

    status = subparsers.add_parser("status", help="遍历验证完整链并输出准入状态")
    add_common(status, include_csv=True)
    status.add_argument("--report-dir", default="reports/prospective/kl8_pick4_v2")
    status.add_argument("--no-write-report", action="store_true")
    return parser


def _key_path(value: str, state_dir: str) -> Path:
    path = Path(value).expanduser().resolve()
    state = Path(state_dir).expanduser().resolve()
    if path == state or state in path.parents:
        raise ValueError("HMAC密钥必须位于state目录外")
    return path


def _load_key(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"HMAC密钥文件不存在：{path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError("HMAC密钥文件权限必须为0600或更严格")
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("HMAC密钥至少32字节")
    return key


def _prepare_register_key(path: Path, generate: bool) -> bytes:
    if generate:
        if path.exists():
            raise FileExistsError("--generate-hmac-key只允许目标路径不存在时使用")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, os.urandom(32))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return _load_key(path)


def main() -> int:
    """执行正式子命令；输出中永不包含密钥。"""

    args = _parser().parse_args()
    try:
        key_path = _key_path(args.hmac_key_file, args.state_dir)
        key = (
            _prepare_register_key(key_path, args.generate_hmac_key)
            if args.command == "register"
            else _load_key(key_path)
        )
        if args.command == "register":
            result = register_prospective(
                args.csv,
                args.state_dir,
                hmac_key=key,
                raw_jsonl=args.raw_jsonl,
                review_report=args.review_report,
                consume_historical_frozen=args.consume_historical_frozen,
            )
        elif args.command == "update":
            result = update_prospective(
                args.csv,
                args.state_dir,
                hmac_key=key,
                anchor_receipt_file=args.anchor_receipt_file,
            )
        else:
            result = prospective_status(
                args.state_dir,
                hmac_key=key,
                canonical_csv=args.csv,
                report_dir=None if args.no_write_report else args.report_dir,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        RuntimeError,
    ) as error:
        print(
            f"错误：{error}\n建议：检查v2 state、canonical CSV、外部回执与密钥权限。",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
