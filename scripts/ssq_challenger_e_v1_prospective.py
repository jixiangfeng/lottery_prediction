#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球 Challenger E1 独立仅未来 HMAC 链命令行。"""

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

from src.analysis.ssq_challenger_e_v1_prospective import (  # noqa: E402
    DEFAULT_CANONICAL_CSV,
    DEFAULT_D8_STATE_DIR,
    DEFAULT_E_REPORT,
    DEFAULT_ENSEMBLE_REPORT,
    DEFAULT_STATE_DIR,
    create_snapshot,
    prospective_status,
    register_prospective,
    update_prospective,
)

KEY_ENV = "SSQ_CHALLENGER_E_V1_HMAC_KEY_FILE"
D8_KEY_ENV = "SSQ_8RED1BLUE_V1_HMAC_KEY_FILE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, reports: bool, d8: bool) -> None:
        command.add_argument("--csv", default=str(DEFAULT_CANONICAL_CSV))
        command.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
        command.add_argument("--hmac-key-file", help=f"独立E1密钥；也可设置{KEY_ENV}")
        if reports:
            command.add_argument("--e-report", default=str(DEFAULT_E_REPORT))
            command.add_argument(
                "--ensemble-report", default=str(DEFAULT_ENSEMBLE_REPORT)
            )
        if d8:
            command.add_argument("--d8-state-dir", default=str(DEFAULT_D8_STATE_DIR))
            command.add_argument(
                "--d8-hmac-key-file", help=f"只读D8链密钥；也可设置{D8_KEY_ENV}"
            )

    common(
        commands.add_parser("register", help="登记唯一未来边界"), reports=True, d8=True
    )
    common(
        commands.add_parser("snapshot", help="创建唯一0000快照"), reports=True, d8=True
    )
    common(
        commands.add_parser("update", help="恰好结算一期并推进"), reports=False, d8=True
    )
    common(
        commands.add_parser("status", help="验签全链并汇总"), reports=False, d8=False
    )
    return parser


def _key_path(value: str | None, env_name: str, forbidden_state: str) -> Path:
    raw = value or os.environ.get(env_name)
    if not raw:
        raise ValueError(f"必须显式提供密钥文件或环境变量{env_name}")
    path = Path(raw).expanduser().resolve()
    state = Path(forbidden_state).expanduser().resolve()
    if path == state or state in path.parents:
        raise ValueError("HMAC密钥必须位于对应state目录外")
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
    """执行固定协议；绝不创建、输出或保存真实密钥。"""

    args = _parser().parse_args(argv)
    try:
        key = _load_key(_key_path(args.hmac_key_file, KEY_ENV, args.state_dir))
        if args.command == "status":
            result = prospective_status(
                args.state_dir, canonical_csv=args.csv, hmac_key=key
            )
        else:
            d8_key = _load_key(
                _key_path(args.d8_hmac_key_file, D8_KEY_ENV, args.d8_state_dir)
            )
            if args.command == "register":
                result = register_prospective(
                    args.csv,
                    args.e_report,
                    args.ensemble_report,
                    args.state_dir,
                    d8_state_dir=args.d8_state_dir,
                    hmac_key=key,
                    d8_hmac_key=d8_key,
                )
            elif args.command == "snapshot":
                result = create_snapshot(
                    args.csv,
                    args.e_report,
                    args.ensemble_report,
                    args.state_dir,
                    d8_state_dir=args.d8_state_dir,
                    hmac_key=key,
                    d8_hmac_key=d8_key,
                )
            else:
                result = update_prospective(
                    args.csv,
                    args.state_dir,
                    d8_state_dir=args.d8_state_dir,
                    hmac_key=key,
                    d8_hmac_key=d8_key,
                )
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"错误：{error}\n建议：检查独立0600密钥、D8只读链、唯一目标期和完整HMAC链。",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
