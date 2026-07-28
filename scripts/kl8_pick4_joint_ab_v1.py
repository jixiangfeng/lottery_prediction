#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快乐8 Pick4 同成本联合概率 A/B v1 命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.kl8_pick4_joint_ab_v1 import (  # noqa: E402
    initialize_joint_ab,
    joint_ab_status,
    step_joint_ab,
)

DEFAULT_CSV = "data/kl8/kl8.csv"
DEFAULT_STATE = "state/kl8_pick4_joint_ab_v1"
DEFAULT_KEY = "~/.hermes/secrets/kl8_pick4_joint_ab_v1.key"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="快乐8 Pick4 同成本联合概率 A/B v1（仅研究）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_fixed_paths(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--csv", default=DEFAULT_CSV)
        subparser.add_argument("--state-dir", default=DEFAULT_STATE)
        subparser.add_argument("--hmac-key-file", default=DEFAULT_KEY)

    initialize = subparsers.add_parser("initialize", help="开奖前创建唯一第0版快照")
    add_fixed_paths(initialize)
    initialize.add_argument("--generate-hmac-key", action="store_true")
    add_fixed_paths(
        subparsers.add_parser("step", help="最多结算一个锁定目标并生成下一快照")
    )
    add_fixed_paths(
        subparsers.add_parser("status", help="验证全链并汇总固定500期A/B状态")
    )
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
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("HMAC密钥文件权限必须为0600或更严格")
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("HMAC密钥至少32字节")
    return key


def _prepare_key(path: Path, generate: bool) -> bytes:
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


def _progress_writer(
    state_dir: str, command: str
) -> Callable[[dict[str, object]], None]:
    progress_path = Path(f"{state_dir}.progress.json").expanduser().resolve()

    def write(payload: dict[str, object]) -> None:
        document = {
            "schemaVersion": "kl8_pick4_joint_ab_v1_progress",
            "command": command,
            **payload,
        }
        content = (
            json.dumps(
                document, ensure_ascii=False, sort_keys=True, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_path.with_name(f".{progress_path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, progress_path)
        directory = os.open(progress_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        print(
            json.dumps(document, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )

    return write


def main(argv: list[str] | None = None) -> int:
    """执行固定 initialize/step/status 操作，不暴露模型或目标覆盖参数。"""

    args = _parser().parse_args(argv)
    try:
        progress = _progress_writer(args.state_dir, args.command)
        progress({"event": "commandStarted"})
        key_path = _key_path(args.hmac_key_file, args.state_dir)
        key = (
            _prepare_key(key_path, args.generate_hmac_key)
            if args.command == "initialize"
            else _load_key(key_path)
        )
        if args.command == "initialize":
            result = initialize_joint_ab(
                args.csv,
                args.state_dir,
                hmac_key=key,
                progress_callback=progress,
            )
        elif args.command == "step":
            result = step_joint_ab(
                args.csv,
                args.state_dir,
                hmac_key=key,
                progress_callback=progress,
            )
        else:
            result = joint_ab_status(args.csv, args.state_dir, hmac_key=key)
        progress(
            {
                "event": "commandCompleted",
                "status": result["status"] if "status" in result else "reported",
            }
        )
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        RuntimeError,
    ) as error:
        print(
            f"错误：{error}\n建议：检查canonical CSV、追加式state和0600本地密钥；禁止补跑、重置或调参。",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
