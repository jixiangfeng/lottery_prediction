# -*- coding: utf-8 -*-
"""仓库 Python 源码语法闸门。"""

from __future__ import annotations

from pathlib import Path


def test_src_and_scripts_compile_without_syntax_errors():
    root = Path(__file__).resolve().parents[1]

    python_files = sorted(
        path
        for directory in (root / "src", root / "scripts")
        for path in directory.rglob("*.py")
    )

    assert python_files
    for path in python_files:
        compile(path.read_bytes(), str(path), "exec")
