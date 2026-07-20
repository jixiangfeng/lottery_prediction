# -*- coding: utf-8 -*-
"""三层设计公共版本与源码指纹。"""

from __future__ import annotations

import hashlib
from pathlib import Path

THREE_LAYER_SCHEMA_VERSION = 1
_THREE_LAYER_SOURCE_FILES = (
    "digit_baselines.py",
    "digit_evaluation.py",
    "digit_raw_evidence.py",
    "digit_strategy_gate.py",
    "digit_strategy_registry.py",
    "digit_full_history_shadow.py",
    "digit_learned_features.py",
    "digit_learned_ranker.py",
    "digit_learned_ranker_adaptive.py",
    "digit_learned_ranker_search.py",
    "digit_learned_ranker_walk_forward.py",
    "digit_online_gradient.py",
    "digit_predictability_audit.py",
    "digit_sparse_frozen.py",
)


def three_layer_source_fingerprint() -> str:
    """计算三层协议关键实现的稳定源码指纹。"""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _THREE_LAYER_SOURCE_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["THREE_LAYER_SCHEMA_VERSION", "three_layer_source_fingerprint"]
