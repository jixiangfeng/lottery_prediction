# -*- coding: utf-8 -*-
"""稀疏learned_ranker_v4一次性Frozen协议锁与防重跑执行。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.analysis.digit_data import load_digit_csv
from src.analysis.digit_learned_ranker import learned_ranker_source_fingerprint
from src.analysis.digit_online_gradient import (
    OnlineGradientConfig,
    OnlineGradientReport,
    run_online_gradient_research,
    write_online_gradient_report,
)
from src.lotteries.base import LotteryRule

_LOCK_SCHEMA = 1


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def sparse_v4_lock_payload() -> dict[str, object]:
    config = OnlineGradientConfig(development_end=1)
    return {
        "schemaVersion": _LOCK_SCHEMA,
        "modelVersion": "learned_ranker_v4",
        "protocol": "sparse_online_gradient_frozen_once",
        "sourceFingerprint": learned_ranker_source_fingerprint(),
        "frozenPeriods": 500,
        "config": {
            "outerPeriods": 500,
            "calibrationInterval": config.calibration_interval,
            "searchLookback": config.search_lookback,
            "validationLookback": config.validation_lookback,
            "warmupHistory": config.warmup_history,
            "learningRates": list(config.learning_rates),
            "shrinkages": list(config.shrinkages),
            "temperature": config.temperature,
            "l2Penalty": config.l2_penalty,
            "gradientClip": config.gradient_clip,
            "weightLimit": config.weight_limit,
            "directTopK": config.direct_top_k,
            "featureL2Multipliers": [
                list(item) for item in config.feature_l2_multipliers
            ],
            "zeroedFeatures": list(config.zeroed_features),
        },
        "gate": {
            "meanLogLossStrictlyBelowUniform": True,
            "meanBrierStrictlyBelowUniform": True,
            "top50OneSidedPValueBelow": 0.05,
            "stableLogLossBlocksRequired": 2,
            "timeBlocks": 3,
            "jointActivationRequiresBothLotteries": True,
        },
        "selectionUsesFrozenTargets": False,
        "prequentialUpdates": "目标期结果只影响下一期及以后",
    }


def create_sparse_v4_lock(path: str | Path) -> tuple[Path, str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = sparse_v4_lock_payload()
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Frozen协议锁已存在且与当前源码/配置不一致") from None
    return destination, payload_sha256(payload)


def load_and_verify_sparse_v4_lock(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = sparse_v4_lock_payload()
    if payload != expected:
        raise RuntimeError("Frozen协议锁与当前源码/配置不一致，禁止评估")
    return payload, payload_sha256(payload)


def claim_frozen_run(path: str | Path, lottery: str, lock_sha256: str) -> Path:
    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "started",
        "lottery": lottery,
        "lockSha256": lock_sha256,
        "rerunAllowed": False,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(marker, flags, 0o444)
    except FileExistsError:
        raise RuntimeError(f"{lottery} Frozen已启动或完成，禁止重复运行") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return marker


def _config_from_lock(
    payload: Mapping[str, Any], history_size: int
) -> OnlineGradientConfig:
    raw = payload["config"]
    if not isinstance(raw, dict):
        raise RuntimeError("Frozen锁config格式无效")
    return OnlineGradientConfig(
        development_end=history_size,
        outer_periods=int(raw["outerPeriods"]),
        calibration_interval=int(raw["calibrationInterval"]),
        search_lookback=int(raw["searchLookback"]),
        validation_lookback=int(raw["validationLookback"]),
        warmup_history=int(raw["warmupHistory"]),
        learning_rates=tuple(float(value) for value in raw["learningRates"]),
        shrinkages=tuple(float(value) for value in raw["shrinkages"]),
        temperature=float(raw["temperature"]),
        l2_penalty=float(raw["l2Penalty"]),
        gradient_clip=float(raw["gradientClip"]),
        weight_limit=float(raw["weightLimit"]),
        direct_top_k=int(raw["directTopK"]),
        feature_l2_multipliers=tuple(
            (str(name), float(value)) for name, value in raw["featureL2Multipliers"]
        ),
        zeroed_features=tuple(str(name) for name in raw["zeroedFeatures"]),
    )


def run_locked_sparse_v4_frozen(
    csv_path: str | Path,
    rule: LotteryRule,
    *,
    lock_path: str | Path,
    marker_path: str | Path,
    output_path: str | Path,
) -> OnlineGradientReport:
    payload, lock_sha256 = load_and_verify_sparse_v4_lock(lock_path)
    claim_frozen_run(marker_path, rule.code, lock_sha256)
    history = load_digit_csv(csv_path, rule)
    report = run_online_gradient_research(
        history,
        rule,
        _config_from_lock(payload, len(history)),
        frozen_test_read=True,
    )
    destination = write_online_gradient_report(report, output_path)
    report_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    completed = {
        "status": "completed",
        "lottery": rule.code,
        "lockSha256": lock_sha256,
        "report": str(destination),
        "reportSha256": report_sha256,
        "rerunAllowed": False,
    }
    Path(marker_path).chmod(0o644)
    Path(marker_path).write_text(
        json.dumps(completed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(marker_path).chmod(0o444)
    return report


__all__ = [
    "claim_frozen_run",
    "create_sparse_v4_lock",
    "load_and_verify_sparse_v4_lock",
    "payload_sha256",
    "run_locked_sparse_v4_frozen",
    "sparse_v4_lock_payload",
]
