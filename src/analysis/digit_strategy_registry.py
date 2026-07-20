# -*- coding: utf-8 -*-
"""实战策略注册表和不可丢失的状态迁移历史。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.analysis.digit_strategy_gate import StrategyStatus


@dataclass(frozen=True)
class StrategyRegistryUpdate:
    strategy_id: str
    lottery: str
    output_kind: str
    requested_status: StrategyStatus
    reasons: tuple[str, ...]
    data_fingerprint: str
    params_fingerprint: str
    source_fingerprint: str
    occurred_at: str
    rollback_to: str | None = None

    def __post_init__(self) -> None:
        if self.lottery not in {"fc3d", "pl3"}:
            raise ValueError("策略注册表只支持fc3d/pl3")
        if self.output_kind not in {"direct", "group", "position"}:
            raise ValueError("output_kind必须是direct/group/position")
        if not self.strategy_id or not self.occurred_at:
            raise ValueError("strategy_id和occurred_at不能为空")


def _effective_status(
    previous: StrategyStatus | None, requested: StrategyStatus
) -> StrategyStatus:
    if previous is StrategyStatus.ACTIVE and requested is not StrategyStatus.ACTIVE:
        return StrategyStatus.DEMOTED
    if previous is StrategyStatus.DEMOTED and requested not in {
        StrategyStatus.ACTIVE,
        StrategyStatus.OBSERVATION,
    }:
        return StrategyStatus.RETIRED
    return requested


def update_strategy_registry(
    path: str | Path, update: StrategyRegistryUpdate
) -> dict[str, Any]:
    """原子更新注册表；状态变化会追加不可变迁移记录。"""

    target = Path(path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
            raise ValueError("策略注册表结构错误")
    else:
        payload = {"schemaVersion": 1, "entries": {}, "transitions": []}
    entries = dict(payload.get("entries", {}))
    transitions = list(payload.get("transitions", []))
    previous_payload = entries.get(update.strategy_id)
    previous_status = (
        StrategyStatus(str(previous_payload["status"]))
        if isinstance(previous_payload, Mapping)
        else None
    )
    status = _effective_status(previous_status, update.requested_status)
    entry = {
        "strategyId": update.strategy_id,
        "lottery": update.lottery,
        "outputKind": update.output_kind,
        "status": status.value,
        "reasons": list(update.reasons),
        "dataFingerprint": update.data_fingerprint,
        "paramsFingerprint": update.params_fingerprint,
        "sourceFingerprint": update.source_fingerprint,
        "updatedAt": update.occurred_at,
        "rollbackTo": update.rollback_to,
    }
    if previous_status is None or previous_status is not status:
        transitions.append(
            {
                "strategyId": update.strategy_id,
                "from": previous_status.value if previous_status else None,
                "to": status.value,
                "occurredAt": update.occurred_at,
                "reasons": list(update.reasons),
                "rollbackTo": update.rollback_to,
                "dataFingerprint": update.data_fingerprint,
                "paramsFingerprint": update.params_fingerprint,
                "sourceFingerprint": update.source_fingerprint,
            }
        )
    entries[update.strategy_id] = entry
    result: dict[str, Any] = {
        "schemaVersion": 1,
        "entries": entries,
        "transitions": transitions,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return result


__all__ = ["StrategyRegistryUpdate", "update_strategy_registry"]
