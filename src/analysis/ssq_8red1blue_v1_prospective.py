# -*- coding: utf-8 -*-
"""双色球8红+1蓝研究影子v1的独立仅未来HMAC链。"""

from __future__ import annotations

import hmac
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from src.analysis.ssq_d8_chain_support import (
    _atomic_replace_artifact,
    _draw_deadline,
    _fsync_directory,
    _next_target,
    _now,
    _write_artifact,
    build_bound_ensemble_report,
    canonical_data_sha256,
    load_and_verify_artifact,
    payload_sha256,
)
from src.analysis.ssq_history import SSQDraw, load_official_history_csv
from src.analysis.ssq_small_compound_8red1blue_v1 import (
    protocol_sha256 as builder_protocol_sha256,
)

SCHEMA_VERSION = "ssq_8red1blue_v1_prospective"
HORIZON = 500
DEFAULT_CANONICAL_CSV = Path("data/ssq/official_history.csv")
DEFAULT_ENSEMBLE_REPORT = Path("reports/research/ssq_ensemble_v1.json")
DEFAULT_STATE_DIR = Path("state/ssq_8red1blue_v1")
FORMAL_RECOMMENDATION_STATUS = "uniform_abstain"

FROZEN_CONFIG: dict[str, object] = {
    "horizon": HORIZON,
    "settlement": "exactly_one_matching_issue_and_date_no_catchup_or_replay",
    "training": "strictly_through_last_observed_result",
    "portfolio": "fixed_D8_28_tickets_zero_overlap_with_existing_B35",
    "combinedNominalAndUnique": 63,
    "promotion": "never_auto_promote",
}

PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "independent_future_only_research_shadow_chain",
    "stateLayout": {
        "versions": "append_only_versions/NNNN/{snapshot,observation,status}.json",
        "manifest": "immutable_registered_boundary",
        "current": "atomic_signed_pointer",
    },
    "integrity": {
        "artifact": "canonical_json_sha256_and_hmac_sha256",
        "chain": "previous_status_artifact_sha256",
        "key": "separate_explicit_external_key_only",
    },
    "frozenConfig": FROZEN_CONFIG,
    "claims": {
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    },
}


def protocol_sha256() -> str:
    """返回独立前瞻链固定协议摘要。"""

    return payload_sha256(PROTOCOL)


def _load_verified_report(
    report_path: str | Path, draws: Sequence[SSQDraw]
) -> dict[str, object]:
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("无法读取当前ssq_ensemble报告") from error
    if not isinstance(report, dict):
        raise ValueError("ssq_ensemble报告必须为JSON对象")
    claimed = report.get("reportSha256")
    unsigned = {key: value for key, value in report.items() if key != "reportSha256"}
    if not isinstance(claimed, str) or not hmac.compare_digest(
        claimed, payload_sha256(unsigned)
    ):
        raise ValueError("ssq_ensemble报告自身SHA校验失败")
    expected = build_bound_ensemble_report(draws)
    if report != expected:
        raise ValueError("ssq_ensemble报告不是当前canonical历史的精确固定重算结果")
    if (
        report.get("recommendationEnabled") is not False
        or report.get("formalCandidates") != []
        or report.get("decision") != FORMAL_RECOMMENDATION_STATUS
    ):
        raise ValueError("ssq_ensemble报告不满足统一放弃研究边界")
    return cast(dict[str, object], report)


def _portfolio_payload(report: Mapping[str, object]) -> dict[str, object]:
    d8 = report.get("smallCompound8Red1BlueV1")
    b = report.get("diversifiedPortfolioV2")
    if not isinstance(d8, Mapping) or not isinstance(b, Mapping):
        raise ValueError("ensemble报告缺少D8或B组合")
    audit = d8.get("audit")
    tickets = d8.get("expandedTickets")
    if not isinstance(audit, Mapping) or not isinstance(tickets, list):
        raise ValueError("D8报告结构非法")
    if (
        len(tickets) != 28
        or audit.get("overlapWithB") != 0
        or audit.get("combinedNominalTicketCount") != 63
        or audit.get("combinedUniqueTicketCount") != 63
    ):
        raise ValueError("D8未满足28注、B零重叠与合并63注审计")
    return {
        "red": d8["red"],
        "blue": d8["blue"],
        "tickets": tickets,
        "fixedCostMultiplier": 28,
        "selectedCandidateRank": d8["selectedCandidateRank"],
        "audit": {
            "overlapWithB": 0,
            "combinedNominalTicketCount": 63,
            "combinedUniqueTicketCount": 63,
            "BExpandedUniqueTicketCount": cast(Mapping[str, object], b["audit"])[
                "expandedUniqueTicketCount"
            ],
        },
        "portfolioSha256": payload_sha256(d8),
        "BPortfolioSha256": payload_sha256(b),
    }


def _snapshot_payload(
    *,
    version: int,
    draws: Sequence[SSQDraw],
    report: Mapping[str, object],
    manifest_sha256: str,
    previous_status_sha256: str | None,
    created_at: datetime,
) -> dict[str, object]:
    target_issue, target_date = _next_target(draws)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "createdAt": created_at.isoformat(timespec="seconds"),
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "canonicalDataSha256": canonical_data_sha256(draws),
        "ensembleReportSha256": report["reportSha256"],
        "targetIssue": target_issue,
        "targetDate": target_date,
        "D8": _portfolio_payload(report),
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }


def _completed_snapshot_payload(
    *,
    version: int,
    draws: Sequence[SSQDraw],
    manifest_sha256: str,
    previous_status_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "createdAt": created_at.isoformat(timespec="seconds"),
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "canonicalDataSha256": canonical_data_sha256(draws),
        "targetIssue": None,
        "targetDate": None,
        "D8": None,
        "completed": True,
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }


def _score_snapshot(snapshot: Mapping[str, object], draw: SSQDraw) -> dict[str, object]:
    d8 = snapshot.get("D8")
    if not isinstance(d8, Mapping):
        raise ValueError("活动快照缺少D8")
    red8_raw = d8.get("red")
    blue = d8.get("blue")
    if not isinstance(red8_raw, list) or not isinstance(blue, int):
        raise ValueError("活动快照D8红蓝字段非法")
    red_overlap = len(set(cast(list[int], red8_raw)).intersection(draw.red))
    return {
        "issue": draw.issue,
        "date": draw.draw_date,
        "red": list(draw.red),
        "blue": draw.blue,
        "red8Overlap": red_overlap,
        "atLeast3": red_overlap >= 3,
        "atLeast4": red_overlap >= 4,
        "atLeast5": red_overlap >= 5,
        "atLeast6": red_overlap >= 6,
        "blueHit": blue == draw.blue,
        "exact6PlusBlue": red_overlap == 6 and blue == draw.blue,
        "exact6PlusNoBlue": red_overlap == 6 and blue != draw.blue,
        "overlapWithB": cast(Mapping[str, object], d8["audit"])["overlapWithB"],
        "combinedNominalTicketCount": 63,
        "combinedUniqueTicketCount": 63,
        "officialPrizeClaims": False,
    }


def _observation_payload(
    version: int,
    settled_snapshot: Mapping[str, object] | None,
    draw: SSQDraw | None,
) -> dict[str, object]:
    if version == 0:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "version": 0,
            "settledSnapshotVersion": None,
            "result": None,
            "genesisWithoutResult": True,
        }
    if settled_snapshot is None or draw is None:
        raise ValueError("非创世版本必须绑定结算快照与恰好一期结果")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "settledSnapshotVersion": version - 1,
        "settledSnapshotArtifactSha256": settled_snapshot["artifactSha256"],
        "targetIssue": settled_snapshot["targetIssue"],
        "targetDate": settled_snapshot["targetDate"],
        "result": _score_snapshot(settled_snapshot, draw),
    }


def _status_payload(
    *,
    version: int,
    snapshot: Mapping[str, object],
    observation: Mapping[str, object],
    manifest_sha256: str,
    previous_status_sha256: str | None,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "completed": version,
        "horizon": HORIZON,
        "remaining": HORIZON - version,
        "targetIssue": snapshot.get("targetIssue"),
        "targetDate": snapshot.get("targetDate"),
        "snapshotArtifactSha256": snapshot["artifactSha256"],
        "observationArtifactSha256": observation["artifactSha256"],
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }


def _publish_version(
    root: Path,
    *,
    version: int,
    snapshot_payload: Mapping[str, object],
    observation_payload: Mapping[str, object],
    manifest_sha256: str,
    previous_status_sha256: str | None,
    key: bytes,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    versions = root / "versions"
    final_dir = versions / f"{version:04d}"
    staging = versions / f".{version:04d}.tmp"
    if final_dir.exists() or staging.exists():
        raise FileExistsError("目标版本已存在或残留临时目录")
    staging.mkdir(mode=0o700)
    try:
        snapshot = _write_artifact(staging / "snapshot.json", snapshot_payload, key)
        observation = _write_artifact(
            staging / "observation.json", observation_payload, key
        )
        status = _write_artifact(
            staging / "status.json",
            _status_payload(
                version=version,
                snapshot=snapshot,
                observation=observation,
                manifest_sha256=manifest_sha256,
                previous_status_sha256=previous_status_sha256,
            ),
            key,
        )
        _fsync_directory(staging)
        os.replace(staging, final_dir)
        _fsync_directory(versions)
        try:
            _atomic_replace_artifact(
                root / "current.json",
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "version": version,
                    "statusArtifactSha256": status["artifactSha256"],
                },
                key,
            )
        except BaseException:
            shutil.rmtree(final_dir)
            _fsync_directory(versions)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return snapshot, observation, status


def register_prospective(
    canonical_csv: str | Path,
    ensemble_report: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """登记唯一未来边界；不生成快照或真实状态样例。"""

    root = Path(state_dir)
    if root.exists():
        raise FileExistsError("D8前瞻状态目录已存在，register只允许一次")
    draws = load_official_history_csv(canonical_csv)
    report = _load_verified_report(ensemble_report, draws)
    target_issue, target_date = _next_target(draws)
    created_at = _now(now)
    if created_at >= _draw_deadline(target_date):
        raise ValueError("登记时间已越过目标期开奖截止")
    staging = root.with_name(f".{root.name}.register.tmp")
    if staging.exists():
        raise FileExistsError("存在D8登记临时目录，需人工审计")
    staging.mkdir(parents=True, mode=0o700)
    try:
        protocol = _write_artifact(
            staging / "protocol.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "prospectiveProtocolSha256": protocol_sha256(),
                "builderProtocolSha256": builder_protocol_sha256(),
                "ensembleProtocolSha256": report["protocolSha256"],
            },
            hmac_key,
        )
        _write_artifact(
            staging / "manifest.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "registeredAt": created_at.isoformat(timespec="seconds"),
                "registeredLatestIssue": draws[-1].issue,
                "registeredLatestDate": draws[-1].draw_date,
                "initialHistoryRows": len(draws),
                "initialTargetIssue": target_issue,
                "initialTargetDate": target_date,
                "initialD8PortfolioSha256": _portfolio_payload(report)[
                    "portfolioSha256"
                ],
                "initialD8Audit": _portfolio_payload(report)["audit"],
                "horizon": HORIZON,
                "frozenConfig": FROZEN_CONFIG,
                "canonicalDataSha256": canonical_data_sha256(draws),
                "verifiedReportSha256": report["reportSha256"],
                "protocolArtifactSha256": protocol["artifactSha256"],
                "researchOnly": True,
                "predictionClaim": False,
                "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
                "autoPromotion": False,
            },
            hmac_key,
        )
        (staging / "versions").mkdir(mode=0o700)
        _atomic_replace_artifact(
            staging / "current.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "version": None,
                "statusArtifactSha256": None,
            },
            hmac_key,
        )
        _fsync_directory(staging)
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, root)
        _fsync_directory(root.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "action": "registered",
        "stateChanged": True,
        "targetIssue": target_issue,
        "targetDate": target_date,
        "D8": _portfolio_payload(report),
        "snapshotCreated": False,
        "researchOnly": True,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
    }


def _verify_registration(
    root: Path, key: bytes
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = load_and_verify_artifact(root / "protocol.json", key)
    manifest = load_and_verify_artifact(root / "manifest.json", key)
    if protocol.get("prospectiveProtocolSha256") != protocol_sha256():
        raise ValueError("D8前瞻协议摘要不一致")
    if protocol.get("builderProtocolSha256") != builder_protocol_sha256():
        raise ValueError("D8构建器协议摘要不一致")
    if manifest.get("protocolArtifactSha256") != protocol.get("artifactSha256"):
        raise ValueError("D8 manifest未绑定protocol")
    if (
        manifest.get("frozenConfig") != FROZEN_CONFIG
        or manifest.get("horizon") != HORIZON
    ):
        raise ValueError("D8 manifest冻结配置不一致")
    return protocol, manifest


def create_snapshot(
    canonical_csv: str | Path,
    ensemble_report: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """生成唯一0000预开奖快照。"""

    root = Path(state_dir)
    _, manifest = _verify_registration(root, hmac_key)
    if any((root / "versions").iterdir()):
        raise FileExistsError("D8首快照已存在")
    draws = load_official_history_csv(canonical_csv)
    if (
        len(draws) != manifest["initialHistoryRows"]
        or canonical_data_sha256(draws) != manifest["canonicalDataSha256"]
    ):
        raise ValueError("snapshot要求canonical仍停留在登记边界")
    report = _load_verified_report(ensemble_report, draws)
    if report["reportSha256"] != manifest["verifiedReportSha256"]:
        raise ValueError("snapshot报告与登记报告不一致")
    created_at = _now(now)
    if created_at >= _draw_deadline(cast(str, manifest["initialTargetDate"])):
        raise ValueError("snapshot时间已越过首目标期开奖截止")
    snapshot_payload = _snapshot_payload(
        version=0,
        draws=draws,
        report=report,
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=None,
        created_at=created_at,
    )
    observation_payload = _observation_payload(0, None, None)
    snapshot, _, _ = _publish_version(
        root,
        version=0,
        snapshot_payload=snapshot_payload,
        observation_payload=observation_payload,
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=None,
        key=hmac_key,
    )
    return {
        "action": "snapshot_created",
        "stateChanged": True,
        "version": 0,
        "completed": 0,
        "horizon": HORIZON,
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "D8": snapshot["D8"],
    }


def _versions(root: Path) -> list[int]:
    values: list[int] = []
    for path in (root / "versions").iterdir():
        if not path.is_dir() or not path.name.isdigit() or len(path.name) != 4:
            raise ValueError("D8版本目录存在非冻结条目")
        values.append(int(path.name))
    values.sort()
    if values != list(range(len(values))):
        raise ValueError("D8版本号不连续")
    return values


def _verify_chain(
    root: Path, key: bytes
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    _, manifest = _verify_registration(root, key)
    versions = _versions(root)
    if not versions:
        raise ValueError("D8尚未创建首快照")
    snapshots: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    previous_status: str | None = None
    latest_status: dict[str, object] | None = None
    for version in versions:
        directory = root / "versions" / f"{version:04d}"
        if {path.name for path in directory.iterdir()} != {
            "snapshot.json",
            "observation.json",
            "status.json",
        }:
            raise ValueError("D8版本文件集合不符合冻结布局")
        snapshot = load_and_verify_artifact(directory / "snapshot.json", key)
        observation = load_and_verify_artifact(directory / "observation.json", key)
        status = load_and_verify_artifact(directory / "status.json", key)
        if (
            snapshot.get("version") != version
            or observation.get("version") != version
            or status.get("version") != version
        ):
            raise ValueError("D8版本内部编号不一致")
        if (
            snapshot.get("previousStatusArtifactSha256") != previous_status
            or status.get("previousStatusArtifactSha256") != previous_status
        ):
            raise ValueError("D8前瞻链断裂")
        if status.get("snapshotArtifactSha256") != snapshot.get(
            "artifactSha256"
        ) or status.get("observationArtifactSha256") != observation.get(
            "artifactSha256"
        ):
            raise ValueError("D8状态未绑定快照或观测")
        if version > 0 and observation.get(
            "settledSnapshotArtifactSha256"
        ) != snapshots[-1].get("artifactSha256"):
            raise ValueError("D8观测未绑定上一预开奖快照")
        snapshots.append(snapshot)
        observations.append(observation)
        latest_status = status
        previous_status = cast(str, status["artifactSha256"])
    current = load_and_verify_artifact(root / "current.json", key)
    if (
        latest_status is None
        or current.get("version") != versions[-1]
        or current.get("statusArtifactSha256") != latest_status.get("artifactSha256")
    ):
        raise ValueError("D8 current指针与链尾不一致")
    return manifest, snapshots, observations


def update_prospective(
    canonical_csv: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """恰好结算一个目标期；禁止追赶、重放与自动晋级。"""

    root = Path(state_dir)
    manifest, snapshots, _ = _verify_chain(root, hmac_key)
    current_snapshot = snapshots[-1]
    completed = len(snapshots) - 1
    if completed >= HORIZON:
        raise ValueError("D8前瞻链已完成500期")
    draws = load_official_history_csv(canonical_csv)
    expected_rows = cast(int, manifest["initialHistoryRows"]) + completed + 1
    if len(draws) != expected_rows:
        raise ValueError(
            "update要求canonical相对当前链尾恰好新增一期，禁止catchup/replay"
        )
    draw = draws[-1]
    if draw.issue != current_snapshot.get(
        "targetIssue"
    ) or draw.draw_date != current_snapshot.get("targetDate"):
        raise ValueError("新增开奖与锁定targetIssue/targetDate不一致")
    created_at = _now(now)
    if created_at < _draw_deadline(draw.draw_date):
        raise ValueError("目标期开奖截止前禁止结算")
    previous_status = load_and_verify_artifact(
        root / "versions" / f"{completed:04d}" / "status.json", hmac_key
    )
    version = completed + 1
    if version == HORIZON:
        snapshot_payload = _completed_snapshot_payload(
            version=version,
            draws=draws,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=cast(str, previous_status["artifactSha256"]),
            created_at=created_at,
        )
    else:
        report = build_bound_ensemble_report(draws)
        snapshot_payload = _snapshot_payload(
            version=version,
            draws=draws,
            report=report,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=cast(str, previous_status["artifactSha256"]),
            created_at=created_at,
        )
    observation_payload = _observation_payload(version, current_snapshot, draw)
    snapshot, observation, _ = _publish_version(
        root,
        version=version,
        snapshot_payload=snapshot_payload,
        observation_payload=observation_payload,
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=cast(str, previous_status["artifactSha256"]),
        key=hmac_key,
    )
    return {
        "action": "updated",
        "stateChanged": True,
        "version": version,
        "completed": version,
        "horizon": HORIZON,
        "settledIssue": draw.issue,
        "result": observation["result"],
        "targetIssue": snapshot.get("targetIssue"),
        "targetDate": snapshot.get("targetDate"),
        "D8": snapshot.get("D8"),
        "autoPromotion": False,
    }


def prospective_status(
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    canonical_csv: str | Path,
    hmac_key: bytes,
) -> dict[str, object]:
    """验签全链并返回只读状态；不补期、不生成快照。"""

    root = Path(state_dir)
    manifest, snapshots, observations = _verify_chain(root, hmac_key)
    completed = len(snapshots) - 1
    draws = load_official_history_csv(canonical_csv)
    expected_rows = cast(int, manifest["initialHistoryRows"]) + completed
    pending_exact_one = len(draws) == expected_rows + 1
    if len(draws) not in {expected_rows, expected_rows + 1}:
        raise ValueError("canonical与D8链边界不相容；禁止隐式追赶")
    results = [
        cast(Mapping[str, object], observation["result"])
        for observation in observations[1:]
    ]
    distribution = {str(index): 0 for index in range(7)}
    for result in results:
        distribution[str(result["red8Overlap"])] += 1
    latest = snapshots[-1]
    return {
        "action": "status",
        "stateChanged": False,
        "completed": completed,
        "horizon": HORIZON,
        "remaining": HORIZON - completed,
        "targetIssue": latest.get("targetIssue"),
        "targetDate": latest.get("targetDate"),
        "D8": latest.get("D8"),
        "pendingExactOneIssueUpdate": pending_exact_one,
        "red8OverlapDistribution": distribution,
        "blueHits": sum(bool(result["blueHit"]) for result in results),
        "exact6PlusBlue": sum(bool(result["exact6PlusBlue"]) for result in results),
        "exact6PlusNoBlue": sum(bool(result["exact6PlusNoBlue"]) for result in results),
        "researchOnly": True,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }


__all__ = [
    "DEFAULT_CANONICAL_CSV",
    "DEFAULT_ENSEMBLE_REPORT",
    "DEFAULT_STATE_DIR",
    "HORIZON",
    "create_snapshot",
    "prospective_status",
    "protocol_sha256",
    "register_prospective",
    "update_prospective",
]
