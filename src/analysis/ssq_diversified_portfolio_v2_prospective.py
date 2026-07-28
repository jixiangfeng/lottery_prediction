# -*- coding: utf-8 -*-
"""双色球分散组合 v2 的仅未来 HMAC 前瞻证据链。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import combinations
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from src.analysis import ssq_diversified_portfolio_v2 as builder
from src.analysis import ssq_ensemble_v1 as ensemble
from src.analysis.ssq_history import SSQDraw, load_official_history_csv

SCHEMA_VERSION = "ssq_diversified_portfolio_v2_prospective_v1"
DEFAULT_STATE_DIR = Path("state/ssq_diversified_portfolio_v2")
DEFAULT_CANONICAL_CSV = Path("data/ssq/official_history.csv")
DEFAULT_ENSEMBLE_REPORT = Path("reports/research/ssq_ensemble_v1.json")
HORIZON = 500
PORTFOLIO_COST = 35
CONTROL_COUNT = 32
GROUP_COUNT = 5
RED_PER_GROUP = 7
CONTROL_SEED_PROTOCOL = (
    "ssq_diversified_portfolio_v2_history|matched_cost_random_c|"
    "sha256_counter_rejection_v1"
)
THRESHOLDS = (3, 4, 5, 6)
METRIC_NAMES = (
    "averageRedHitsPerTicket",
    "maximumRedHitsAnyTicket",
    *(f"anyTicketRedAtLeast{threshold}" for threshold in THRESHOLDS),
    "blueAnyHit",
    *(f"ticketRedAtLeast{threshold}PlusBlue" for threshold in THRESHOLDS),
    *(f"ticketRedAtLeast{threshold}PlusNoBlue" for threshold in THRESHOLDS),
)
FORMAL_RECOMMENDATION_STATUS = "uniform_abstain"
SHANGHAI = ZoneInfo("Asia/Shanghai")
DRAW_WEEKDAYS = (1, 3, 6)  # 周二、周四、周日。
DRAW_CUTOFF = time(hour=21, minute=30)

FROZEN_CONFIG: dict[str, object] = {
    "horizon": HORIZON,
    "schedule": "Asia/Shanghai Tuesday/Thursday/Sunday",
    "drawCutoff": "21:30:00+08:00",
    "portfolioCost": PORTFOLIO_COST,
    "A": "concentrated_5x7_shared_blue_shadow",
    "B": "diversified_5x7_distinct_blue_primary",
    "C": "32_deterministic_matched_cost_controls",
    "settlement": "exactly_one_matching_issue_and_date_no_catch_up",
    "training": "strictly_through_last_observed_result",
    "promotion": "never_auto_promote",
}

PROTOCOL: dict[str, object] = {
    "schemaVersion": SCHEMA_VERSION,
    "purpose": "future_only_research_shadow_chain",
    "stateLayout": {
        "versions": "append_only_versions/NNNN/{snapshot,observation,status}.json",
        "manifest": "immutable_registered_boundary",
        "current": "atomic_signed_pointer",
    },
    "integrity": {
        "artifact": "canonical_json_sha256_and_hmac_sha256",
        "chain": "previous_status_artifact_sha256",
        "key": "external_explicit_cli_or_environment_only",
    },
    "frozenConfig": FROZEN_CONFIG,
    "claims": {
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    },
}


@dataclass(frozen=True)
class Portfolio:
    """五组红7、逐组蓝球和35注展开票。"""

    red7_groups: tuple[tuple[int, ...], ...]
    blues: tuple[int, ...]
    tickets: tuple[tuple[tuple[int, ...], int], ...]


class _Sha256CounterRng:
    """与回溯 C 协议一致的跨运行稳定计数器随机源。"""

    def __init__(self, seed: bytes) -> None:
        self._seed = seed
        self._counter = 0

    def _uint256(self) -> int:
        digest = hashlib.sha256(self._seed + self._counter.to_bytes(16, "big")).digest()
        self._counter += 1
        return int.from_bytes(digest, "big")

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("随机上界必须为正数")
        space = 1 << 256
        limit = space - space % upper
        while True:
            value = self._uint256()
            if value < limit:
                return value % upper

    def shuffled(self, values: Sequence[int]) -> list[int]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            other = self.randbelow(index + 1)
            result[index], result[other] = result[other], result[index]
        return result


def _expand(
    red7_groups: Sequence[Sequence[int]], blues: Sequence[int]
) -> tuple[tuple[tuple[int, ...], int], ...]:
    tickets = tuple(
        (tuple(red6), blue)
        for red7, blue in zip(red7_groups, blues)
        for red6 in combinations(red7, 6)
    )
    if len(tickets) != PORTFOLIO_COST or len(set(tickets)) != PORTFOLIO_COST:
        raise ValueError("组合必须形成35注全局唯一票")
    return tickets


def _portfolio_a(
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    pair_modifiers: Sequence[float],
) -> Portfolio:
    ranked = ensemble.beam_red_combinations(red_probabilities, pair_modifiers)
    red7_groups: list[tuple[int, ...]] = []
    seen_tickets: set[tuple[tuple[int, ...], int]] = set()
    shared_blue = ensemble.blue_top1(blue_probabilities)
    for base_index, (base_red, _) in enumerate(ranked):
        base_set = set(base_red)
        extra_red: int | None = None
        for later_red, _ in ranked[base_index + 1 :]:
            absent = sorted(set(later_red) - base_set)
            if absent:
                extra_red = absent[0]
                break
        if extra_red is None:
            continue
        red7 = tuple(sorted((*base_red, extra_red)))
        if red7 in red7_groups:
            continue
        tickets = {(tuple(red6), shared_blue) for red6 in combinations(red7, 6)}
        if len(tickets) != 7:
            raise ValueError("A单组红7未展开为7注唯一票")
        if tickets.intersection(seen_tickets):
            continue
        red7_groups.append(red7)
        seen_tickets.update(tickets)
        if len(red7_groups) == GROUP_COUNT:
            break
    if len(red7_groups) != GROUP_COUNT:
        raise ValueError("A排名不足以构造5组互异且全局唯一的集中组合")
    blues = (shared_blue,) * GROUP_COUNT
    return Portfolio(tuple(red7_groups), blues, _expand(red7_groups, blues))


def _portfolio_b(
    red_probabilities: Sequence[float], blue_probabilities: Sequence[float]
) -> tuple[Portfolio, dict[str, object]]:
    document = builder.build_diversified_portfolio_v2(
        red_probabilities, blue_probabilities
    )
    builder.validate_diversified_portfolio_v2(
        document,
        red_probabilities=red_probabilities,
        blue_probabilities=blue_probabilities,
    )
    groups_raw = cast(list[dict[str, object]], document["groups"])
    red7_groups = tuple(tuple(cast(list[int], group["red"])) for group in groups_raw)
    blues = tuple(cast(int, group["blue"]) for group in groups_raw)
    return Portfolio(red7_groups, blues, _expand(red7_groups, blues)), document


def build_matched_control_c(issue: str, control_index: int) -> Portfolio:
    """按既有回溯协议，仅由期号与编号生成一组 C。"""

    if not issue.isdigit() or not 0 <= control_index < CONTROL_COUNT:
        raise ValueError("C期号或控制编号非法")
    seed = hashlib.sha256(
        f"{CONTROL_SEED_PROTOCOL}|issue={issue}|control={control_index}".encode()
    ).digest()
    rng = _Sha256CounterRng(seed)
    blues = tuple(rng.shuffled(tuple(range(1, 17)))[:GROUP_COUNT])
    red7_groups: list[tuple[int, ...]] = []
    attempts = 0
    while len(red7_groups) < GROUP_COUNT:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("C未能在固定安全次数内完成构造")
        red7 = tuple(sorted(rng.shuffled(tuple(range(1, 34)))[:RED_PER_GROUP]))
        if red7 not in red7_groups:
            red7_groups.append(red7)
    return Portfolio(tuple(red7_groups), blues, _expand(red7_groups, blues))


def _score(portfolio: Portfolio, draw: SSQDraw) -> dict[str, float]:
    actual_red = set(draw.red)
    red_hits = tuple(
        len(set(red6).intersection(actual_red)) for red6, _ in portfolio.tickets
    )
    blue_hits = tuple(blue == draw.blue for _, blue in portfolio.tickets)
    maximum = max(red_hits)
    metrics: dict[str, float] = {
        "averageRedHitsPerTicket": sum(red_hits) / PORTFOLIO_COST,
        "maximumRedHitsAnyTicket": float(maximum),
        "blueAnyHit": float(any(blue_hits)),
    }
    for threshold in THRESHOLDS:
        metrics[f"anyTicketRedAtLeast{threshold}"] = float(maximum >= threshold)
        metrics[f"ticketRedAtLeast{threshold}PlusBlue"] = float(
            any(red >= threshold and blue for red, blue in zip(red_hits, blue_hits))
        )
        metrics[f"ticketRedAtLeast{threshold}PlusNoBlue"] = float(
            any(red >= threshold and not blue for red, blue in zip(red_hits, blue_hits))
        )
    return metrics


def _issue_mean(scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(scores) != CONTROL_COUNT:
        raise ValueError("每期C必须恰好包含32组评分")
    return {
        name: sum(score[name] for score in scores) / CONTROL_COUNT
        for name in METRIC_NAMES
    }


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    """返回稳定规范 JSON 的 SHA-256。"""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def protocol_sha256() -> str:
    """返回冻结前瞻协议摘要。"""

    return payload_sha256(PROTOCOL)


def _artifact_hmac(payload: Mapping[str, object], digest: str, key: bytes) -> str:
    if len(key) < 32:
        raise ValueError("HMAC密钥至少32字节")
    message = _canonical_bytes(payload) + b"\n" + digest.encode("ascii")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _artifact_document(payload: Mapping[str, object], key: bytes) -> dict[str, object]:
    unsigned = dict(payload)
    if "artifactSha256" in unsigned or "artifactHmacSha256" in unsigned:
        raise ValueError("payload不得预置签名字段")
    digest = payload_sha256(unsigned)
    return {
        **unsigned,
        "artifactSha256": digest,
        "artifactHmacSha256": _artifact_hmac(unsigned, digest, key),
    }


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_artifact(
    path: Path, payload: Mapping[str, object], key: bytes
) -> dict[str, object]:
    document = _artifact_document(payload, key)
    _write_exclusive(path, _canonical_bytes(document) + b"\n")
    return document


def _atomic_replace_artifact(
    path: Path, payload: Mapping[str, object], key: bytes
) -> dict[str, object]:
    document = _artifact_document(payload, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(document) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def load_and_verify_artifact(path: str | Path, hmac_key: bytes) -> dict[str, object]:
    """加载 JSON artifact，并验证自哈希与外部 HMAC。"""

    artifact_path = Path(path)
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取artifact：{artifact_path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"artifact必须是JSON对象：{artifact_path}")
    digest = document.get("artifactSha256")
    signature = document.get("artifactHmacSha256")
    if not isinstance(digest, str) or not isinstance(signature, str):
        raise ValueError(f"artifact缺少SHA/HMAC：{artifact_path}")
    payload = {
        key: value
        for key, value in document.items()
        if key not in {"artifactSha256", "artifactHmacSha256"}
    }
    expected_digest = payload_sha256(payload)
    if not hmac.compare_digest(digest, expected_digest):
        raise ValueError(f"artifact SHA校验失败：{artifact_path}")
    expected_signature = _artifact_hmac(payload, digest, hmac_key)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError(f"artifact HMAC校验失败：{artifact_path}")
    return cast(dict[str, object], document)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(tz=SHANGHAI)
    if value.tzinfo is None:
        raise ValueError("now必须带时区")
    return value.astimezone(SHANGHAI)


def _draw_deadline(draw_date: str) -> datetime:
    day = date.fromisoformat(draw_date)
    return datetime.combine(day, DRAW_CUTOFF, tzinfo=SHANGHAI)


def _next_target(draws: Sequence[SSQDraw]) -> tuple[str, str]:
    if not draws:
        raise ValueError("canonical历史为空")
    latest = draws[-1]
    latest_date = date.fromisoformat(latest.draw_date)
    target_date = latest_date + timedelta(days=1)
    while target_date.weekday() not in DRAW_WEEKDAYS:
        target_date += timedelta(days=1)
    issue_year = int(latest.issue[:4])
    sequence = int(latest.issue[4:])
    target_issue = (
        f"{issue_year}{sequence + 1:03d}"
        if target_date.year == issue_year
        else f"{target_date.year}001"
    )
    return target_issue, target_date.isoformat()


def _draw_payload(draw: SSQDraw) -> dict[str, object]:
    return {
        "issue": draw.issue,
        "date": draw.draw_date,
        "red": list(draw.red),
        "blue": draw.blue,
        "sourceUrl": draw.source_url,
        "rawHash": draw.raw_hash,
    }


def canonical_data_sha256(draws: Sequence[SSQDraw]) -> str:
    """返回与现有 ssq_ensemble 报告一致的 canonical 数据摘要。"""

    return payload_sha256([_draw_payload(draw) for draw in draws])


def build_bound_ensemble_report(draws: Sequence[SSQDraw]) -> dict[str, object]:
    """重算固定 ensemble，并绑定 canonical 最新期与数据摘要。"""

    report = ensemble.evaluate_ssq_ensemble(draws)
    bound = dict(report)
    bound["latestIssue"] = draws[-1].issue
    bound["dataSha256"] = canonical_data_sha256(draws)
    bound["reportSha256"] = payload_sha256(bound)
    return bound


def _load_verified_report(
    report_path: str | Path, draws: Sequence[SSQDraw]
) -> dict[str, object]:
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("无法读取当前ssq_ensemble报告") from error
    if not isinstance(report, dict):
        raise ValueError("ssq_ensemble报告必须是JSON对象")
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
        report.get("decision") != FORMAL_RECOMMENDATION_STATUS
        or report.get("recommendationEnabled") is not False
        or report.get("formalCandidates") != []
    ):
        raise ValueError("ssq_ensemble报告不满足统一放弃研究边界")
    return cast(dict[str, object], report)


def _portfolio_payload(portfolio: Portfolio) -> dict[str, object]:
    return {
        "cost": PORTFOLIO_COST,
        "red7Groups": [list(group) for group in portfolio.red7_groups],
        "blues": list(portfolio.blues),
        "tickets": [
            {"red": list(red), "blue": blue} for red, blue in portfolio.tickets
        ],
        "portfolioSha256": payload_sha256(
            {
                "red7Groups": [list(group) for group in portfolio.red7_groups],
                "blues": list(portfolio.blues),
                "tickets": [
                    {"red": list(red), "blue": blue} for red, blue in portfolio.tickets
                ],
            }
        ),
    }


def _portfolio_from_payload(payload: object, label: str) -> Portfolio:
    if not isinstance(payload, Mapping) or payload.get("cost") != PORTFOLIO_COST:
        raise ValueError(f"{label}成本必须恰好为35")
    groups_raw = payload.get("red7Groups")
    blues_raw = payload.get("blues")
    tickets_raw = payload.get("tickets")
    if not isinstance(groups_raw, list) or not isinstance(blues_raw, list):
        raise ValueError(f"{label}组合字段类型非法")
    if not isinstance(tickets_raw, list):
        raise ValueError(f"{label}票字段类型非法")
    groups = tuple(tuple(cast(list[int], group)) for group in groups_raw)
    blues = tuple(cast(list[int], blues_raw))
    tickets: list[tuple[tuple[int, ...], int]] = []
    for ticket in tickets_raw:
        if not isinstance(ticket, Mapping):
            raise ValueError(f"{label}票必须是对象")
        red = ticket.get("red")
        blue = ticket.get("blue")
        if (
            not isinstance(red, list)
            or isinstance(blue, bool)
            or not isinstance(blue, int)
        ):
            raise ValueError(f"{label}票字段非法")
        tickets.append((tuple(cast(list[int], red)), blue))
    portfolio = Portfolio(groups, blues, tuple(tickets))
    normalized = _portfolio_payload(portfolio)
    normalized_keys = set(normalized)
    if not normalized_keys.issubset(payload):
        raise ValueError(f"{label}组合缺少冻结字段")
    if {key: payload[key] for key in normalized_keys} != normalized:
        raise ValueError(f"{label}组合结构、展开或摘要不匹配")
    if len(portfolio.red7_groups) != 5 or len(portfolio.tickets) != PORTFOLIO_COST:
        raise ValueError(f"{label}必须是5x7且恰好35注")
    if len(set(portfolio.tickets)) != PORTFOLIO_COST:
        raise ValueError(f"{label}必须包含35注唯一票")
    return portfolio


def _prediction_inputs(report: Mapping[str, object]) -> tuple[list[float], list[float]]:
    audit = report.get("auditMetadata")
    if not isinstance(audit, Mapping):
        raise ValueError("ensemble报告缺少auditMetadata")
    probabilities = audit.get("finalNextProbabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError("ensemble报告缺少最终前序概率")
    red = probabilities.get("red")
    blue = probabilities.get("blue")
    if not isinstance(red, list) or not isinstance(blue, list):
        raise ValueError("ensemble最终概率格式非法")
    red_values = [float(value) for value in red]
    blue_values = [float(value) for value in blue]
    if len(red_values) != 33 or len(blue_values) != 16:
        raise ValueError("ensemble最终概率维度非法")
    if not all(math.isfinite(value) for value in (*red_values, *blue_values)):
        raise ValueError("ensemble最终概率必须有限")
    return red_values, blue_values


def _trained_state(
    draws: Sequence[SSQDraw],
) -> tuple[dict[str, object], list[float], list[float], list[float]]:
    state = ensemble.FixedEnsembleState()
    for draw in draws:
        state.score_then_update(draw)
    red, blue, pairs = state.predict()
    report = build_bound_ensemble_report(draws)
    report_red, report_blue = _prediction_inputs(report)
    if payload_sha256(red) != payload_sha256(report_red):
        raise RuntimeError("严格前序红球状态与ensemble报告不一致")
    if payload_sha256(blue) != payload_sha256(report_blue):
        raise RuntimeError("严格前序蓝球状态与ensemble报告不一致")
    return report, red, blue, pairs


def _snapshot_payload(
    *,
    version: int,
    draws: Sequence[SSQDraw],
    report: Mapping[str, object],
    red_probabilities: Sequence[float],
    blue_probabilities: Sequence[float],
    pair_modifiers: Sequence[float],
    manifest_sha256: str,
    previous_status_sha256: str | None,
    created_at: datetime,
) -> dict[str, object]:
    target_issue, target_date = _next_target(draws)
    if created_at >= _draw_deadline(target_date):
        raise ValueError("首快照/下一快照必须在目标期开奖截止前生成")
    portfolio_a = _portfolio_a(red_probabilities, blue_probabilities, pair_modifiers)
    portfolio_b, document_b = _portfolio_b(red_probabilities, blue_probabilities)
    controls = [
        build_matched_control_c(target_issue, index) for index in range(CONTROL_COUNT)
    ]
    payload: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "active_snapshot",
        "version": version,
        "createdAt": created_at.isoformat(timespec="seconds"),
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "targetIssue": target_issue,
        "targetDate": target_date,
        "horizon": HORIZON,
        "completedBeforeSnapshot": version,
        "prospectiveProtocolSha256": protocol_sha256(),
        "builderProtocolSha256": builder.protocol_sha256(),
        "dataSha256": canonical_data_sha256(draws),
        "reportSha256": report["reportSha256"],
        "ensembleProtocolSha256": report["protocolSha256"],
        "redProbabilitiesSha256": payload_sha256(list(red_probabilities)),
        "blueProbabilitiesSha256": payload_sha256(list(blue_probabilities)),
        "pairModifiersSha256": payload_sha256(list(pair_modifiers)),
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "portfolios": {
            "A": {
                "role": "concentrated_shared_blue_shadow",
                **_portfolio_payload(portfolio_a),
            },
            "B": {
                "role": "diversified_distinct_blue_primary",
                **_portfolio_payload(portfolio_b),
                "builderDocumentSha256": payload_sha256(document_b),
            },
            "C": {
                "role": "matched_cost_control",
                "count": CONTROL_COUNT,
                "protocolSha256": payload_sha256(
                    {
                        "controlCount": CONTROL_COUNT,
                        "builder": "build_matched_control_c",
                        "targetIssue": target_issue,
                    }
                ),
                "portfolios": [
                    {"controlIndex": index, **_portfolio_payload(portfolio)}
                    for index, portfolio in enumerate(controls)
                ],
            },
        },
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }
    _validate_snapshot(payload)
    return payload


def _completed_snapshot_payload(
    version: int,
    draws: Sequence[SSQDraw],
    report: Mapping[str, object],
    manifest_sha256: str,
    previous_status_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "completed_snapshot",
        "version": version,
        "createdAt": created_at.isoformat(timespec="seconds"),
        "trainedThroughIssue": draws[-1].issue,
        "trainedThroughDate": draws[-1].draw_date,
        "historyRows": len(draws),
        "targetIssue": None,
        "targetDate": None,
        "horizon": HORIZON,
        "completedBeforeSnapshot": version,
        "prospectiveProtocolSha256": protocol_sha256(),
        "builderProtocolSha256": builder.protocol_sha256(),
        "dataSha256": canonical_data_sha256(draws),
        "reportSha256": report["reportSha256"],
        "ensembleProtocolSha256": report["protocolSha256"],
        "manifestArtifactSha256": manifest_sha256,
        "previousStatusArtifactSha256": previous_status_sha256,
        "portfolios": {},
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }


def _validate_snapshot(snapshot: Mapping[str, object]) -> None:
    if snapshot.get("kind") == "completed_snapshot":
        if snapshot.get("targetIssue") is not None or snapshot.get("portfolios") != {}:
            raise ValueError("完成快照不得包含未来目标或组合")
        return
    if snapshot.get("kind") != "active_snapshot":
        raise ValueError("快照kind非法")
    portfolios = snapshot.get("portfolios")
    if not isinstance(portfolios, Mapping):
        raise ValueError("快照缺少A/B/C组合")
    portfolio_a = _portfolio_from_payload(portfolios.get("A"), "A")
    portfolio_b = _portfolio_from_payload(portfolios.get("B"), "B")
    if len(set(portfolio_a.blues)) != 1:
        raise ValueError("A必须使用共享蓝球")
    if len(set(portfolio_b.blues)) != 5:
        raise ValueError("B必须使用5个互异蓝球")
    control_payload = portfolios.get("C")
    if not isinstance(control_payload, Mapping) or control_payload.get("count") != 32:
        raise ValueError("C必须包含32组同成本控制")
    controls = control_payload.get("portfolios")
    if not isinstance(controls, list) or len(controls) != CONTROL_COUNT:
        raise ValueError("C完整组合数量必须恰好为32")
    target_issue = snapshot.get("targetIssue")
    if not isinstance(target_issue, str):
        raise ValueError("活动快照缺少targetIssue")
    for index, control in enumerate(controls):
        if not isinstance(control, Mapping) or control.get("controlIndex") != index:
            raise ValueError("C控制编号不连续")
        parsed = _portfolio_from_payload(control, f"C{index:02d}")
        if parsed != build_matched_control_c(target_issue, index):
            raise ValueError("C控制不匹配固定期号种子协议")


def _observation_payload(
    version: int,
    previous_snapshot: Mapping[str, object] | None,
    draw: SSQDraw | None,
) -> dict[str, object]:
    if version == 0:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "genesis_no_result",
            "version": 0,
            "observed": False,
            "researchOnly": True,
            "predictionClaim": False,
        }
    if previous_snapshot is None or draw is None:
        raise ValueError("结算版本必须提供前一快照与官方结果")
    portfolios = cast(Mapping[str, object], previous_snapshot["portfolios"])
    portfolio_a = _portfolio_from_payload(portfolios["A"], "A")
    portfolio_b = _portfolio_from_payload(portfolios["B"], "B")
    control_group = cast(Mapping[str, object], portfolios["C"])
    controls_raw = cast(list[object], control_group["portfolios"])
    controls = [
        _portfolio_from_payload(control, f"C{index:02d}")
        for index, control in enumerate(controls_raw)
    ]
    score_a = _score(portfolio_a, draw)
    score_b = _score(portfolio_b, draw)
    control_scores = [_score(portfolio, draw) for portfolio in controls]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "settled_observation",
        "version": version,
        "settlesSnapshotVersion": version - 1,
        "snapshotArtifactSha256": previous_snapshot["artifactSha256"],
        "observed": True,
        "officialResult": _draw_payload(draw),
        "metrics": {
            "A": score_a,
            "B": score_b,
            "C32Mean": _issue_mean(control_scores),
            "C32": control_scores,
        },
        "cost": {"A": 35, "B": 35, "eachC": 35, "controlCount": 32},
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "autoPromotion": False,
    }


def _aggregate_observations(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    settled = [item for item in observations if item.get("observed") is True]
    totals = {
        label: {name: 0.0 for name in METRIC_NAMES} for label in ("A", "B", "C32Mean")
    }
    for observation in settled:
        metrics = observation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("observation缺少metrics")
        for label in totals:
            values = metrics.get(label)
            if not isinstance(values, Mapping):
                raise ValueError(f"observation缺少{label}指标")
            for name in METRIC_NAMES:
                value = values.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"observation {label}.{name}非法")
                totals[label][name] += float(value)
    completed = len(settled)
    return {
        "completed": completed,
        "horizon": HORIZON,
        "remaining": HORIZON - completed,
        "metricsPerIssue": {
            label: {
                name: (totals[label][name] / completed if completed else 0.0)
                for name in METRIC_NAMES
            }
            for label in totals
        },
    }


def _status_payload(
    *,
    version: int,
    snapshot_document: Mapping[str, object],
    observation_document: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
    manifest_sha256: str,
    previous_status_sha256: str | None,
) -> dict[str, object]:
    aggregate = _aggregate_observations(observations)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "completed": aggregate["completed"],
        "horizon": HORIZON,
        "remaining": aggregate["remaining"],
        "targetIssue": snapshot_document["targetIssue"],
        "targetDate": snapshot_document["targetDate"],
        "metricsPerIssue": aggregate["metricsPerIssue"],
        "snapshotArtifactSha256": snapshot_document["artifactSha256"],
        "observationArtifactSha256": observation_document["artifactSha256"],
        "previousStatusArtifactSha256": previous_status_sha256,
        "manifestArtifactSha256": manifest_sha256,
        "hmacVerified": True,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "formalCandidates": [],
        "autoPromotion": False,
        "formalActivation": False,
    }


def _version_numbers(root: Path) -> list[int]:
    versions_root = root / "versions"
    if not versions_root.is_dir():
        return []
    versions = sorted(
        int(path.name)
        for path in versions_root.iterdir()
        if path.is_dir() and len(path.name) == 4 and path.name.isdigit()
    )
    if versions != list(range(len(versions))):
        raise ValueError("版本目录必须从0000连续且不可跳号")
    unexpected = [
        path.name
        for path in versions_root.iterdir()
        if not (path.is_dir() and len(path.name) == 4 and path.name.isdigit())
    ]
    if unexpected:
        raise ValueError("versions目录包含未提交或非法条目")
    return versions


def _publish_version(
    root: Path,
    version: int,
    snapshot_payload: Mapping[str, object],
    observation_payload: Mapping[str, object],
    status_factory: Callable[
        [Mapping[str, object], Mapping[str, object]], Mapping[str, object]
    ],
    key: bytes,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    versions_root = root / "versions"
    final_dir = versions_root / f"{version:04d}"
    staging = versions_root / f".{version:04d}.tmp"
    if final_dir.exists() or staging.exists():
        raise FileExistsError(f"版本{version:04d}已存在或残留临时目录")
    staging.mkdir(mode=0o700)
    try:
        snapshot_document = _write_artifact(
            staging / "snapshot.json", snapshot_payload, key
        )
        observation_document = _write_artifact(
            staging / "observation.json", observation_payload, key
        )
        status_payload = status_factory(snapshot_document, observation_document)
        status_document = _write_artifact(staging / "status.json", status_payload, key)
        _fsync_directory(staging)
        os.replace(staging, final_dir)
        _fsync_directory(versions_root)
        try:
            _atomic_replace_artifact(
                root / "current.json",
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "version": version,
                    "statusArtifactSha256": status_document["artifactSha256"],
                },
                key,
            )
        except BaseException:
            shutil.rmtree(final_dir)
            _fsync_directory(versions_root)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return snapshot_document, observation_document, status_document


def register_prospective(
    canonical_csv: str | Path,
    ensemble_report: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """登记唯一未来边界；不生成快照、不写入任何开奖结果。"""

    root = Path(state_dir)
    if root.exists():
        raise FileExistsError("前瞻状态目录已存在，register只允许执行一次")
    draws = load_official_history_csv(canonical_csv)
    report = _load_verified_report(ensemble_report, draws)
    target_issue, target_date = _next_target(draws)
    created_at = _now(now)
    if created_at >= _draw_deadline(target_date):
        raise ValueError("登记时间已越过下一目标期开奖截止")
    staging = root.with_name(f".{root.name}.register.tmp")
    if staging.exists():
        raise FileExistsError("存在登记临时目录，需人工审计后处理")
    staging.mkdir(parents=True, mode=0o700)
    try:
        protocol_document = _write_artifact(
            staging / "protocol.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "prospectiveProtocolSha256": protocol_sha256(),
                "builderProtocolSha256": builder.protocol_sha256(),
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
                "horizon": HORIZON,
                "frozenConfig": FROZEN_CONFIG,
                "canonicalDataSha256": canonical_data_sha256(draws),
                "verifiedReportSha256": report["reportSha256"],
                "protocolArtifactSha256": protocol_document["artifactSha256"],
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
        "latestIssue": draws[-1].issue,
        "targetIssue": target_issue,
        "targetDate": target_date,
        "snapshotCreated": False,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
    }


def _verify_registration(
    root: Path, key: bytes
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = load_and_verify_artifact(root / "protocol.json", key)
    manifest = load_and_verify_artifact(root / "manifest.json", key)
    if protocol.get("prospectiveProtocolSha256") != protocol_sha256():
        raise ValueError("前瞻协议摘要与当前冻结实现不一致")
    if protocol.get("builderProtocolSha256") != builder.protocol_sha256():
        raise ValueError("B构建器协议摘要不一致")
    if manifest.get("protocolArtifactSha256") != protocol.get("artifactSha256"):
        raise ValueError("manifest未绑定protocol artifact")
    if (
        manifest.get("frozenConfig") != FROZEN_CONFIG
        or manifest.get("horizon") != HORIZON
    ):
        raise ValueError("manifest冻结配置不一致")
    return protocol, manifest


def create_snapshot(
    canonical_csv: str | Path,
    ensemble_report: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """登记后生成唯一 0000 首快照，不包含实际开奖结果。"""

    root = Path(state_dir)
    _, manifest = _verify_registration(root, hmac_key)
    if _version_numbers(root):
        raise FileExistsError("首快照已存在，禁止重复生成")
    current = load_and_verify_artifact(root / "current.json", hmac_key)
    if current.get("version") is not None:
        raise ValueError("current指针与空版本目录不一致")
    draws = load_official_history_csv(canonical_csv)
    if len(draws) != manifest["initialHistoryRows"]:
        raise ValueError("snapshot要求canonical仍停留在登记边界")
    if canonical_data_sha256(draws) != manifest["canonicalDataSha256"]:
        raise ValueError("snapshot前canonical登记前缀已变化")
    report = _load_verified_report(ensemble_report, draws)
    if report["reportSha256"] != manifest["verifiedReportSha256"]:
        raise ValueError("snapshot报告与登记时验证报告不一致")
    report_red, report_blue = _prediction_inputs(report)
    state = ensemble.FixedEnsembleState()
    for draw in draws:
        state.score_then_update(draw)
    state_red, state_blue, pair_modifiers = state.predict()
    if payload_sha256(state_red) != payload_sha256(report_red):
        raise ValueError("登记报告红球概率与严格前序状态不一致")
    if payload_sha256(state_blue) != payload_sha256(report_blue):
        raise ValueError("登记报告蓝球概率与严格前序状态不一致")
    snapshot_payload = _snapshot_payload(
        version=0,
        draws=draws,
        report=report,
        red_probabilities=state_red,
        blue_probabilities=state_blue,
        pair_modifiers=pair_modifiers,
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=None,
        created_at=_now(now),
    )
    observation_payload = _observation_payload(0, None, None)

    def status_factory(
        snapshot: Mapping[str, object], observation: Mapping[str, object]
    ) -> dict[str, object]:
        return _status_payload(
            version=0,
            snapshot_document=snapshot,
            observation_document=observation,
            observations=[observation],
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=None,
        )

    snapshot, _, _ = _publish_version(
        root,
        0,
        snapshot_payload,
        observation_payload,
        status_factory,
        hmac_key,
    )
    return {
        "action": "snapshot_created",
        "stateChanged": True,
        "version": 0,
        "completed": 0,
        "horizon": HORIZON,
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
    }


def _verify_chain(root: Path, key: bytes, draws: Sequence[SSQDraw]) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    _, manifest = _verify_registration(root, key)
    versions = _version_numbers(root)
    if not versions:
        raise ValueError("尚未创建首快照")
    initial_rows = cast(int, manifest["initialHistoryRows"])
    settled_versions = len(versions) - 1
    minimum_rows = initial_rows + settled_versions
    if len(draws) not in {minimum_rows, minimum_rows + 1}:
        raise ValueError("canonical长度与前瞻版本边界不相容；update必须恰好一期")
    snapshots: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    previous_status_sha: str | None = None
    for version in versions:
        version_dir = root / "versions" / f"{version:04d}"
        expected_names = {"snapshot.json", "observation.json", "status.json"}
        if {path.name for path in version_dir.iterdir()} != expected_names:
            raise ValueError(f"版本{version:04d}文件集合不符合冻结布局")
        snapshot = load_and_verify_artifact(version_dir / "snapshot.json", key)
        observation = load_and_verify_artifact(version_dir / "observation.json", key)
        status = load_and_verify_artifact(version_dir / "status.json", key)
        if snapshot.get("version") != version or observation.get("version") != version:
            raise ValueError(f"版本{version:04d}内部版本号不一致")
        if status.get("version") != version:
            raise ValueError(f"版本{version:04d}状态版本号不一致")
        if snapshot.get("previousStatusArtifactSha256") != previous_status_sha:
            raise ValueError(f"版本{version:04d}快照链断裂")
        if status.get("previousStatusArtifactSha256") != previous_status_sha:
            raise ValueError(f"版本{version:04d}状态链断裂")
        if status.get("snapshotArtifactSha256") != snapshot.get("artifactSha256"):
            raise ValueError(f"版本{version:04d}状态未绑定快照")
        if status.get("observationArtifactSha256") != observation.get("artifactSha256"):
            raise ValueError(f"版本{version:04d}状态未绑定观测")
        if version == 0:
            if observation.get("kind") != "genesis_no_result":
                raise ValueError("0000 observation不得包含实际结果")
        else:
            if observation.get("settlesSnapshotVersion") != version - 1:
                raise ValueError("observation未结算紧邻前一快照")
            previous_snapshot = snapshots[-1]
            if observation.get("snapshotArtifactSha256") != previous_snapshot.get(
                "artifactSha256"
            ):
                raise ValueError("observation未绑定被结算快照")
            result = observation.get("officialResult")
            expected_draw = draws[initial_rows + version - 1]
            if result != _draw_payload(expected_draw):
                raise ValueError("observation官方结果与canonical精确前缀不一致")
            expected_observation = _observation_payload(
                version, previous_snapshot, expected_draw
            )
            unsigned = {
                key_: value
                for key_, value in observation.items()
                if key_ not in {"artifactSha256", "artifactHmacSha256"}
            }
            if unsigned != expected_observation:
                raise ValueError("observation A/B/C评分复算不一致")
        _validate_snapshot(snapshot)
        snapshots.append(snapshot)
        observations.append(observation)
        expected_status = _status_payload(
            version=version,
            snapshot_document=snapshot,
            observation_document=observation,
            observations=observations,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=previous_status_sha,
        )
        unsigned_status = {
            key_: value
            for key_, value in status.items()
            if key_ not in {"artifactSha256", "artifactHmacSha256"}
        }
        if unsigned_status != expected_status:
            raise ValueError(f"版本{version:04d}累计状态复算不一致")
        previous_status_sha = cast(str, status["artifactSha256"])
        statuses.append(status)
    current = load_and_verify_artifact(root / "current.json", key)
    if current.get("version") != versions[-1]:
        raise ValueError("current版本指针不匹配最新提交")
    if current.get("statusArtifactSha256") != statuses[-1].get("artifactSha256"):
        raise ValueError("current未绑定最新status")
    return manifest, snapshots[-1], observations, statuses


def _assert_active_snapshot_rebuild_compatible(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    """只兼容登记报告哈希漂移；概率、号码、组合等仍须精确相等。"""

    actual_report_sha = actual.get("reportSha256")
    expected_report_sha = expected.get("reportSha256")
    if actual_report_sha != expected_report_sha:
        if actual_report_sha != manifest.get("verifiedReportSha256"):
            raise ValueError("当前快照报告哈希未绑定登记报告")
        normalized_expected = dict(expected)
        normalized_expected["reportSha256"] = actual_report_sha
    else:
        normalized_expected = dict(expected)
    if dict(actual) != normalized_expected:
        raise ValueError("当前快照无法由开奖前精确历史前缀确定性重建")


def _verify_active_snapshot_prefix(
    snapshot: Mapping[str, object],
    manifest: Mapping[str, object],
    draws: Sequence[SSQDraw],
) -> None:
    """用目标期开奖前的精确历史前缀重建当前活动快照。"""

    if snapshot.get("kind") != "active_snapshot":
        raise ValueError("当前快照不是可重建的活动快照")
    created_at_raw = snapshot.get("createdAt")
    if not isinstance(created_at_raw, str):
        raise ValueError("当前快照createdAt非法")
    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except ValueError as error:
        raise ValueError("当前快照createdAt不是ISO时间") from error
    report, red, blue, pairs = _trained_state(draws)
    expected = _snapshot_payload(
        version=cast(int, snapshot["version"]),
        draws=draws,
        report=report,
        red_probabilities=red,
        blue_probabilities=blue,
        pair_modifiers=pairs,
        manifest_sha256=cast(str, manifest["artifactSha256"]),
        previous_status_sha256=cast(
            str | None, snapshot["previousStatusArtifactSha256"]
        ),
        created_at=created_at,
    )
    unsigned = {
        key: value
        for key, value in snapshot.items()
        if key not in {"artifactSha256", "artifactHmacSha256"}
    }
    _assert_active_snapshot_rebuild_compatible(unsigned, expected, manifest)


def update_prospective(
    canonical_csv: str | Path,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    """验签全链，恰好消费一期开奖结果并原子追加下一版本。"""

    root = Path(state_dir)
    draws = load_official_history_csv(canonical_csv)
    manifest, snapshot, observations, statuses = _verify_chain(root, hmac_key, draws)
    completed = len(observations) - 1
    if completed >= HORIZON:
        raise ValueError("500期前瞻观察已完成")
    initial_rows = cast(int, manifest["initialHistoryRows"])
    if len(draws) != initial_rows + completed + 1:
        raise ValueError("update必须且只能新增恰好一期canonical开奖")
    if snapshot.get("kind") != "active_snapshot":
        raise ValueError("当前没有可结算活动快照")
    draw = draws[-1]
    if draw.issue != snapshot.get("targetIssue"):
        raise ValueError("新开奖issue必须精确等于锁定targetIssue")
    if draw.draw_date != snapshot.get("targetDate"):
        raise ValueError("新开奖date必须精确等于锁定targetDate")
    current_time = _now(now)
    if current_time < _draw_deadline(draw.draw_date):
        raise ValueError("目标期开奖截止前禁止update")
    _verify_active_snapshot_prefix(snapshot, manifest, draws[:-1])
    new_version = completed + 1
    observation_payload = _observation_payload(new_version, snapshot, draw)
    report, red, blue, pairs = _trained_state(draws)
    previous_status_sha = cast(str, statuses[-1]["artifactSha256"])
    if new_version == HORIZON:
        snapshot_payload = _completed_snapshot_payload(
            new_version,
            draws,
            report,
            cast(str, manifest["artifactSha256"]),
            previous_status_sha,
            current_time,
        )
    else:
        snapshot_payload = _snapshot_payload(
            version=new_version,
            draws=draws,
            report=report,
            red_probabilities=red,
            blue_probabilities=blue,
            pair_modifiers=pairs,
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=previous_status_sha,
            created_at=current_time,
        )

    def status_factory(
        snapshot_document: Mapping[str, object],
        observation_document: Mapping[str, object],
    ) -> dict[str, object]:
        return _status_payload(
            version=new_version,
            snapshot_document=snapshot_document,
            observation_document=observation_document,
            observations=[*observations, observation_document],
            manifest_sha256=cast(str, manifest["artifactSha256"]),
            previous_status_sha256=previous_status_sha,
        )

    snapshot_document, _, status_document = _publish_version(
        root,
        new_version,
        snapshot_payload,
        observation_payload,
        status_factory,
        hmac_key,
    )
    return {
        "action": "updated",
        "stateChanged": True,
        "version": new_version,
        "completed": new_version,
        "horizon": HORIZON,
        "targetIssue": snapshot_document["targetIssue"],
        "targetDate": snapshot_document["targetDate"],
        "metricsPerIssue": status_document["metricsPerIssue"],
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "formalActivation": False,
        "autoPromotion": False,
    }


def prospective_status(
    state_dir: str | Path = DEFAULT_STATE_DIR,
    *,
    canonical_csv: str | Path,
    hmac_key: bytes,
) -> dict[str, object]:
    """验证完整链并返回不含秘密的 A/B/C 累计状态。"""

    draws = load_official_history_csv(canonical_csv)
    _, snapshot, observations, statuses = _verify_chain(
        Path(state_dir), hmac_key, draws
    )
    status = statuses[-1]
    completed = len(observations) - 1
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": status["version"],
        "completed": completed,
        "horizon": HORIZON,
        "remaining": HORIZON - completed,
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "metricsPerIssue": status["metricsPerIssue"],
        "canonicalVisibleLatestIssue": draws[-1].issue,
        "canonicalVisibleLatestDate": draws[-1].draw_date,
        "hmacVerified": True,
        "researchOnly": True,
        "predictionClaim": False,
        "formalRecommendationStatus": FORMAL_RECOMMENDATION_STATUS,
        "formalCandidates": [],
        "formalActivation": False,
        "autoPromotion": False,
    }


__all__ = [
    "DEFAULT_CANONICAL_CSV",
    "DEFAULT_ENSEMBLE_REPORT",
    "DEFAULT_STATE_DIR",
    "FROZEN_CONFIG",
    "HORIZON",
    "PROTOCOL",
    "build_bound_ensemble_report",
    "canonical_data_sha256",
    "create_snapshot",
    "load_and_verify_artifact",
    "payload_sha256",
    "prospective_status",
    "protocol_sha256",
    "register_prospective",
    "update_prospective",
]
