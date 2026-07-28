# -*- coding: utf-8 -*-
"""快乐8 Pick4 正式前瞻证据链 v2。

本模块只负责本地可验证链；可信前瞻发布时间仍依赖 Telegram 等外部平台。
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence, cast
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy

from src.analysis.kl8_pick4_rank_challenger import (
    PICK4_RANK_FEATURES,
    Kl8Pick4RankConfig,
    audit_rank_probabilities,
    build_pick4_rank_panel,
    build_pick4_ranker,
    ranked_pick4_portfolio,
)
from src.analysis.kl8_pick5_probability_v1 import (
    canonical_kl8_sha256,
    normalize_kl8_dataframe,
)

SCHEMA_VERSION = "kl8_pick4_prospective_v2"
REJECTED_SCHEMA_VERSION = "kl8_pick4_prospective_v1"
CONSUMED_BOUNDARY = {"firstIssue": "2025045", "lastIssue": "2026193", "periods": 500}
DEFAULT_REVIEW_REPORT = Path(
    "reports/development/kl8_pick4_rank_challenger_v2_official_20260723.json"
)
SHANGHAI = ZoneInfo("Asia/Shanghai")
DRAW_HOUR = 21
DRAW_MINUTE = 30
REQUIRED_OBSERVATIONS = 500
CHECKPOINTS = (100, 200, 300, 400, 500)


@dataclass(frozen=True)
class Kl8Pick4ProspectiveConfig:
    """前瞻链配置；formal 配置必须逐字段等于唯一 canonical 常量。"""

    profile: str = "formal"
    formal_eligible: bool = True
    rank_config: Kl8Pick4RankConfig = field(default_factory=Kl8Pick4RankConfig)
    required_observations: int = REQUIRED_OBSERVATIONS
    checkpoints: tuple[int, ...] = CHECKPOINTS
    frozen_periods: int = 500

    @classmethod
    def smoke(
        cls, *, full_history_periods: int | None = None
    ) -> "Kl8Pick4ProspectiveConfig":
        """构造仅限临时目录和直接 core 测试使用的 smoke 配置。"""

        periods = full_history_periods or 24
        if periods < 4:
            raise ValueError("smoke历史至少需要4期")
        rank = Kl8Pick4RankConfig(
            initial_train=max(2, periods - 2),
            evaluation_periods=2,
            refit_interval=2,
            stability_blocks=5,
            seed=20260723,
            n_estimators=4,
            learning_rate=0.04,
            num_leaves=3,
            max_depth=2,
            min_child_samples=2,
            reg_alpha=0.2,
            reg_lambda=1.0,
            lambdarank_truncation_level=4,
            label_gain=(0, 1),
            probability_score_scale=0.1,
            probability_shrinkage=0.1,
            epsilon=1e-6,
            bootstrap_resamples=100,
            bootstrap_block_length=2,
            n_jobs=1,
        )
        return cls(
            profile="smoke",
            formal_eligible=False,
            rank_config=rank,
            required_observations=REQUIRED_OBSERVATIONS,
            checkpoints=CHECKPOINTS,
            frozen_periods=0,
        )

    def to_dict(self) -> dict[str, object]:
        """返回用于签名和严格比较的稳定配置字典。"""

        rank = asdict(self.rank_config)
        rank["label_gain"] = list(self.rank_config.label_gain)
        return {
            "profile": self.profile,
            "formalEligible": self.formal_eligible,
            "rankConfig": rank,
            "requiredObservations": self.required_observations,
            "checkpoints": list(self.checkpoints),
            "frozenPeriods": self.frozen_periods,
        }


FORMAL_CANONICAL_CONFIG = Kl8Pick4ProspectiveConfig().to_dict()


def _validate_config(config: Kl8Pick4ProspectiveConfig) -> None:
    if config.profile == "formal":
        if config.to_dict() != FORMAL_CANONICAL_CONFIG:
            raise ValueError("formal配置必须逐字段匹配唯一canonical常量")
        return
    if config.profile != "smoke" or config.formal_eligible:
        raise ValueError("非正式配置必须明确为smoke且formalEligible=false")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    """计算规范 JSON payload 的 SHA-256。"""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label}必须为整数")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须为数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}必须有限")
    return result


def _artifact_hmac(payload: Mapping[str, object], sha256: str, key: bytes) -> str:
    if not key:
        raise ValueError("HMAC密钥不得为空")
    message = _canonical_bytes(payload) + b"\n" + sha256.encode("ascii")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _artifact_document(payload: Mapping[str, object], key: bytes) -> dict[str, object]:
    unsigned = dict(payload)
    if "artifactSha256" in unsigned or "artifactHmacSha256" in unsigned:
        raise ValueError("payload不得预置artifact签名字段")
    sha256 = payload_sha256(unsigned)
    return {
        **unsigned,
        "artifactSha256": sha256,
        "artifactHmacSha256": _artifact_hmac(unsigned, sha256, key),
    }


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_artifact(
    path: Path, payload: Mapping[str, object], key: bytes
) -> dict[str, object]:
    document = _artifact_document(payload, key)
    _write_bytes(path, _canonical_bytes(document) + b"\n")
    return document


def load_and_verify_artifact(path: str | Path, hmac_key: bytes) -> dict[str, object]:
    """加载 JSON artifact，并同时验证自哈希与 state 外 HMAC。"""

    artifact_path = Path(path)
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取artifact：{artifact_path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"artifact必须为JSON对象：{artifact_path}")
    sha256 = document.get("artifactSha256")
    signature = document.get("artifactHmacSha256")
    if not isinstance(sha256, str) or not isinstance(signature, str):
        raise ValueError(f"artifact缺少SHA/HMAC：{artifact_path}")
    payload = {
        k: v
        for k, v in document.items()
        if k not in {"artifactSha256", "artifactHmacSha256"}
    }
    expected_sha = payload_sha256(payload)
    if not hmac.compare_digest(sha256, expected_sha):
        raise ValueError(f"artifact自哈希不匹配：{artifact_path}")
    expected_hmac = _artifact_hmac(payload, sha256, hmac_key)
    if not hmac.compare_digest(signature, expected_hmac):
        raise ValueError(f"artifact HMAC不匹配：{artifact_path}")
    return cast(dict[str, object], document)


def _model_hmac(content: bytes, sha256: str, key: bytes) -> str:
    return hmac.new(
        key, content + b"\n" + sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()


def _model_binding(content: bytes, key: bytes) -> tuple[str, str]:
    sha256 = _bytes_sha256(content)
    return sha256, _model_hmac(content, sha256, key)


def _verify_model(path: Path, sha256: object, signature: object, key: bytes) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"历史模型缺失：{path}") from error
    actual_sha, actual_hmac = _model_binding(content, key)
    if sha256 != actual_sha or signature != actual_hmac:
        raise ValueError(f"模型SHA/HMAC不匹配：{path}")
    return content.decode("utf-8")


def load_full_canonical_csv(path: str | Path) -> pd.DataFrame:
    """加载 canonical KL8 CSV 并严格规范化。"""

    frame = pd.read_csv(path, dtype={"issue": str, "date": str, "numbers": str})
    return normalize_kl8_dataframe(frame)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _environment_fingerprint(root: Path | None = None) -> dict[str, object]:
    project = root or _project_root()
    dependency_files = ("requirements.txt", "requirements-dev.txt", "pyproject.toml")
    source_files = (
        "src/analysis/kl8_pick4_prospective.py",
        "src/analysis/kl8_pick4_rank_challenger.py",
        "src/analysis/kl8_feature_discovery_v2.py",
        "src/analysis/kl8_pick5_probability_v1.py",
    )
    hashes: dict[str, str] = {}
    for relative in (*dependency_files, *source_files):
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"环境指纹必需文件缺失：{relative}")
        hashes[relative] = _bytes_sha256(path.read_bytes())
    return {
        "sysVersion": sys.version,
        "packages": {
            "lightgbm": importlib.metadata.version("lightgbm"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "fileSha256": hashes,
    }


def build_next_issue_features(
    history: pd.DataFrame, *, sentinel_numbers: Sequence[int] | None = None
) -> pd.DataFrame:
    """用 prior-only 哨兵行构造下一期 80 个号码特征。"""

    normalized = normalize_kl8_dataframe(history)
    if normalized.empty:
        raise ValueError("至少需要1期历史")
    sentinel = list(sentinel_numbers or range(1, 21))
    if len(sentinel) != 20 or len(set(sentinel)) != 20:
        raise ValueError("哨兵必须是20个唯一号码")
    target_issue, target_date = _next_target(normalized)
    appended = pd.concat(
        [
            normalized,
            pd.DataFrame(
                [{"issue": target_issue, "date": target_date, "numbers": sentinel}]
            ),
        ],
        ignore_index=True,
    )
    panel = build_pick4_rank_panel(appended)
    result = panel.loc[
        panel["periodIndex"] == len(normalized), ["number", *PICK4_RANK_FEATURES]
    ].reset_index(drop=True)
    if len(result) != 80 or result["number"].tolist() != list(range(1, 81)):
        raise RuntimeError("下一期特征必须严格覆盖1..80")
    return result


def _fit_model(history: pd.DataFrame, config: Kl8Pick4ProspectiveConfig) -> str:
    panel = build_pick4_rank_panel(history)
    periods = len(history)
    ranker = build_pick4_ranker(config.rank_config)
    ranker.fit(
        panel.loc[:, PICK4_RANK_FEATURES].to_numpy(dtype=np.float64),
        panel["target"].to_numpy(dtype=np.int8),
        group=np.full(periods, 80, dtype=np.int32),
    )
    return str(ranker.booster_.model_to_string())


def _predict_snapshot_values(
    history: pd.DataFrame, model_text: str, config: Kl8Pick4ProspectiveConfig
) -> tuple[list[int], list[list[int]], list[float]]:
    features = build_next_issue_features(history)
    scores = np.asarray(
        lgb.Booster(model_str=model_text).predict(features.loc[:, PICK4_RANK_FEATURES]),
        dtype=np.float64,
    )
    if scores.shape != (80,) or not np.isfinite(scores).all():
        raise ValueError("模型分数必须是80维有限向量")
    ranking = np.lexsort((np.arange(80), -scores)) + 1
    primary = sorted(int(value) for value in ranking[:4])
    tickets = ranked_pick4_portfolio(scores)
    probabilities = audit_rank_probabilities(scores, config.rank_config)
    return primary, tickets, [float(value) for value in probabilities]


def _next_target(history: pd.DataFrame) -> tuple[str, str]:
    latest = history.iloc[-1]
    latest_date = datetime.strptime(str(latest["date"]), "%Y-%m-%d").date()
    target_date = latest_date + timedelta(days=1)
    issue = str(latest["issue"])
    issue_year, sequence = int(issue[:4]), int(issue[4:])
    if target_date.year == issue_year:
        target_issue = f"{issue_year}{sequence + 1:03d}"
    else:
        target_issue = f"{target_date.year}001"
    return target_issue, target_date.isoformat()


def _now_shanghai(now: datetime | None = None) -> datetime:
    value = now or datetime.now(tz=SHANGHAI)
    if value.tzinfo is None:
        raise ValueError("now必须带时区")
    return value.astimezone(SHANGHAI)


def _draw_deadline(target_date: str) -> datetime:
    day = datetime.strptime(target_date, "%Y-%m-%d")
    return day.replace(hour=DRAW_HOUR, minute=DRAW_MINUTE, tzinfo=SHANGHAI)


def _created_at_for_target(target_date: str, now: datetime | None) -> str:
    created = _now_shanghai(now)
    if created >= _draw_deadline(target_date):
        raise ValueError("已晚于目标期当日21:30，fail closed；不可创建当日目标快照")
    return created.isoformat(timespec="seconds")


def _validate_created_at(created_at: object, target_date: object) -> None:
    if not isinstance(created_at, str) or not isinstance(target_date, str):
        raise ValueError("createdAt/targetDate格式无效")
    try:
        value = datetime.fromisoformat(created_at).astimezone(SHANGHAI)
    except ValueError as error:
        raise ValueError("createdAt必须为ISO时区时间") from error
    if value >= _draw_deadline(target_date):
        raise ValueError("createdAt必须早于目标期当日21:30+08:00")


def _validate_snapshot_predictions(snapshot: Mapping[str, object]) -> None:
    probabilities = np.asarray(snapshot.get("probabilities80"), dtype=np.float64)
    if probabilities.shape != (80,) or not np.isfinite(probabilities).all():
        raise ValueError("概率必须是80维有限向量")
    if not np.all((probabilities > 0.0) & (probabilities < 1.0)):
        raise ValueError("每个概率必须严格满足0<p<1")
    if not np.isclose(float(probabilities.sum()), 20.0, atol=1e-10):
        raise ValueError("概率总和必须为20")
    tickets_raw = snapshot.get("fiveTickets")
    primary_raw = snapshot.get("primaryTop4")
    if not isinstance(tickets_raw, list) or len(tickets_raw) != 5:
        raise ValueError("fiveTickets必须恰好5票")
    tickets = [
        [_integer(value, "票面号码") for value in cast(list[object], ticket)]
        for ticket in tickets_raw
    ]
    if any(len(ticket) != 4 or len(set(ticket)) != 4 for ticket in tickets):
        raise ValueError("每票必须恰好4个唯一号码")
    union = {value for ticket in tickets for value in ticket}
    if len(union) != 20 or any(value < 1 or value > 80 for value in union):
        raise ValueError("五票必须恰好覆盖模型Top20")
    if not isinstance(primary_raw, list) or len(primary_raw) != 4:
        raise ValueError("primaryTop4必须恰好4个号码")
    primary = {_integer(value, "primaryTop4号码") for value in primary_raw}
    if len(primary) != 4 or not primary.issubset(union):
        raise ValueError("primaryTop4必须属于五票Top20并集")


def _snapshot_payload(
    version: int,
    history: pd.DataFrame,
    model_sha: str,
    model_hmac: str,
    config: Kl8Pick4ProspectiveConfig,
    created_at: str,
    predictions: tuple[list[int], list[list[int]], list[float]],
) -> dict[str, object]:
    target_issue, target_date = _next_target(history)
    primary, tickets, probabilities = predictions
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "profile": config.profile,
        "formalEligible": config.formal_eligible,
        "historicalFrozenConsumed": config.profile == "formal",
        "afterIssue": str(history.iloc[-1]["issue"]),
        "targetIssue": target_issue,
        "targetDate": target_date,
        "createdAt": created_at,
        "historyRows": len(history),
        "dataSha256": canonical_kl8_sha256(history),
        "modelSha256": model_sha,
        "modelHmacSha256": model_hmac,
        "primaryTop4": primary,
        "fiveTickets": tickets,
        "probabilities80": probabilities,
        "externalAnchorRequired": True,
    }
    _validate_snapshot_predictions(payload)
    return payload


def _anchor_payload(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": snapshot["version"],
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "createdAt": snapshot["createdAt"],
        "snapshotArtifactSha256": snapshot["artifactSha256"],
        "snapshotArtifactHmacSha256": snapshot["artifactHmacSha256"],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_version(
    state_dir: Path, version: int, files: Mapping[str, object], hmac_key: bytes
) -> dict[str, object]:
    versions = state_dir / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    destination = versions / f"{version:04d}"
    if destination.exists():
        raise FileExistsError(f"已提交版本不可覆盖：{destination}")
    staging = versions / f"{version:04d}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    documents: dict[str, dict[str, object]] = {}
    model_bytes = cast(bytes, files["model.txt"])
    _write_bytes(staging / "model.txt", model_bytes)
    for name in (
        "snapshot.json",
        "anchor_payload.json",
        "observation.json",
        "state.json",
    ):
        if name in files:
            documents[name] = _write_artifact(
                staging / name, cast(Mapping[str, object], files[name]), hmac_key
            )
    manifest = {
        name: _bytes_sha256((staging / name).read_bytes())
        for name in sorted(path.name for path in staging.iterdir())
    }
    state_document = documents["state.json"]
    commit = _write_artifact(
        staging / "commit.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "version": version,
            "manifestBytesSha256": manifest,
            "stateArtifactSha256": state_document["artifactSha256"],
            "snapshotArtifactSha256": documents["snapshot.json"]["artifactSha256"],
            "modelSha256": _bytes_sha256(model_bytes),
            "observationArtifactSha256": (
                documents.get("observation.json", {}).get("artifactSha256")
            ),
        },
        hmac_key,
    )
    _fsync_directory(staging)
    staging.rename(destination)
    _fsync_directory(versions)
    return commit


def _formal_evidence(
    history: pd.DataFrame, raw_jsonl: Path, review_report: Path
) -> dict[str, object]:
    if not raw_jsonl.is_file() or not review_report.is_file():
        raise FileNotFoundError("formal登记要求raw JSONL和旧审查报告均存在")
    if len(history) < 500:
        raise ValueError("formal canonical历史不足500期")
    frozen = history.iloc[-500:].reset_index(drop=True)
    if (
        str(frozen.iloc[0]["issue"]) != CONSUMED_BOUNDARY["firstIssue"]
        or str(frozen.iloc[-1]["issue"]) != CONSUMED_BOUNDARY["lastIssue"]
        or len(frozen) != CONSUMED_BOUNDARY["periods"]
    ):
        raise ValueError("旧Frozen边界必须逐字段为2025045..2026193恰500期")
    raw_rows: list[dict[str, object]] = []
    for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        raw_rows.append(
            {
                "issue": str(record["issue"]),
                "date": str(record["date"]),
                "numbers": record["numbers"],
            }
        )
    raw_frame = normalize_kl8_dataframe(pd.DataFrame(raw_rows))
    if canonical_kl8_sha256(raw_frame) != canonical_kl8_sha256(history):
        raise ValueError("raw JSONL与canonical CSV完整语义不一致")
    return {
        "historicalFrozenConsumed": True,
        "boundary": CONSUMED_BOUNDARY,
        "rawJsonlBytesSha256": _bytes_sha256(raw_jsonl.read_bytes()),
        "canonicalFullSemanticSha256": canonical_kl8_sha256(history),
        "consumed500SemanticSha256": canonical_kl8_sha256(frozen),
        "reviewReportPath": str(review_report),
        "reviewReportBytesSha256": _bytes_sha256(review_report.read_bytes()),
    }


def _ensure_smoke_temp_dir(state_dir: Path, config: Kl8Pick4ProspectiveConfig) -> None:
    if config.profile != "smoke":
        return
    temporary = Path(tempfile.gettempdir()).resolve()
    try:
        state_dir.resolve().relative_to(temporary)
    except ValueError as error:
        raise ValueError("smoke只能写入系统临时路径") from error


def register_prospective(
    csv_path: str | Path,
    state_dir: str | Path,
    *,
    hmac_key: bytes,
    raw_jsonl: str | Path | None = None,
    review_report: str | Path | None = None,
    consume_historical_frozen: bool = False,
    config: Kl8Pick4ProspectiveConfig | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """事务化登记 v2 第0版，返回开奖前需外部发布的锚点 payload。"""

    active = config or Kl8Pick4ProspectiveConfig()
    _validate_config(active)
    destination = Path(state_dir)
    _ensure_smoke_temp_dir(destination, active)
    if destination.exists():
        raise FileExistsError(f"state目录已存在：{destination}")
    history = load_full_canonical_csv(csv_path)
    if active.profile == "formal" and not consume_historical_frozen:
        raise ValueError("formal登记必须显式确认消费旧Frozen")
    target_issue, target_date = _next_target(history)
    created_at = _created_at_for_target(target_date, now)
    environment = _environment_fingerprint()
    evidence: dict[str, object] = {"historicalFrozenConsumed": False}
    if active.profile == "formal":
        if raw_jsonl is None:
            raise ValueError("formal登记必须提供--raw-jsonl")
        report = (
            Path(review_report)
            if review_report
            else _project_root() / DEFAULT_REVIEW_REPORT
        )
        evidence = _formal_evidence(history, Path(raw_jsonl), report)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.register.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        protocol = _write_artifact(
            staging / "protocol.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "profile": active.profile,
                "formalEligible": active.formal_eligible,
                "canonicalConfig": active.to_dict(),
                "configSha256": payload_sha256(active.to_dict()),
                "environmentFingerprint": environment,
                "initialHistoryRows": len(history),
                "initialDataSha256": canonical_kl8_sha256(history),
                "requiredObservations": active.required_observations,
                "checkpoints": list(active.checkpoints),
                "targetRule": "daily_exact_issue_same_year_seq_plus_1_cross_year_001",
                "reschedulePolicy": "not_implemented_fail_closed",
            },
            hmac_key,
        )
        lineage = _write_artifact(
            staging / "lineage.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "rejectedLineage": {
                    "schemaVersion": REJECTED_SCHEMA_VERSION,
                    "status": "review_rejected",
                    "officialChain": False,
                },
                "frozenEvidence": evidence,
                "localTrustLimitation": "本地所有者可控制文件与密钥；正式证据依赖外部不可事后改写时间戳",
            },
            hmac_key,
        )
        model_bytes = _fit_model(history, active).encode("utf-8")
        model_sha, model_hmac = _model_binding(model_bytes, hmac_key)
        snapshot_payload = _snapshot_payload(
            0,
            history,
            model_sha,
            model_hmac,
            active,
            created_at,
            _predict_snapshot_values(history, model_bytes.decode("utf-8"), active),
        )
        snapshot_document = _artifact_document(snapshot_payload, hmac_key)
        anchor_payload = _anchor_payload(snapshot_document)
        state_payload = {
            "schemaVersion": SCHEMA_VERSION,
            "version": 0,
            "profile": active.profile,
            "formalEligible": active.formal_eligible,
            "observed": 0,
            "latestIssue": str(history.iloc[-1]["issue"]),
            "dataSha256": canonical_kl8_sha256(history),
            "protocolArtifactSha256": protocol["artifactSha256"],
            "lineageArtifactSha256": lineage["artifactSha256"],
            "previousStateArtifactSha256": None,
            "previousModelSha256": None,
            "observationArtifactSha256": None,
            "currentSnapshotArtifactSha256": snapshot_document["artifactSha256"],
            "modelSha256": model_sha,
            "modelHmacSha256": model_hmac,
        }
        _publish_version(
            staging,
            0,
            {
                "model.txt": model_bytes,
                "snapshot.json": snapshot_payload,
                "anchor_payload.json": anchor_payload,
                "state.json": state_payload,
            },
            hmac_key,
        )
        _fsync_directory(staging)
        staging.rename(destination)
        _fsync_directory(parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "status": "registered",
        "schemaVersion": SCHEMA_VERSION,
        "version": 0,
        "targetIssue": target_issue,
        "targetDate": target_date,
        "externalAnchorPayload": anchor_payload,
    }


def _version_numbers(root: Path) -> list[int]:
    versions = root / "versions"
    if not versions.is_dir():
        raise ValueError("versions目录缺失")
    for partial in versions.glob("*.tmp"):
        shutil.rmtree(partial)
    values = sorted(
        int(path.name)
        for path in versions.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if values != list(range(len(values))):
        raise ValueError("已提交版本必须从0000严格连续")
    return values


def _verify_commit(version_dir: Path, key: bytes) -> dict[str, object]:
    commit = load_and_verify_artifact(version_dir / "commit.json", key)
    manifest = commit.get("manifestBytesSha256")
    if not isinstance(manifest, dict):
        raise ValueError("commit manifest格式无效")
    actual = {
        name: _bytes_sha256((version_dir / name).read_bytes())
        for name in sorted(manifest)
        if name != "commit.json"
    }
    if actual != manifest:
        raise ValueError(f"commit manifest不匹配：{version_dir}")
    return commit


def _proper_scores(
    probabilities: Sequence[float], actual_numbers: set[int]
) -> tuple[float, float, float, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(
        [1.0 if number in actual_numbers else 0.0 for number in range(1, 81)]
    )
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    baseline = np.full(80, 0.25)
    log_loss = float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    )
    base_log = float(
        -np.mean(target * np.log(baseline) + (1.0 - target) * np.log(1.0 - baseline))
    )
    brier = float(np.mean((values - target) ** 2))
    base_brier = float(np.mean((baseline - target) ** 2))
    return log_loss, brier, log_loss - base_log, brier - base_brier


def _observation_payload(
    previous_snapshot: Mapping[str, object],
    row: Mapping[str, object],
    receipt: Mapping[str, object],
) -> dict[str, object]:
    actual = {
        _integer(value, "开奖号码") for value in cast(Sequence[object], row["numbers"])
    }
    primary = {
        _integer(value, "primaryTop4号码")
        for value in cast(Sequence[object], previous_snapshot["primaryTop4"])
    }
    tickets = cast(Sequence[Sequence[object]], previous_snapshot["fiveTickets"])
    ticket_hits = [
        len(actual & {_integer(value, "票面号码") for value in ticket})
        for ticket in tickets
    ]
    log_loss, brier, delta_log, delta_brier = _proper_scores(
        cast(Sequence[float], previous_snapshot["probabilities80"]), actual
    )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "version": _integer(previous_snapshot["version"], "快照版本") + 1,
        "issue": str(row["issue"]),
        "date": str(row["date"]),
        "numbers": sorted(actual),
        "evaluatedSnapshotArtifactSha256": previous_snapshot["artifactSha256"],
        "evaluatedSnapshotArtifactHmacSha256": previous_snapshot["artifactHmacSha256"],
        "snapshotCreatedAt": previous_snapshot["createdAt"],
        "primaryHits": len(actual & primary),
        "ticketHits": ticket_hits,
        "portfolioTotalHits": sum(ticket_hits),
        "logLoss": log_loss,
        "brier": brier,
        "deltaLogLoss": delta_log,
        "deltaBrier": delta_brier,
        "externalAnchorVerified": True,
        "anchorProvider": receipt["provider"],
        "anchorMessageId": receipt["messageId"],
        "anchoredAt": receipt["anchoredAt"],
    }


def _verify_receipt(
    receipt_path: str | Path, anchor: Mapping[str, object], used: set[tuple[str, str]]
) -> dict[str, object]:
    try:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("外部锚点回执不可读") from error
    if not isinstance(receipt, dict):
        raise ValueError("外部锚点回执必须为JSON对象")
    for key in (
        "version",
        "targetIssue",
        "targetDate",
        "createdAt",
        "snapshotArtifactSha256",
        "snapshotArtifactHmacSha256",
    ):
        if receipt.get(key) != anchor.get(key):
            raise ValueError(f"外部锚点回执未绑定{key}")
    provider, message_id = receipt.get("provider"), receipt.get("messageId")
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(message_id, str)
        or not message_id
    ):
        raise ValueError("回执必须含唯一provider/messageId")
    if (provider, message_id) in used:
        raise ValueError("provider/messageId必须全链唯一")
    anchored = datetime.fromisoformat(str(receipt.get("anchoredAt"))).astimezone(
        SHANGHAI
    )
    if anchored >= _draw_deadline(str(anchor["targetDate"])):
        raise ValueError("外部锚点必须在开奖前完成")
    return cast(dict[str, object], receipt)


def _compare_observation(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    ignored = {"artifactSha256", "artifactHmacSha256"}
    if {k: v for k, v in actual.items() if k not in ignored} != dict(expected):
        raise ValueError("observation与canonical开奖重导结果不一致")


def _verify_chain(
    root: Path,
    key: bytes,
    history: pd.DataFrame,
    config: Kl8Pick4ProspectiveConfig,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    protocol = load_and_verify_artifact(root / "protocol.json", key)
    lineage = load_and_verify_artifact(root / "lineage.json", key)
    if (
        protocol.get("schemaVersion") != SCHEMA_VERSION
        or protocol.get("canonicalConfig") != config.to_dict()
    ):
        raise ValueError("协议schema/config不匹配")
    if protocol.get("environmentFingerprint") != _environment_fingerprint():
        raise ValueError("运行环境指纹不匹配")
    if lineage.get("rejectedLineage") != {
        "schemaVersion": REJECTED_SCHEMA_VERSION,
        "status": "review_rejected",
        "officialChain": False,
    }:
        raise ValueError("v1被否决谱系声明缺失")
    initial_rows = _integer(protocol["initialHistoryRows"], "initialHistoryRows")
    versions = _version_numbers(root)
    if not versions or len(history) != initial_rows + versions[-1]:
        raise ValueError("canonical历史行数必须等于初始历史+已观测版本")
    observations: list[dict[str, object]] = []
    previous_state_sha: object = None
    previous_model_sha: object = None
    previous_snapshot: dict[str, object] | None = None
    used_receipts: set[tuple[str, str]] = set()
    latest_state: dict[str, object] = {}
    latest_snapshot: dict[str, object] = {}
    for version in versions:
        version_dir = root / "versions" / f"{version:04d}"
        commit = _verify_commit(version_dir, key)
        state = load_and_verify_artifact(version_dir / "state.json", key)
        snapshot = load_and_verify_artifact(version_dir / "snapshot.json", key)
        anchor = load_and_verify_artifact(version_dir / "anchor_payload.json", key)
        if state.get("artifactSha256") != commit.get("stateArtifactSha256"):
            raise ValueError("commit未绑定state")
        model_text = _verify_model(
            version_dir / "model.txt",
            state.get("modelSha256"),
            state.get("modelHmacSha256"),
            key,
        )
        prefix = history.iloc[: initial_rows + version].reset_index(drop=True)
        target_issue, target_date = _next_target(prefix)
        checks = (
            (state.get("version"), version, "state版本"),
            (snapshot.get("version"), version, "snapshot版本"),
            (
                state.get("previousStateArtifactSha256"),
                previous_state_sha,
                "previous state",
            ),
            (state.get("previousModelSha256"), previous_model_sha, "previous model"),
            (
                state.get("protocolArtifactSha256"),
                protocol.get("artifactSha256"),
                "protocol pointer",
            ),
            (
                state.get("lineageArtifactSha256"),
                lineage.get("artifactSha256"),
                "lineage pointer",
            ),
            (snapshot.get("targetIssue"), target_issue, "targetIssue"),
            (snapshot.get("targetDate"), target_date, "targetDate"),
            (
                snapshot.get("dataSha256"),
                canonical_kl8_sha256(prefix),
                "prefix语义哈希",
            ),
            (snapshot.get("modelSha256"), state.get("modelSha256"), "snapshot model"),
            (
                state.get("currentSnapshotArtifactSha256"),
                snapshot.get("artifactSha256"),
                "current snapshot",
            ),
            (
                anchor.get("snapshotArtifactSha256"),
                snapshot.get("artifactSha256"),
                "anchor snapshot",
            ),
            (
                anchor.get("snapshotArtifactHmacSha256"),
                snapshot.get("artifactHmacSha256"),
                "anchor HMAC",
            ),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(f"全链{label}不匹配")
        _validate_created_at(snapshot.get("createdAt"), snapshot.get("targetDate"))
        _validate_snapshot_predictions(snapshot)
        if version > 0:
            observation = load_and_verify_artifact(
                version_dir / "observation.json", key
            )
            if previous_snapshot is None:
                raise ValueError("上一快照缺失")
            row = cast(
                Mapping[str, object], history.iloc[initial_rows + version - 1].to_dict()
            )
            receipt_projection = {
                "provider": observation.get("anchorProvider"),
                "messageId": observation.get("anchorMessageId"),
                "anchoredAt": observation.get("anchoredAt"),
            }
            expected_observation = _observation_payload(
                previous_snapshot, row, receipt_projection
            )
            _compare_observation(observation, expected_observation)
            receipt_key = (
                str(observation["anchorProvider"]),
                str(observation["anchorMessageId"]),
            )
            if receipt_key in used_receipts:
                raise ValueError("历史外部回执不唯一")
            used_receipts.add(receipt_key)
            anchored = datetime.fromisoformat(
                str(observation["anchoredAt"])
            ).astimezone(SHANGHAI)
            if anchored >= _draw_deadline(str(observation["date"])):
                raise ValueError("历史外部锚点晚于开奖")
            if state.get("observationArtifactSha256") != observation.get(
                "artifactSha256"
            ):
                raise ValueError("state未绑定observation")
            observations.append(observation)
        elif state.get("observationArtifactSha256") is not None:
            raise ValueError("v0不得绑定observation")
        previous_state_sha = state["artifactSha256"]
        previous_model_sha = state["modelSha256"]
        previous_snapshot = snapshot
        latest_state, latest_snapshot = state, snapshot
        expected = _predict_snapshot_values(prefix, model_text, config)
        actual = (
            snapshot["primaryTop4"],
            snapshot["fiveTickets"],
            snapshot["probabilities80"],
        )
        if _canonical_bytes(actual) != _canonical_bytes(expected):
            raise ValueError("快照与对应history prefix+model+config重算不一致")
    return latest_state, latest_snapshot, observations


def _summarize(
    observations: Sequence[Mapping[str, object]], config: Kl8Pick4ProspectiveConfig
) -> tuple[dict[str, object], list[dict[str, object]]]:
    count = len(observations)
    primary = np.asarray(
        [_number(item["primaryHits"], "primaryHits") for item in observations]
    )
    tickets = np.asarray(
        [item["ticketHits"] for item in observations], dtype=np.float64
    )
    portfolio = np.asarray(
        [
            _number(item["portfolioTotalHits"], "portfolioTotalHits")
            for item in observations
        ]
    )
    delta_log = np.asarray(
        [_number(item["deltaLogLoss"], "deltaLogLoss") for item in observations]
    )
    delta_brier = np.asarray(
        [_number(item["deltaBrier"], "deltaBrier") for item in observations]
    )

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else 0.0

    def pvalue(values: np.ndarray, null: float, *, lower: bool = False) -> float:
        if values.size == 0:
            return 1.0
        centered = (-values if lower else values) - (-null if lower else null)
        rng = np.random.default_rng(config.rank_config.seed + values.size)
        means = np.empty(config.rank_config.bootstrap_resamples)
        for index in range(len(means)):
            means[index] = rng.choice(centered, size=len(centered), replace=True).mean()
        return float((1 + np.sum(means <= 0.0)) / (len(means) + 1))

    raw_p = {
        "primaryTop4": pvalue(primary, 1.0),
        "meanHitsPerTicket": pvalue(tickets.reshape(-1), 1.0),
        "portfolio": pvalue(portfolio, 5.0),
        "logLoss": pvalue(delta_log, 0.0, lower=True),
        "brier": pvalue(delta_brier, 0.0, lower=True),
    }
    ordered = sorted(raw_p, key=raw_p.get)  # type: ignore[arg-type]
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, raw_p[name] * (len(ordered) - index)))
        adjusted[name] = running
    block_size = count // 5 if count >= 5 else 0
    blocks: list[dict[str, object]] = []
    if block_size:
        for index in range(5):
            start = index * block_size
            stop = count if index == 4 else (index + 1) * block_size
            blocks.append(
                {
                    "primaryMeanHits": mean(primary[start:stop]),
                    "meanHitsPerTicket": mean(tickets[start:stop]),
                    "portfolioMeanTotalHits": mean(portfolio[start:stop]),
                    "deltaLogLoss": mean(delta_log[start:stop]),
                    "deltaBrier": mean(delta_brier[start:stop]),
                }
            )
    return (
        {
            "observed": count,
            "profile": config.profile,
            "formalEligible": config.formal_eligible,
            "primaryMeanHits": mean(primary),
            "meanHitsPerTicket": mean(tickets),
            "portfolioMeanTotalHits": mean(portfolio),
            "deltaLogLoss": mean(delta_log),
            "deltaBrier": mean(delta_brier),
            "holmAdjustedPValues": adjusted,
            "externalAnchorVerifiedCount": sum(
                item.get("externalAnchorVerified") is True for item in observations
            ),
        },
        blocks,
    )


def evaluate_formal_gate(
    summary: Mapping[str, object], blocks: Sequence[Mapping[str, object]]
) -> tuple[bool, list[str]]:
    """仅允许恰好500条、formalEligible 的正式链激活。"""

    reasons: list[str] = []
    if summary.get("profile") != "formal" or summary.get("formalEligible") is not True:
        reasons.append("not_formal_eligible")
    if summary.get("observed") != REQUIRED_OBSERVATIONS:
        reasons.append("observed_must_equal_500")
    if summary.get("externalAnchorVerifiedCount") != REQUIRED_OBSERVATIONS:
        reasons.append("all_external_anchors_required")
    thresholds = {
        "primaryMeanHits": (1.0, "above"),
        "meanHitsPerTicket": (1.0, "above"),
        "portfolioMeanTotalHits": (5.0, "above"),
        "deltaLogLoss": (0.0, "below"),
        "deltaBrier": (0.0, "below"),
    }
    for name, (threshold, direction) in thresholds.items():
        value = summary.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            reasons.append(f"{name}_not_finite")
        elif direction == "above" and float(value) <= threshold:
            reasons.append(f"{name}_not_above_baseline")
        elif direction == "below" and float(value) >= threshold:
            reasons.append(f"{name}_not_below_baseline")
    pvalues = summary.get("holmAdjustedPValues")
    if not isinstance(pvalues, Mapping) or len(pvalues) != 5:
        reasons.append("pvalues_invalid")
    else:
        for name, value in pvalues.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                reasons.append(f"pvalue_{name}_outside_0_1")
            elif float(value) > 0.01:
                reasons.append(f"pvalue_{name}_above_0_01")
    if len(blocks) != 5:
        reasons.append("five_stability_blocks_required")
    else:
        for index, block in enumerate(blocks):
            for name, (threshold, direction) in thresholds.items():
                value = block.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ):
                    reasons.append(f"block_{index}_{name}_not_finite")
                elif direction == "above" and float(value) <= threshold:
                    reasons.append(f"block_{index}_{name}_failed")
                elif direction == "below" and float(value) >= threshold:
                    reasons.append(f"block_{index}_{name}_failed")
    return not reasons, reasons


def update_prospective(
    csv_path: str | Path,
    state_dir: str | Path,
    *,
    hmac_key: bytes,
    anchor_receipt_file: str | Path,
    config: Kl8Pick4ProspectiveConfig | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """恰好消费当前快照 targetIssue/targetDate，并事务发布下一版。"""

    active = config or Kl8Pick4ProspectiveConfig()
    _validate_config(active)
    root = Path(state_dir)
    protocol = load_and_verify_artifact(root / "protocol.json", hmac_key)
    initial_rows = _integer(protocol["initialHistoryRows"], "initialHistoryRows")
    history = load_full_canonical_csv(csv_path)
    committed_versions = _version_numbers(root)
    if not committed_versions:
        raise ValueError("没有可更新的已提交版本")
    version = committed_versions[-1]
    if len(history) != initial_rows + version + 1:
        raise ValueError("update必须恰好新增1期canonical开奖")
    previous_history = history.iloc[:-1].reset_index(drop=True)
    state, snapshot, observations = _verify_chain(
        root, hmac_key, previous_history, active
    )
    row = history.iloc[-1]
    if str(row["issue"]) != snapshot["targetIssue"]:
        raise ValueError("新开奖issue必须精确等于targetIssue")
    if str(row["date"]) != snapshot["targetDate"]:
        raise ValueError("新开奖date必须精确等于targetDate")
    anchor = load_and_verify_artifact(
        root / "versions" / f"{version:04d}" / "anchor_payload.json", hmac_key
    )
    used = {
        (str(item["anchorProvider"]), str(item["anchorMessageId"]))
        for item in observations
    }
    receipt = _verify_receipt(anchor_receipt_file, anchor, used)
    observation_payload = _observation_payload(
        snapshot, cast(Mapping[str, object], row.to_dict()), receipt
    )
    observation_document = _artifact_document(observation_payload, hmac_key)
    new_version = version + 1
    new_target_issue, new_target_date = _next_target(history)
    created_at = _created_at_for_target(new_target_date, now)
    model_bytes = _fit_model(history, active).encode("utf-8")
    model_sha, model_hmac = _model_binding(model_bytes, hmac_key)
    snapshot_payload = _snapshot_payload(
        new_version,
        history,
        model_sha,
        model_hmac,
        active,
        created_at,
        _predict_snapshot_values(history, model_bytes.decode("utf-8"), active),
    )
    snapshot_document = _artifact_document(snapshot_payload, hmac_key)
    anchor_payload = _anchor_payload(snapshot_document)
    summary, blocks = _summarize([*observations, observation_document], active)
    formal_activated, reasons = evaluate_formal_gate(summary, blocks)
    state_payload = {
        "schemaVersion": SCHEMA_VERSION,
        "version": new_version,
        "profile": active.profile,
        "formalEligible": active.formal_eligible,
        "observed": new_version,
        "latestIssue": str(row["issue"]),
        "dataSha256": canonical_kl8_sha256(history),
        "protocolArtifactSha256": protocol["artifactSha256"],
        "lineageArtifactSha256": load_and_verify_artifact(
            root / "lineage.json", hmac_key
        )["artifactSha256"],
        "previousStateArtifactSha256": state["artifactSha256"],
        "previousModelSha256": state["modelSha256"],
        "observationArtifactSha256": observation_document["artifactSha256"],
        "currentSnapshotArtifactSha256": snapshot_document["artifactSha256"],
        "modelSha256": model_sha,
        "modelHmacSha256": model_hmac,
        "formalPredictionActivated": formal_activated,
        "gateReasons": reasons,
    }
    _publish_version(
        root,
        new_version,
        {
            "model.txt": model_bytes,
            "snapshot.json": snapshot_payload,
            "anchor_payload.json": anchor_payload,
            "observation.json": observation_payload,
            "state.json": state_payload,
        },
        hmac_key,
    )
    return {
        "status": "updated",
        "version": new_version,
        "observed": new_version,
        "targetIssue": new_target_issue,
        "targetDate": new_target_date,
        "formalPredictionActivated": formal_activated,
        "externalAnchorPayload": anchor_payload,
    }


def prospective_status(
    state_dir: str | Path,
    *,
    hmac_key: bytes,
    canonical_csv: str | Path | None = None,
    config: Kl8Pick4ProspectiveConfig | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, object]:
    """遍历验证0..latest完整链，并返回准入状态。"""

    active = config or Kl8Pick4ProspectiveConfig()
    _validate_config(active)
    root = Path(state_dir)
    if canonical_csv is None:
        raise ValueError("status必须提供canonical CSV以逐期重导observation")
    history = load_full_canonical_csv(canonical_csv)
    state, snapshot, observations = _verify_chain(root, hmac_key, history, active)
    summary, blocks = _summarize(observations, active)
    activated, reasons = evaluate_formal_gate(summary, blocks)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "profile": active.profile,
        "formalEligible": active.formal_eligible,
        "version": state["version"],
        "observed": len(observations),
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "summary": summary,
        "stabilityBlocks": blocks,
        "formalPredictionActivated": activated,
        "gateReasons": reasons,
    }
    if report_dir is not None:
        version = _integer(state["version"], "状态版本")
        report_path = Path(report_dir) / f"status_{version:04d}.json"
        if report_path.exists():
            raise FileExistsError(f"状态报告不可覆盖：{report_path}")
        _write_artifact(report_path, payload, hmac_key)
    return payload


__all__ = [
    "CHECKPOINTS",
    "CONSUMED_BOUNDARY",
    "FORMAL_CANONICAL_CONFIG",
    "Kl8Pick4ProspectiveConfig",
    "Kl8Pick4RankConfig",
    "build_next_issue_features",
    "evaluate_formal_gate",
    "load_and_verify_artifact",
    "load_full_canonical_csv",
    "payload_sha256",
    "prospective_status",
    "register_prospective",
    "update_prospective",
]
