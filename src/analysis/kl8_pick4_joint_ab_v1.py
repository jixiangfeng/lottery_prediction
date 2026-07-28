# -*- coding: utf-8 -*-
"""快乐8 Pick4 同成本联合概率挑战器的 500 期 A/B 前瞻协议。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import binomtest

from src.analysis import kl8_pick4_prospective as v2_prospective
from src.analysis.kl8_pick4_joint_portfolio_v1 import (
    ConditionalPoisson20,
    best_improvement_partition,
    canonical_portfolio,
    evaluation_to_dict,
)
from src.analysis.kl8_pick4_rank_challenger import (
    PICK4_RANK_FEATURES,
    audit_rank_probabilities,
    ranked_pick4_portfolio,
)
from src.analysis.kl8_pick5_probability_v1 import canonical_kl8_sha256

SCHEMA_VERSION = "kl8_pick4_joint_ab_v1"
REQUIRED_DRAWS = 500
BLOCK_SIZE = 100
BLOCKS = 5
RANDOM_BENCHMARKS = 32
SHANGHAI = ZoneInfo("Asia/Shanghai")
ProgressCallback = Callable[[dict[str, object]], None]
FloatArray = NDArray[np.float64]


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    """返回稳定 JSON payload 的 SHA-256。"""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label}必须为整数")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label}必须为数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}必须为有限数值")
    return result


def _artifact_hmac(payload: Mapping[str, object], sha256: str, key: bytes) -> str:
    return hmac.new(
        key, _canonical_bytes(payload) + b"\n" + sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()


def _artifact_document(payload: Mapping[str, object], key: bytes) -> dict[str, object]:
    unsigned = dict(payload)
    sha256 = payload_sha256(unsigned)
    return {
        **unsigned,
        "artifactSha256": sha256,
        "artifactHmacSha256": _artifact_hmac(unsigned, sha256, key),
    }


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(
    path: Path, payload: Mapping[str, object], key: bytes
) -> dict[str, object]:
    document = _artifact_document(payload, key)
    _write_bytes(
        path,
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        + b"\n",
    )
    return document


def load_and_verify_artifact(path: str | Path, hmac_key: bytes) -> dict[str, object]:
    """加载 JSON artifact，并验证稳定哈希及 HMAC-SHA256。"""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact不可读取：{path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"artifact必须是JSON对象：{path}")
    sha256 = document.get("artifactSha256")
    signature = document.get("artifactHmacSha256")
    unsigned = {
        key: value
        for key, value in document.items()
        if key not in {"artifactSha256", "artifactHmacSha256"}
    }
    actual_sha = payload_sha256(unsigned)
    if not isinstance(sha256, str) or sha256 != actual_sha:
        raise ValueError(f"artifact SHA-256不匹配：{path}")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, _artifact_hmac(unsigned, actual_sha, hmac_key)
    ):
        raise ValueError(f"artifact HMAC-SHA256不匹配：{path}")
    return cast(dict[str, object], document)


def _model_binding(content: bytes, key: bytes) -> tuple[str, str]:
    sha256 = _bytes_sha256(content)
    signature = hmac.new(
        key, content + b"\n" + sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return sha256, signature


def _verify_model(
    path: Path, expected_sha: object, expected_hmac: object, key: bytes
) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"锁定模型缺失：{path}") from error
    sha256, signature = _model_binding(content, key)
    if expected_sha != sha256 or expected_hmac != signature:
        raise ValueError(f"锁定模型SHA/HMAC不匹配：{path}")
    return content.decode("utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_hashes() -> dict[str, str]:
    root = _project_root()
    paths = (
        "src/analysis/kl8_pick4_joint_ab_v1.py",
        "src/analysis/kl8_pick4_joint_portfolio_v1.py",
        "src/analysis/kl8_pick4_prospective.py",
        "src/analysis/kl8_pick4_rank_challenger.py",
        "src/analysis/kl8_feature_discovery_v2.py",
        "src/analysis/kl8_pick5_probability_v1.py",
    )
    result: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"协议源码缺失：{relative}")
        result[relative] = _bytes_sha256(path.read_bytes())
    return result


def _protocol_payload(initial_history: pd.DataFrame) -> dict[str, object]:
    sources = _source_hashes()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "researchOnly": True,
        "requiredDraws": REQUIRED_DRAWS,
        "blockSize": BLOCK_SIZE,
        "blocks": BLOCKS,
        "randomSameCostBenchmarks": RANDOM_BENCHMARKS,
        "controlDefinition": "current v2 scores + ranked_pick4_portfolio(scores)",
        "challengerDefinition": "same Top20 union; deterministic best-improvement pair swaps",
        "objectiveOrder": ["sumExact4", "sumAtLeast3", "sumAtLeast2"],
        "realizedPayoutSurrogate": {"exact2": 1, "exact3": 2, "exact4": 3},
        "optimizerUsesRealizedPayout": False,
        "jointMarginalPreservationClaim": False,
        "initialHistoryRows": len(initial_history),
        "initialDataSha256": canonical_kl8_sha256(initial_history),
        "sourceSha256": sources,
        "codeSha256": payload_sha256(sources),
        "formalCandidates": [],
        "productionActivation": False,
        "noTuningRetryReset": True,
    }


def _created_at(now: datetime | None = None) -> str:
    value = now or datetime.now(tz=SHANGHAI)
    if value.tzinfo is None:
        raise ValueError("createdAt必须包含时区")
    return value.astimezone(SHANGHAI).isoformat()


def _ensure_predraw(created_at: str, target_date: str) -> None:
    created = datetime.fromisoformat(created_at).astimezone(SHANGHAI)
    deadline = datetime.fromisoformat(f"{target_date}T21:30:00+08:00")
    if created >= deadline:
        raise ValueError("初始化/快照创建必须早于目标期开奖时间")


def _ranking_payload(scores: FloatArray) -> list[dict[str, object]]:
    order = np.lexsort((np.arange(80), -scores))
    return [
        {"rank": rank + 1, "number": int(index + 1), "score": float(scores[index])}
        for rank, index in enumerate(order)
    ]


def _random_benchmark_portfolios(
    union: Sequence[int], seed_material: str
) -> list[list[list[int]]]:
    generator = random.Random(int(seed_material[:16], 16))
    canonical_union = sorted(int(value) for value in union)
    portfolios: list[list[list[int]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    while len(portfolios) < RANDOM_BENCHMARKS:
        shuffled = canonical_union.copy()
        generator.shuffle(shuffled)
        portfolio = canonical_portfolio(
            [shuffled[index : index + 4] for index in range(0, 20, 4)]
        )
        if portfolio in seen:
            continue
        seen.add(portfolio)
        portfolios.append([list(ticket) for ticket in portfolio])
    return portfolios


def _build_snapshot_prediction(
    history: pd.DataFrame,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, dict[str, object]]:
    config = v2_prospective.Kl8Pick4ProspectiveConfig()
    model_text = v2_prospective._fit_model(history, config)
    features = v2_prospective.build_next_issue_features(history)
    scores = np.asarray(
        lgb.Booster(model_str=model_text).predict(features.loc[:, PICK4_RANK_FEATURES]),
        dtype=np.float64,
    )
    if scores.shape != (80,) or not np.isfinite(scores).all():
        raise ValueError("v2预测分数必须是80维有限向量")
    probabilities = audit_rank_probabilities(scores, config.rank_config)
    control_tickets = ranked_pick4_portfolio(scores)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "portfolioOptimizationStarted",
                "historyRows": len(history),
                "controlTickets": control_tickets,
            }
        )
    control, challenger = best_improvement_partition(
        probabilities.tolist(),
        control_tickets,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "portfolioOptimizationCompleted",
                "historyRows": len(history),
                "controlObjective": list(control.objective),
                "challengerObjective": list(challenger.objective),
            }
        )
    union = sorted(number for ticket in control.tickets for number in ticket)
    delta = [
        challenger.objective[index] - control.objective[index] for index in range(3)
    ]
    prediction = {
        "scoreRanking": _ranking_payload(scores),
        "probabilities80": [float(value) for value in probabilities],
        "auditedMarginals80": [float(value) for value in probabilities],
        "jointModelMarginalPreservationClaim": False,
        "top20Union": union,
        "controlA": evaluation_to_dict(control),
        "challengerB": evaluation_to_dict(challenger),
        "objectiveDeltaBMinusA": {
            "sumExact4": delta[0],
            "sumAtLeast3": delta[1],
            "sumAtLeast2": delta[2],
        },
        "objectiveIsPayoutSurrogateOnly": True,
    }
    prediction["randomSameCostPortfolios"] = _random_benchmark_portfolios(
        union,
        payload_sha256(
            {"data": canonical_kl8_sha256(history), "prediction": prediction}
        ),
    )
    return model_text, prediction


def _snapshot_payload(
    version: int,
    history: pd.DataFrame,
    protocol: Mapping[str, object],
    model_text: str,
    prediction: Mapping[str, object],
    key: bytes,
    previous_snapshot_sha: str | None,
    now: datetime | None,
) -> dict[str, object]:
    target_issue, target_date = v2_prospective._next_target(history)
    created_at = _created_at(now)
    _ensure_predraw(created_at, target_date)
    model_sha, model_hmac = _model_binding(model_text.encode("utf-8"), key)
    latest = history.iloc[-1]
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "researchOnly": True,
        "afterIssue": str(latest["issue"]),
        "targetIssue": target_issue,
        "targetDate": target_date,
        "createdAt": created_at,
        "historyRows": len(history),
        "latestVisibleIssue": str(latest["issue"]),
        "latestVisibleResult": [int(value) for value in latest["numbers"]],
        "dataSha256": canonical_kl8_sha256(history),
        "sourceSha256": protocol["sourceSha256"],
        "codeSha256": protocol["codeSha256"],
        "protocolArtifactSha256": protocol["artifactSha256"],
        "modelSha256": model_sha,
        "modelHmacSha256": model_hmac,
        "previousSnapshotArtifactSha256": previous_snapshot_sha,
        "formalCandidates": [],
        "productionActivation": False,
        **prediction,
    }
    _validate_snapshot(payload)
    return payload


def _objective_tuple(arm: Mapping[str, object]) -> tuple[float, float, float]:
    objective = cast(Mapping[str, object], arm["objective"])
    return (
        _number(objective["sumExact4"], "sumExact4"),
        _number(objective["sumAtLeast3"], "sumAtLeast3"),
        _number(objective["sumAtLeast2"], "sumAtLeast2"),
    )


def _validate_snapshot(snapshot: Mapping[str, object]) -> None:
    if (
        snapshot.get("formalCandidates") != []
        or snapshot.get("productionActivation") is not False
    ):
        raise ValueError("A/B快照不得包含正式候选或生产激活")
    ranking = cast(list[Mapping[str, object]], snapshot.get("scoreRanking"))
    if not isinstance(ranking, list) or len(ranking) != 80:
        raise ValueError("scoreRanking必须完整覆盖80个号码")
    numbers = [_integer(item["number"], "ranking number") for item in ranking]
    if set(numbers) != set(range(1, 81)) or [
        _integer(item["rank"], "ranking rank") for item in ranking
    ] != list(range(1, 81)):
        raise ValueError("scoreRanking号码或名次无效")
    scores = np.empty(80, dtype=np.float64)
    for item in ranking:
        scores[_integer(item["number"], "ranking number") - 1] = _number(
            item["score"], "ranking score"
        )
    probabilities = cast(Sequence[float], snapshot["probabilities80"])
    distribution = ConditionalPoisson20(probabilities)
    control_raw = cast(Mapping[str, object], snapshot["controlA"])
    challenger_raw = cast(Mapping[str, object], snapshot["challengerB"])
    control = canonical_portfolio(cast(Sequence[Sequence[int]], control_raw["tickets"]))
    challenger = canonical_portfolio(
        cast(Sequence[Sequence[int]], challenger_raw["tickets"])
    )
    expected_control = canonical_portfolio(ranked_pick4_portfolio(scores))
    if control != expected_control:
        raise ValueError("控制A必须精确复用v2 ranked_pick4_portfolio(scores)")
    control_union = sorted(number for ticket in control for number in ticket)
    challenger_union = sorted(number for ticket in challenger for number in ticket)
    if control_union != challenger_union or control_union != snapshot.get("top20Union"):
        raise ValueError("A/B必须使用完全相同的Top20并集")
    for label, raw, tickets in (
        ("A", control_raw, control),
        ("B", challenger_raw, challenger),
    ):
        expected = evaluation_to_dict(distribution.evaluate_portfolio(tickets))
        if raw != expected:
            raise ValueError(f"{label}票组PMF或目标值无法精确重导")
    if _objective_tuple(challenger_raw) < _objective_tuple(control_raw):
        raise ValueError("挑战B联合目标不得劣于控制A")
    benchmarks = snapshot.get("randomSameCostPortfolios")
    if not isinstance(benchmarks, list) or len(benchmarks) != RANDOM_BENCHMARKS:
        raise ValueError("随机同成本基准数量不符合固定协议")
    for portfolio in benchmarks:
        if (
            sorted(
                number
                for ticket in canonical_portfolio(
                    cast(Sequence[Sequence[int]], portfolio)
                )
                for number in ticket
            )
            != control_union
        ):
            raise ValueError("随机基准必须保持相同Top20并集与成本")


def _observation_payload(
    snapshot: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, object]:
    drawn = {int(value) for value in cast(Sequence[int], row["numbers"])}

    def arm_result(arm_name: str) -> dict[str, object]:
        arm = cast(Mapping[str, object], snapshot[arm_name])
        tickets = canonical_portfolio(cast(Sequence[Sequence[int]], arm["tickets"]))
        hits = [len(set(ticket) & drawn) for ticket in tickets]
        return {
            "ticketHits": hits,
            "bestTicketAtLeast2": max(hits) >= 2,
            "bestTicketAtLeast3": max(hits) >= 3,
            "bestTicketExact4": max(hits) == 4,
            "exact4TicketCount": hits.count(4),
            "totalHits": sum(hits),
            "payoutSurrogate": sum(
                1 if hit == 2 else 2 if hit == 3 else 3 if hit == 4 else 0
                for hit in hits
            ),
        }

    arm_a = arm_result("controlA")
    arm_b = arm_result("challengerB")
    if arm_a["totalHits"] != arm_b["totalHits"]:
        raise RuntimeError("相同Top20并集的A/B总命中数必须相等")
    benchmark_scores: list[int] = []
    for portfolio in cast(
        list[Sequence[Sequence[int]]], snapshot["randomSameCostPortfolios"]
    ):
        hits = [len(set(ticket) & drawn) for ticket in canonical_portfolio(portfolio)]
        benchmark_scores.append(
            sum(
                1 if hit == 2 else 2 if hit == 3 else 3 if hit == 4 else 0
                for hit in hits
            )
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "snapshotArtifactSha256": snapshot["artifactSha256"],
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "drawNumbers": sorted(drawn),
        "armA": arm_a,
        "armB": arm_b,
        "pairedPayoutDifferenceBMinusA": _integer(
            arm_b["payoutSurrogate"], "B payoutSurrogate"
        )
        - _integer(arm_a["payoutSurrogate"], "A payoutSurrogate"),
        "randomSameCostPayoutSurrogates": benchmark_scores,
        "formalCandidates": [],
        "productionActivation": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_version(
    root: Path,
    version: int,
    files: Mapping[str, bytes | Mapping[str, object]],
    key: bytes,
) -> None:
    versions = root / "versions"
    target = versions / f"{version:04d}"
    if target.exists():
        raise FileExistsError(f"版本不可覆盖：{target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{version:04d}-", dir=versions))
    try:
        manifest: dict[str, str] = {}
        for name, content in files.items():
            if isinstance(content, bytes):
                _write_bytes(staging / name, content)
            else:
                _write_json(staging / name, content, key)
            manifest[name] = _bytes_sha256((staging / name).read_bytes())
        commit = {
            "schemaVersion": SCHEMA_VERSION,
            "version": version,
            "manifestBytesSha256": manifest,
        }
        _write_json(staging / "commit.json", commit, key)
        _fsync_directory(staging)
        os.rename(staging, target)
        _fsync_directory(versions)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _version_numbers(root: Path) -> list[int]:
    versions = root / "versions"
    if not versions.is_dir():
        return []
    if any(path.name.startswith(".") for path in versions.iterdir()):
        raise ValueError("检测到未完成的版本暂存目录")
    numbers = sorted(
        int(path.name)
        for path in versions.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if numbers != list(range(len(numbers))):
        raise ValueError("版本目录必须从0000开始连续且只追加")
    return numbers


def _verify_commit(version_dir: Path, key: bytes) -> None:
    commit = load_and_verify_artifact(version_dir / "commit.json", key)
    manifest = cast(Mapping[str, object], commit.get("manifestBytesSha256"))
    actual_names = {
        path.name for path in version_dir.iterdir() if path.name != "commit.json"
    }
    if set(manifest) != actual_names:
        raise ValueError(f"版本manifest文件集合不匹配：{version_dir}")
    for name, expected in manifest.items():
        if _bytes_sha256((version_dir / name).read_bytes()) != expected:
            raise ValueError(f"版本文件字节哈希不匹配：{version_dir / name}")


def _verify_chain(
    root: Path,
    key: bytes,
    history: pd.DataFrame,
) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    list[dict[str, object]],
    dict[str, object],
]:
    protocol = load_and_verify_artifact(root / "protocol.json", key)
    if (
        protocol.get("schemaVersion") != SCHEMA_VERSION
        or protocol.get("requiredDraws") != REQUIRED_DRAWS
    ):
        raise ValueError("A/B协议schema或固定500期配置不匹配")
    if protocol.get("sourceSha256") != _source_hashes() or protocol.get(
        "codeSha256"
    ) != payload_sha256(_source_hashes()):
        raise ValueError("A/B协议源码哈希与当前代码不一致")
    initial_rows = _integer(protocol["initialHistoryRows"], "initialHistoryRows")
    if (
        len(history) < initial_rows
        or canonical_kl8_sha256(history.iloc[:initial_rows].reset_index(drop=True))
        != protocol["initialDataSha256"]
    ):
        raise ValueError("初始化canonical历史前缀已变化")
    versions = _version_numbers(root)
    if not versions:
        raise ValueError("A/B状态没有已提交版本")
    if len(history) < initial_rows + versions[-1]:
        raise ValueError("canonical历史少于已结算A/B状态，禁止回退")
    observations: list[dict[str, object]] = []
    previous_state_sha: str | None = None
    previous_snapshot_sha: str | None = None
    current_snapshot: dict[str, object] | None = None
    state: dict[str, object] = {}
    for version in versions:
        version_dir = root / "versions" / f"{version:04d}"
        _verify_commit(version_dir, key)
        state = load_and_verify_artifact(version_dir / "state.json", key)
        if state.get("version") != version or state.get("observed") != version:
            raise ValueError("state版本/observed必须与目录版本一致")
        if state.get("previousStateArtifactSha256") != previous_state_sha:
            raise ValueError("previous state哈希链断裂")
        if (
            state.get("formalCandidates") != []
            or state.get("productionActivation") is not False
        ):
            raise ValueError("A/B状态不得产生正式候选或生产激活")
        if version > 0:
            observation = load_and_verify_artifact(
                version_dir / "observation.json", key
            )
            expected_row = history.iloc[initial_rows + version - 1]
            previous_snapshot = load_and_verify_artifact(
                root / "versions" / f"{version - 1:04d}" / "snapshot.json", key
            )
            expected_observation = _observation_payload(
                previous_snapshot, cast(Mapping[str, object], expected_row.to_dict())
            )
            unsigned = {
                key_: value
                for key_, value in observation.items()
                if key_ not in {"artifactSha256", "artifactHmacSha256"}
            }
            if unsigned != expected_observation:
                raise ValueError("observation未使用锁定票组或canonical目标期开奖")
            if state.get("observationArtifactSha256") != observation["artifactSha256"]:
                raise ValueError("state未绑定本期observation")
            observations.append(observation)
        if version < REQUIRED_DRAWS:
            current_snapshot = load_and_verify_artifact(
                version_dir / "snapshot.json", key
            )
            prefix = history.iloc[: initial_rows + version].reset_index(drop=True)
            if current_snapshot.get("historyRows") != len(
                prefix
            ) or current_snapshot.get("dataSha256") != canonical_kl8_sha256(prefix):
                raise ValueError("快照包含未来数据或历史前缀不匹配")
            if (
                current_snapshot.get("protocolArtifactSha256")
                != protocol["artifactSha256"]
            ):
                raise ValueError("快照未绑定固定协议")
            if (
                current_snapshot.get("previousSnapshotArtifactSha256")
                != previous_snapshot_sha
            ):
                raise ValueError("previous snapshot哈希链断裂")
            _verify_model(
                version_dir / "model.txt",
                current_snapshot["modelSha256"],
                current_snapshot["modelHmacSha256"],
                key,
            )
            _validate_snapshot(current_snapshot)
            if (
                state.get("currentSnapshotArtifactSha256")
                != current_snapshot["artifactSha256"]
            ):
                raise ValueError("state未绑定当前快照")
            previous_snapshot_sha = cast(str, current_snapshot["artifactSha256"])
        else:
            current_snapshot = None
            if "snapshot.json" in {path.name for path in version_dir.iterdir()}:
                raise ValueError("第500期结案版本不得生成第501期快照")
        previous_state_sha = cast(str, state["artifactSha256"])
    if len(observations) != _integer(state["observed"], "state observed"):
        raise ValueError("observation数量与state不一致")
    return state, current_snapshot, observations, protocol


def initialize_joint_ab(
    csv_path: str | Path,
    state_dir: str | Path,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """在目标期开奖前创建唯一的第0版研究快照。"""

    history = v2_prospective.load_full_canonical_csv(csv_path)
    if history.empty:
        raise ValueError("初始化至少需要1期canonical历史")
    root = Path(state_dir)
    if root.exists():
        raise FileExistsError(f"A/B state已存在，禁止重置或重试：{root}")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=parent))
    try:
        (staging / "versions").mkdir(mode=0o700)
        protocol = _write_json(
            staging / "protocol.json", _protocol_payload(history), hmac_key
        )
        model_text, prediction = _build_snapshot_prediction(
            history, progress_callback=progress_callback
        )
        snapshot_payload = _snapshot_payload(
            0, history, protocol, model_text, prediction, hmac_key, None, now
        )
        model_sha, model_hmac = _model_binding(model_text.encode("utf-8"), hmac_key)
        state_payload = {
            "schemaVersion": SCHEMA_VERSION,
            "version": 0,
            "observed": 0,
            "latestSettledIssue": None,
            "protocolArtifactSha256": protocol["artifactSha256"],
            "previousStateArtifactSha256": None,
            "observationArtifactSha256": None,
            "currentSnapshotArtifactSha256": _artifact_document(
                snapshot_payload, hmac_key
            )["artifactSha256"],
            "modelSha256": model_sha,
            "modelHmacSha256": model_hmac,
            "reviewStatus": "research-only-observed-0-not-evidence",
            "formalCandidates": [],
            "productionActivation": False,
        }
        _publish_version(
            staging,
            0,
            {
                "model.txt": model_text.encode("utf-8"),
                "snapshot.json": snapshot_payload,
                "state.json": state_payload,
            },
            hmac_key,
        )
        _fsync_directory(staging)
        os.rename(staging, root)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    snapshot = load_and_verify_artifact(root / "versions/0000/snapshot.json", hmac_key)
    return {
        "status": "initialized",
        "observed": 0,
        "targetIssue": snapshot["targetIssue"],
        "targetDate": snapshot["targetDate"],
        "researchOnly": True,
        "formalCandidates": [],
        "productionActivation": False,
    }


def step_joint_ab(
    csv_path: str | Path,
    state_dir: str | Path,
    *,
    hmac_key: bytes,
    now: datetime | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """最多结算一个已锁定目标，并在未满500期时生成唯一下一快照。"""

    history = v2_prospective.load_full_canonical_csv(csv_path)
    root = Path(state_dir)
    state, snapshot, observations, protocol = _verify_chain(root, hmac_key, history)
    observed = _integer(state["observed"], "state observed")
    initial_rows = _integer(protocol["initialHistoryRows"], "initialHistoryRows")
    if observed == REQUIRED_DRAWS:
        return {
            "status": "complete",
            "observed": observed,
            "reviewStatus": "pending-human-review",
        }
    expected_rows = initial_rows + observed
    if len(history) == expected_rows:
        if snapshot is None:
            raise RuntimeError("未满500期时必须存在锁定快照")
        return {
            "status": "unavailable",
            "observed": observed,
            "targetIssue": snapshot["targetIssue"],
        }
    if len(history) > expected_rows + 1:
        raise ValueError("检测到未逐期快照锁定的开奖，拒绝backfill/catch-up")
    if snapshot is None:
        raise RuntimeError("当前目标快照缺失")
    row = history.iloc[-1]
    if (
        str(row["issue"]) != snapshot["targetIssue"]
        or str(row["date"]) != snapshot["targetDate"]
    ):
        raise ValueError("新增canonical开奖必须精确匹配锁定targetIssue/targetDate")
    observation_payload = _observation_payload(
        snapshot, cast(Mapping[str, object], row.to_dict())
    )
    new_version = observed + 1
    files: dict[str, bytes | Mapping[str, object]] = {
        "observation.json": observation_payload
    }
    next_snapshot_sha: str | None = None
    model_sha: str | None = None
    model_hmac: str | None = None
    if new_version < REQUIRED_DRAWS:
        model_text, prediction = _build_snapshot_prediction(
            history, progress_callback=progress_callback
        )
        next_snapshot = _snapshot_payload(
            new_version,
            history,
            protocol,
            model_text,
            prediction,
            hmac_key,
            cast(str, snapshot["artifactSha256"]),
            now,
        )
        next_snapshot_sha = cast(
            str, _artifact_document(next_snapshot, hmac_key)["artifactSha256"]
        )
        model_sha, model_hmac = _model_binding(model_text.encode("utf-8"), hmac_key)
        files.update(
            {"model.txt": model_text.encode("utf-8"), "snapshot.json": next_snapshot}
        )
    observation_sha = cast(
        str, _artifact_document(observation_payload, hmac_key)["artifactSha256"]
    )
    state_payload = {
        "schemaVersion": SCHEMA_VERSION,
        "version": new_version,
        "observed": new_version,
        "latestSettledIssue": str(row["issue"]),
        "protocolArtifactSha256": protocol["artifactSha256"],
        "previousStateArtifactSha256": state["artifactSha256"],
        "observationArtifactSha256": observation_sha,
        "currentSnapshotArtifactSha256": next_snapshot_sha,
        "modelSha256": model_sha,
        "modelHmacSha256": model_hmac,
        "reviewStatus": (
            "pending-human-review"
            if new_version == REQUIRED_DRAWS
            else "research-only-running"
        ),
        "formalCandidates": [],
        "productionActivation": False,
    }
    files["state.json"] = state_payload
    _publish_version(root, new_version, files, hmac_key)
    return {
        "status": "settled" if new_version < REQUIRED_DRAWS else "complete",
        "observed": new_version,
        "targetIssue": (
            None
            if new_version == REQUIRED_DRAWS
            else cast(Mapping[str, object], files["snapshot.json"])["targetIssue"]
        ),
        "reviewStatus": state_payload["reviewStatus"],
        "formalCandidates": [],
        "productionActivation": False,
    }


def _binary_paired(
    records: Sequence[Mapping[str, object]], key: str
) -> dict[str, object]:
    b_wins = sum(
        bool(cast(Mapping[str, object], item["armB"])[key])
        and not bool(cast(Mapping[str, object], item["armA"])[key])
        for item in records
    )
    a_wins = sum(
        bool(cast(Mapping[str, object], item["armA"])[key])
        and not bool(cast(Mapping[str, object], item["armB"])[key])
        for item in records
    )
    discordant = a_wins + b_wins
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(b_wins, discordant, 0.5, alternative="greater").pvalue)
    )
    return {
        "bWins": b_wins,
        "aWins": a_wins,
        "discordant": discordant,
        "exactPValue": p_value,
    }


def _sign_test(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    differences = [
        _integer(item["pairedPayoutDifferenceBMinusA"], "paired payout difference")
        for item in records
    ]
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    nonzero = positive + negative
    p_value = (
        1.0
        if nonzero == 0
        else float(binomtest(positive, nonzero, 0.5, alternative="greater").pvalue)
    )
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - nonzero,
        "exactPValue": p_value,
    }


def _holm(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def _rates(records: Sequence[Mapping[str, object]], arm: str) -> dict[str, float]:
    if not records:
        return {
            "atLeast2": 0.0,
            "atLeast3": 0.0,
            "exact4": 0.0,
            "meanExact4Tickets": 0.0,
            "meanPayoutSurrogate": 0.0,
        }
    values = [cast(Mapping[str, object], item[arm]) for item in records]
    count = len(values)
    return {
        "atLeast2": sum(bool(item["bestTicketAtLeast2"]) for item in values) / count,
        "atLeast3": sum(bool(item["bestTicketAtLeast3"]) for item in values) / count,
        "exact4": sum(bool(item["bestTicketExact4"]) for item in values) / count,
        "meanExact4Tickets": math.fsum(
            _number(item["exact4TicketCount"], "exact4TicketCount") for item in values
        )
        / count,
        "meanPayoutSurrogate": math.fsum(
            _number(item["payoutSurrogate"], "payoutSurrogate") for item in values
        )
        / count,
    }


def _paired_differences(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    rates_a = _rates(records, "armA")
    rates_b = _rates(records, "armB")
    return {
        key: rates_b[key] - rates_a[key]
        for key in ("atLeast2", "atLeast3", "exact4", "meanPayoutSurrogate")
    }


def _blocks(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(BLOCKS):
        subset = records[index * BLOCK_SIZE : (index + 1) * BLOCK_SIZE]
        result.append(
            {
                "block": index + 1,
                "drawRange": [index * BLOCK_SIZE + 1, (index + 1) * BLOCK_SIZE],
                "observed": len(subset),
                "complete": len(subset) == BLOCK_SIZE,
                "pairedDifferences": _paired_differences(subset),
            }
        )
    return result


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    binary = {
        "atLeast2": _binary_paired(records, "bestTicketAtLeast2"),
        "atLeast3": _binary_paired(records, "bestTicketAtLeast3"),
        "exact4": _binary_paired(records, "bestTicketExact4"),
    }
    sign = _sign_test(records)
    raw = {
        name: _number(value["exactPValue"], f"{name} exactPValue")
        for name, value in binary.items()
    }
    raw["payoutSurrogate"] = _number(sign["exactPValue"], "payoutSurrogate exactPValue")
    benchmark_values = [
        value
        for record in records
        for value in cast(Sequence[int], record["randomSameCostPayoutSurrogates"])
    ]
    return {
        "observed": len(records),
        "armA": _rates(records, "armA"),
        "armB": _rates(records, "armB"),
        "pairedDifferencesBMinusA": _paired_differences(records),
        "mcnemarExact": binary,
        "payoutSignExact": sign,
        "holmAdjustedPairedPValues": _holm(raw),
        "randomSameCostBenchmarks": {
            "portfoliosPerDraw": RANDOM_BENCHMARKS,
            "observations": len(benchmark_values),
            "meanPayoutSurrogate": (
                0.0
                if not benchmark_values
                else math.fsum(benchmark_values) / len(benchmark_values)
            ),
        },
    }


def _gate(
    summary: Mapping[str, object], blocks: Sequence[Mapping[str, object]]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["observed"] != REQUIRED_DRAWS:
        reasons.append("requires_exactly_500_settled_draws")
        return False, reasons
    differences = cast(Mapping[str, float], summary["pairedDifferencesBMinusA"])
    if differences["exact4"] <= 0.0:
        reasons.append("exact4_not_improved")
    if differences["atLeast3"] <= 0.0:
        reasons.append("atLeast3_not_improved")
    if differences["atLeast2"] < 0.0:
        reasons.append("atLeast2_worsened")
    if differences["meanPayoutSurrogate"] <= 0.0:
        reasons.append("paired_payout_surrogate_not_positive")
    adjusted = cast(Mapping[str, float], summary["holmAdjustedPairedPValues"])
    for name in ("exact4", "atLeast3", "atLeast2", "payoutSurrogate"):
        if adjusted[name] > 0.05:
            reasons.append(f"holm_{name}_not_significant")
    for block in blocks:
        if block["complete"] is not True:
            reasons.append(f"block_{block['block']}_incomplete")
            continue
        block_differences = cast(Mapping[str, float], block["pairedDifferences"])
        if any(
            block_differences[name] < 0.0
            for name in ("exact4", "atLeast3", "atLeast2", "meanPayoutSurrogate")
        ):
            reasons.append(f"block_{block['block']}_negative")
    return not reasons, reasons


def joint_ab_status(
    csv_path: str | Path,
    state_dir: str | Path,
    *,
    hmac_key: bytes,
) -> dict[str, object]:
    """验证完整链并汇总固定 A/B 指标；永不自动激活生产。"""

    history = v2_prospective.load_full_canonical_csv(csv_path)
    state, snapshot, observations, protocol = _verify_chain(
        Path(state_dir), hmac_key, history
    )
    observed = _integer(state["observed"], "state observed")
    initial_rows = _integer(protocol["initialHistoryRows"], "initialHistoryRows")
    if observed < REQUIRED_DRAWS and len(history) > initial_rows + observed + 1:
        raise ValueError("检测到超过一个未处理开奖，拒绝catch-up状态聚合")
    summary = _summary(observations)
    blocks = _blocks(observations)
    gates_passed, gate_reasons = _gate(summary, blocks)
    target_available = False
    if snapshot is not None and len(history) == initial_rows + observed + 1:
        row = history.iloc[-1]
        target_available = (
            str(row["issue"]) == snapshot["targetIssue"]
            and str(row["date"]) == snapshot["targetDate"]
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "researchOnly": True,
        "observed": observed,
        "requiredDraws": REQUIRED_DRAWS,
        "targetIssue": None if snapshot is None else snapshot["targetIssue"],
        "targetAvailable": target_available,
        "summary": summary,
        "fixed100DrawBlocks": blocks,
        "predeclaredGatesPassed": gates_passed,
        "gateReasons": gate_reasons,
        "reviewStatus": (
            "pending-human-review"
            if observed == REQUIRED_DRAWS
            else "research-only-running"
        ),
        "formalCandidates": [],
        "productionActivation": False,
    }


__all__ = [
    "BLOCKS",
    "BLOCK_SIZE",
    "RANDOM_BENCHMARKS",
    "REQUIRED_DRAWS",
    "initialize_joint_ab",
    "joint_ab_status",
    "load_and_verify_artifact",
    "payload_sha256",
    "step_joint_ab",
]
