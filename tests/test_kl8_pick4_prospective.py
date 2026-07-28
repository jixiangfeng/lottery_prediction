# -*- coding: utf-8 -*-
"""KL8 Pick4 前瞻协议 v2 的攻击回归测试。"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.analysis import kl8_pick4_prospective as prospective
from src.analysis.kl8_pick4_prospective import (
    Kl8Pick4ProspectiveConfig,
    evaluate_formal_gate,
    load_and_verify_artifact,
    prospective_status,
    register_prospective,
    update_prospective,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _history(periods: int = 12, *, start: datetime | None = None) -> pd.DataFrame:
    first = start or datetime(2026, 1, 1, tzinfo=SHANGHAI)
    rows: list[dict[str, Any]] = []
    for index in range(periods):
        date = (first + timedelta(days=index)).date()
        issue = f"{date.year}{index + 1:03d}"
        numbers = sorted((((index * 7 + offset * 3) % 80) + 1) for offset in range(20))
        rows.append({"issue": issue, "date": date.isoformat(), "numbers": numbers})
    return pd.DataFrame(rows)


def _write_csv(path: Path, history: pd.DataFrame) -> None:
    serial = history.copy()
    serial["numbers"] = serial["numbers"].map(lambda values: " ".join(map(str, values)))
    serial.to_csv(path, index=False)


def _key(tmp_path: Path) -> tuple[Path, bytes]:
    path = tmp_path / "outside.key"
    key = b"v2-test-key-material-must-stay-secret-0123456789"
    path.write_bytes(key)
    path.chmod(0o600)
    return path, key


@pytest.fixture
def smoke_config() -> Kl8Pick4ProspectiveConfig:
    return Kl8Pick4ProspectiveConfig.smoke(full_history_periods=12)


def _register_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> tuple[Path, Path, bytes, pd.DataFrame]:
    history = _history()
    csv_path = tmp_path / "history.csv"
    state_dir = tmp_path / "state-v2"
    _, key = _key(tmp_path)
    _write_csv(csv_path, history)
    monkeypatch.setattr(prospective, "_fit_model", lambda *_: "model-v0")
    monkeypatch.setattr(
        prospective,
        "_predict_snapshot_values",
        lambda *_: (
            [1, 2, 3, 4],
            [
                [1, 6, 11, 16],
                [2, 7, 12, 17],
                [3, 8, 13, 18],
                [4, 9, 14, 19],
                [5, 10, 15, 20],
            ],
            [0.25] * 80,
        ),
    )
    now = datetime(2026, 1, 12, 20, 0, tzinfo=SHANGHAI)
    result = register_prospective(
        csv_path, state_dir, hmac_key=key, config=smoke_config, now=now
    )
    assert result["externalAnchorPayload"]["targetIssue"] == "2026013"
    return csv_path, state_dir, key, history


def test_schema_v2_and_smoke_never_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    _, state_dir, key, _ = _register_smoke(tmp_path, monkeypatch, smoke_config)
    protocol = load_and_verify_artifact(state_dir / "protocol.json", key)
    snapshot = load_and_verify_artifact(state_dir / "versions/0000/snapshot.json", key)
    assert protocol["schemaVersion"] == "kl8_pick4_prospective_v2"
    assert protocol["profile"] == "smoke"
    assert protocol["formalEligible"] is False
    assert snapshot["historicalFrozenConsumed"] is False
    assert snapshot["targetIssue"] == "2026013"
    assert snapshot["targetDate"] == "2026-01-13"


def test_hmac_rejects_coherent_self_rehash_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    _, state_dir, key, _ = _register_smoke(tmp_path, monkeypatch, smoke_config)
    path = state_dir / "versions/0000/snapshot.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["probabilities80"][0] = 0.24
    document["probabilities80"][1] = 0.26
    unsigned = {
        k: v
        for k, v in document.items()
        if k not in {"artifactSha256", "artifactHmacSha256"}
    }
    document["artifactSha256"] = prospective.payload_sha256(unsigned)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commit_path = state_dir / "versions/0000/commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["manifestBytesSha256"]["snapshot.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    commit_unsigned = {
        key: value
        for key, value in commit.items()
        if key not in {"artifactSha256", "artifactHmacSha256"}
    }
    commit["artifactSha256"] = prospective.payload_sha256(commit_unsigned)
    commit_path.write_text(
        json.dumps(commit, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="HMAC"):
        prospective_status(
            state_dir,
            hmac_key=key,
            canonical_csv=tmp_path / "history.csv",
            config=smoke_config,
        )


def test_wrong_key_and_model_replacement_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    csv_path, state_dir, key, _ = _register_smoke(tmp_path, monkeypatch, smoke_config)
    with pytest.raises(ValueError, match="HMAC"):
        prospective_status(
            state_dir,
            hmac_key=b"wrong-key-material" * 3,
            canonical_csv=csv_path,
            config=smoke_config,
        )
    model = state_dir / "versions/0000/model.txt"
    model.write_text("replacement", encoding="utf-8")
    with pytest.raises(ValueError, match="模型|manifest"):
        prospective_status(
            state_dir, hmac_key=key, canonical_csv=csv_path, config=smoke_config
        )


def test_exact_target_and_anchor_receipt_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    csv_path, state_dir, key, history = _register_smoke(
        tmp_path, monkeypatch, smoke_config
    )
    bad = history.copy()
    bad.loc[len(bad)] = {
        "issue": "2026014",
        "date": "2026-01-13",
        "numbers": list(range(21, 41)),
    }
    _write_csv(csv_path, bad)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="targetIssue"):
        update_prospective(
            csv_path,
            state_dir,
            hmac_key=key,
            anchor_receipt_file=receipt,
            config=smoke_config,
            now=datetime(2026, 1, 14, 10, tzinfo=SHANGHAI),
        )

    good = history.copy()
    good.loc[len(good)] = {
        "issue": "2026013",
        "date": "2026-01-13",
        "numbers": list(range(21, 41)),
    }
    _write_csv(csv_path, good)
    anchor = json.loads(
        (state_dir / "versions/0000/anchor_payload.json").read_text(encoding="utf-8")
    )
    late = {
        **anchor,
        "provider": "telegram",
        "messageId": "m1",
        "anchoredAt": "2026-01-13T21:31:00+08:00",
    }
    receipt.write_text(json.dumps(late), encoding="utf-8")
    with pytest.raises(ValueError, match="开奖前"):
        update_prospective(
            csv_path,
            state_dir,
            hmac_key=key,
            anchor_receipt_file=receipt,
            config=smoke_config,
            now=datetime(2026, 1, 14, 10, tzinfo=SHANGHAI),
        )


def test_update_commits_full_recursive_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    csv_path, state_dir, key, history = _register_smoke(
        tmp_path, monkeypatch, smoke_config
    )
    history.loc[len(history)] = {
        "issue": "2026013",
        "date": "2026-01-13",
        "numbers": list(range(1, 21)),
    }
    _write_csv(csv_path, history)
    anchor = json.loads(
        (state_dir / "versions/0000/anchor_payload.json").read_text(encoding="utf-8")
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                **anchor,
                "provider": "telegram",
                "messageId": "m1",
                "anchoredAt": "2026-01-13T20:10:00+08:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prospective, "_fit_model", lambda *_: "model-v1")
    update_prospective(
        csv_path,
        state_dir,
        hmac_key=key,
        anchor_receipt_file=receipt,
        config=smoke_config,
        now=datetime(2026, 1, 14, 10, tzinfo=SHANGHAI),
    )
    status = prospective_status(
        state_dir, hmac_key=key, canonical_csv=csv_path, config=smoke_config
    )
    assert status["observed"] == 1
    assert status["formalPredictionActivated"] is False
    state1 = load_and_verify_artifact(state_dir / "versions/0001/state.json", key)
    commit0 = load_and_verify_artifact(state_dir / "versions/0000/commit.json", key)
    assert state1["previousStateArtifactSha256"] == commit0["stateArtifactSha256"]
    assert state1["observationArtifactSha256"]


def test_observation_metric_self_rehash_without_hmac_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    csv_path, state_dir, key, history = _register_smoke(
        tmp_path, monkeypatch, smoke_config
    )
    history.loc[len(history)] = {
        "issue": "2026013",
        "date": "2026-01-13",
        "numbers": list(range(1, 21)),
    }
    _write_csv(csv_path, history)
    anchor = json.loads(
        (state_dir / "versions/0000/anchor_payload.json").read_text(encoding="utf-8")
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                **anchor,
                "provider": "telegram",
                "messageId": "metric-attack",
                "anchoredAt": "2026-01-13T20:10:00+08:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prospective, "_fit_model", lambda *_: "model-v1")
    update_prospective(
        csv_path,
        state_dir,
        hmac_key=key,
        anchor_receipt_file=receipt,
        config=smoke_config,
        now=datetime(2026, 1, 14, 10, tzinfo=SHANGHAI),
    )
    observation_path = state_dir / "versions/0001/observation.json"
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["primaryHits"] = 4
    unsigned = {
        name: value
        for name, value in observation.items()
        if name not in {"artifactSha256", "artifactHmacSha256"}
    }
    observation["artifactSha256"] = prospective.payload_sha256(unsigned)
    observation_path.write_text(
        json.dumps(observation, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commit_path = state_dir / "versions/0001/commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["manifestBytesSha256"]["observation.json"] = hashlib.sha256(
        observation_path.read_bytes()
    ).hexdigest()
    commit_unsigned = {
        name: value
        for name, value in commit.items()
        if name not in {"artifactSha256", "artifactHmacSha256"}
    }
    commit["artifactSha256"] = prospective.payload_sha256(commit_unsigned)
    commit_path.write_text(
        json.dumps(commit, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="HMAC"):
        prospective_status(
            state_dir,
            hmac_key=key,
            canonical_csv=csv_path,
            config=smoke_config,
        )


def test_partial_staging_is_cleaned_and_committed_version_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_config: Kl8Pick4ProspectiveConfig,
) -> None:
    _, state_dir, key, _ = _register_smoke(tmp_path, monkeypatch, smoke_config)
    partial = state_dir / "versions/0001.tmp"
    partial.mkdir()
    (partial / "snapshot.json").write_text("partial", encoding="utf-8")
    status = prospective_status(
        state_dir,
        hmac_key=key,
        canonical_csv=tmp_path / "history.csv",
        config=smoke_config,
    )
    assert status["observed"] == 0
    assert not partial.exists()
    with pytest.raises(FileExistsError):
        prospective._publish_version(state_dir, 0, {}, key)


def test_snapshot_contract_rejects_bad_probabilities_and_wrong_top4() -> None:
    probabilities = [0.25] * 80
    valid = {
        "primaryTop4": [1, 2, 3, 4],
        "fiveTickets": [
            [1, 6, 11, 16],
            [2, 7, 12, 17],
            [3, 8, 13, 18],
            [4, 9, 14, 19],
            [5, 10, 15, 20],
        ],
        "probabilities80": probabilities,
    }
    prospective._validate_snapshot_predictions(valid)
    invalid = dict(valid)
    invalid["probabilities80"] = [-0.1, 0.6] + probabilities[2:]
    with pytest.raises(ValueError, match="0<p<1"):
        prospective._validate_snapshot_predictions(invalid)
    invalid = dict(valid)
    invalid["primaryTop4"] = [21, 22, 23, 24]
    with pytest.raises(ValueError, match="Top4"):
        prospective._validate_snapshot_predictions(invalid)


def test_gate_requires_exact_500_finite_pvalues_and_formal_profile() -> None:
    summary = {
        "observed": 500,
        "profile": "formal",
        "formalEligible": True,
        "primaryMeanHits": 1.1,
        "meanHitsPerTicket": 1.1,
        "portfolioMeanTotalHits": 5.1,
        "deltaLogLoss": -0.01,
        "deltaBrier": -0.01,
        "holmAdjustedPValues": {
            "primaryTop4": 0.01,
            "meanHitsPerTicket": 0.01,
            "portfolio": 0.01,
            "logLoss": 0.01,
            "brier": 0.01,
        },
        "externalAnchorVerifiedCount": 500,
    }
    blocks = [
        {
            "primaryMeanHits": 1.1,
            "meanHitsPerTicket": 1.1,
            "portfolioMeanTotalHits": 5.1,
            "deltaLogLoss": -0.01,
            "deltaBrier": -0.01,
        }
        for _ in range(5)
    ]
    assert evaluate_formal_gate(summary, blocks)[0] is True
    for field, value in (
        ("observed", 501),
        ("observed", 499),
        ("profile", "smoke"),
        ("formalEligible", False),
    ):
        changed = dict(summary)
        changed[field] = value
        assert evaluate_formal_gate(changed, blocks)[0] is False
    changed = dict(summary)
    changed["holmAdjustedPValues"] = {**summary["holmAdjustedPValues"], "brier": -0.1}
    assert evaluate_formal_gate(changed, blocks)[0] is False


def test_formal_config_is_unique_and_environment_files_required(tmp_path: Path) -> None:
    formal = Kl8Pick4ProspectiveConfig()
    prospective._validate_config(formal)
    changed_rank = replace(formal.rank_config, bootstrap_resamples=100)
    with pytest.raises(ValueError, match="canonical"):
        prospective._validate_config(
            prospective.Kl8Pick4ProspectiveConfig(rank_config=changed_rank)
        )
    with pytest.raises(FileNotFoundError):
        prospective._environment_fingerprint(tmp_path)


def test_hmac_formula_binds_payload_and_sha(tmp_path: Path) -> None:
    key = b"secret" * 8
    path = tmp_path / "artifact.json"
    prospective._write_artifact(path, {"value": 1}, key)
    document = load_and_verify_artifact(path, key)
    unsigned = {"value": 1}
    sha = hashlib.sha256(prospective._canonical_bytes(unsigned)).hexdigest()
    expected = hmac.new(
        key,
        prospective._canonical_bytes(unsigned) + b"\n" + sha.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    assert document["artifactSha256"] == sha
    assert document["artifactHmacSha256"] == expected
