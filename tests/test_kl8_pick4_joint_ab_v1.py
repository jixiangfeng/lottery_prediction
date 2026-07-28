# -*- coding: utf-8 -*-
"""KL8 Pick4 同成本联合概率 A/B v1 的协议攻击回归测试。"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.analysis import kl8_pick4_joint_ab_v1 as joint_ab
from src.analysis.kl8_pick4_joint_portfolio_v1 import (
    ConditionalPoisson20,
    evaluation_to_dict,
)
from src.analysis.kl8_pick4_rank_challenger import ranked_pick4_portfolio

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _history(periods: int = 12) -> pd.DataFrame:
    first = datetime(2026, 7, 1, tzinfo=SHANGHAI)
    rows: list[dict[str, Any]] = []
    for index in range(periods):
        date = (first + timedelta(days=index)).date()
        numbers = sorted((((index * 11 + offset * 3) % 80) + 1) for offset in range(20))
        rows.append(
            {
                "issue": f"2026{index + 1:03d}",
                "date": date.isoformat(),
                "numbers": numbers,
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, history: pd.DataFrame) -> None:
    serial = history.copy()
    serial["numbers"] = serial["numbers"].map(lambda values: " ".join(map(str, values)))
    serial.to_csv(path, index=False)


def _fake_prediction(
    history: pd.DataFrame, **_: object
) -> tuple[str, dict[str, object]]:
    scores = np.linspace(1.0, 0.0, 80, dtype=np.float64) + len(history) * 1e-9
    ranking_order = np.lexsort((np.arange(80), -scores))
    ranking = [
        {"rank": rank + 1, "number": int(index + 1), "score": float(scores[index])}
        for rank, index in enumerate(ranking_order)
    ]
    probabilities = [0.25] * 80
    distribution = ConditionalPoisson20(probabilities)
    control = distribution.evaluate_portfolio(ranked_pick4_portfolio(scores))
    challenger = control
    union = sorted(number for ticket in control.tickets for number in ticket)
    benchmarks = [
        [list(ticket) for ticket in control.tickets]
        for _ in range(joint_ab.RANDOM_BENCHMARKS)
    ]
    return (
        f"locked-model-{len(history)}",
        {
            "scoreRanking": ranking,
            "probabilities80": probabilities,
            "auditedMarginals80": probabilities,
            "jointModelMarginalPreservationClaim": False,
            "top20Union": union,
            "controlA": evaluation_to_dict(control),
            "challengerB": evaluation_to_dict(challenger),
            "objectiveDeltaBMinusA": {
                "sumExact4": 0.0,
                "sumAtLeast3": 0.0,
                "sumAtLeast2": 0.0,
            },
            "objectiveIsPayoutSurrogateOnly": True,
            "randomSameCostPortfolios": benchmarks,
        },
    )


@pytest.fixture
def initialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, bytes, pd.DataFrame]:
    history = _history()
    csv_path = tmp_path / "history.csv"
    state_dir = tmp_path / "state"
    key = b"joint-ab-test-key-material-0123456789-abcdef"
    _write_csv(csv_path, history)
    monkeypatch.setattr(joint_ab, "_build_snapshot_prediction", _fake_prediction)
    joint_ab.initialize_joint_ab(
        csv_path,
        state_dir,
        hmac_key=key,
        now=datetime(2026, 7, 12, 20, 0, tzinfo=SHANGHAI),
    )
    return csv_path, state_dir, key, history


def _append_target(
    csv_path: Path, history: pd.DataFrame, *, extra: int = 1
) -> pd.DataFrame:
    updated = history.copy()
    for offset in range(extra):
        index = len(updated)
        date = datetime(2026, 7, 1, tzinfo=SHANGHAI) + timedelta(days=index)
        numbers = sorted((((index * 11 + item * 3) % 80) + 1) for item in range(20))
        updated.loc[len(updated)] = {
            "issue": f"2026{index + 1:03d}",
            "date": date.date().isoformat(),
            "numbers": numbers,
        }
    _write_csv(csv_path, updated)
    return updated


def test_initialize_has_no_future_leakage_and_no_formal_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _history()
    csv_path = tmp_path / "history.csv"
    state_dir = tmp_path / "state"
    key = b"joint-ab-test-key-material-0123456789-abcdef"
    seen_lengths: list[int] = []

    def recorder(
        frame: pd.DataFrame, **kwargs: object
    ) -> tuple[str, dict[str, object]]:
        seen_lengths.append(len(frame))
        return _fake_prediction(frame, **kwargs)

    _write_csv(csv_path, history)
    monkeypatch.setattr(joint_ab, "_build_snapshot_prediction", recorder)
    result = joint_ab.initialize_joint_ab(
        csv_path,
        state_dir,
        hmac_key=key,
        now=datetime(2026, 7, 12, 20, 0, tzinfo=SHANGHAI),
    )
    assert seen_lengths == [12]
    assert result["targetIssue"] == "2026013"
    assert result["formalCandidates"] == []
    status = joint_ab.joint_ab_status(csv_path, state_dir, hmac_key=key)
    assert status["observed"] == 0
    assert status["formalCandidates"] == []
    assert status["productionActivation"] is False


def test_hmac_tamper_and_append_only_reinitialize_are_rejected(
    initialized: tuple[Path, Path, bytes, pd.DataFrame],
) -> None:
    csv_path, state_dir, key, _ = initialized
    with pytest.raises(FileExistsError, match="禁止重置"):
        joint_ab.initialize_joint_ab(csv_path, state_dir, hmac_key=key)
    snapshot_path = state_dir / "versions/0000/snapshot.json"
    document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document["top20Union"][0] = 80
    snapshot_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希|HMAC"):
        joint_ab.joint_ab_status(csv_path, state_dir, hmac_key=key)


def test_unavailable_is_noop_and_step_is_idempotent(
    initialized: tuple[Path, Path, bytes, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path, state_dir, key, history = initialized
    before = sorted(path.name for path in (state_dir / "versions").iterdir())
    assert (
        joint_ab.step_joint_ab(csv_path, state_dir, hmac_key=key)["status"]
        == "unavailable"
    )
    assert sorted(path.name for path in (state_dir / "versions").iterdir()) == before
    updated = _append_target(csv_path, history)
    result = joint_ab.step_joint_ab(
        csv_path,
        state_dir,
        hmac_key=key,
        now=datetime(2026, 7, 13, 20, 0, tzinfo=SHANGHAI),
    )
    assert result["observed"] == 1
    assert (
        joint_ab.step_joint_ab(csv_path, state_dir, hmac_key=key)["status"]
        == "unavailable"
    )
    status = joint_ab.joint_ab_status(csv_path, state_dir, hmac_key=key)
    assert status["observed"] == 1
    assert len(updated) == 13


def test_settlement_uses_locked_tickets_and_total_hits_are_equal(
    initialized: tuple[Path, Path, bytes, pd.DataFrame],
) -> None:
    csv_path, state_dir, key, history = initialized
    updated = _append_target(csv_path, history)
    joint_ab.step_joint_ab(
        csv_path,
        state_dir,
        hmac_key=key,
        now=datetime(2026, 7, 13, 20, 0, tzinfo=SHANGHAI),
    )
    snapshot = joint_ab.load_and_verify_artifact(
        state_dir / "versions/0000/snapshot.json", key
    )
    observation = joint_ab.load_and_verify_artifact(
        state_dir / "versions/0001/observation.json", key
    )
    expected = joint_ab._observation_payload(snapshot, updated.iloc[-1].to_dict())
    unsigned = {
        name: value
        for name, value in observation.items()
        if not name.startswith("artifact")
    }
    assert unsigned == expected
    assert observation["armA"]["totalHits"] == observation["armB"]["totalHits"]


def test_catch_up_refusal(
    initialized: tuple[Path, Path, bytes, pd.DataFrame],
) -> None:
    csv_path, state_dir, key, history = initialized
    _append_target(csv_path, history, extra=2)
    with pytest.raises(ValueError, match="backfill|catch-up"):
        joint_ab.step_joint_ab(csv_path, state_dir, hmac_key=key)


def test_exact_500_stop_and_gate_requires_exact_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"observed": 500}
    protocol = {"initialHistoryRows": 12}
    monkeypatch.setattr(
        joint_ab.v2_prospective, "load_full_canonical_csv", lambda _: _history()
    )
    monkeypatch.setattr(
        joint_ab, "_verify_chain", lambda *_: (state, None, [], protocol)
    )
    assert joint_ab.step_joint_ab("unused.csv", "unused", hmac_key=b"x" * 32) == {
        "status": "complete",
        "observed": 500,
        "reviewStatus": "pending-human-review",
    }
    incomplete, reasons = joint_ab._gate({"observed": 499}, [])
    assert incomplete is False
    assert reasons == ["requires_exactly_500_settled_draws"]


def test_cli_exposes_only_fixed_operational_actions() -> None:
    script = Path("scripts/kl8_pick4_joint_ab_v1.py")
    spec = importlib.util.spec_from_file_location("joint_ab_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module._parser()
    assert set(parser._subparsers._group_actions[0].choices) == {
        "initialize",
        "step",
        "status",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["initialize", "--target", "2026999"])


def test_cli_progress_is_atomic(tmp_path: Path) -> None:
    script = Path("scripts/kl8_pick4_joint_ab_v1.py")
    spec = importlib.util.spec_from_file_location("joint_ab_progress_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state_dir = tmp_path / "state"
    write = module._progress_writer(str(state_dir), "initialize")
    write({"event": "portfolioOptimizationStarted"})
    progress_path = Path(f"{state_dir}.progress.json")
    assert (
        json.loads(progress_path.read_text(encoding="utf-8"))["event"]
        == "portfolioOptimizationStarted"
    )
    assert not list(tmp_path.glob(".*.tmp"))
