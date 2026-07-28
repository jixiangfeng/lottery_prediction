# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.analysis import ssq_diversified_portfolio_v2_prospective as prospective
from src.analysis.ssq_history import SSQDraw

SHANGHAI = ZoneInfo("Asia/Shanghai")
SOURCE_URL = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?"
    "name=ssq&pageNo=1&pageSize=100&issueCount=&issueStart=&issueEnd=&dayStart="
    "&dayEnd=&week=&systemType=PC"
)
KEY = b"test-only-ssq-prospective-key-32-bytes"


def _draw(issue: int, draw_date: str, offset: int = 0) -> SSQDraw:
    red = tuple(sorted((((index * 5 + offset) % 33) + 1) for index in range(6)))
    if len(set(red)) != 6:
        red = (1, 6, 11, 16, 21, 26)
    return SSQDraw(
        issue=f"2026{issue:03d}",
        draw_date=draw_date,
        red=red,
        blue=(offset % 16) + 1,
        source_url=SOURCE_URL,
        raw_hash=f"{issue:064x}"[-64:],
        raw={},
    )


def _base_draws() -> list[SSQDraw]:
    dates = [
        "2026-01-04",
        "2026-01-06",
        "2026-01-08",
        "2026-01-11",
        "2026-01-13",
        "2026-01-15",
        "2026-01-18",
        "2026-01-20",
    ]
    return [_draw(index + 1, value, index) for index, value in enumerate(dates)]


def _write_csv(path: Path, draws: list[SSQDraw]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("issue", "date", "red", "blue", "source_url", "raw_hash"))
        for draw in reversed(draws):
            writer.writerow(
                (
                    draw.issue,
                    draw.draw_date,
                    " ".join(f"{ball:02d}" for ball in draw.red),
                    f"{draw.blue:02d}",
                    draw.source_url,
                    draw.raw_hash,
                )
            )


def _write_report(path: Path, draws: list[SSQDraw]) -> None:
    path.write_text(
        json.dumps(
            prospective.build_bound_ensemble_report(draws),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _registered_snapshot(tmp_path: Path) -> tuple[Path, Path, list[SSQDraw]]:
    draws = _base_draws()
    csv_path = tmp_path / "ssq.csv"
    report_path = tmp_path / "report.json"
    state_dir = tmp_path / "state"
    _write_csv(csv_path, draws)
    _write_report(report_path, draws)
    now = datetime(2026, 1, 21, 12, tzinfo=SHANGHAI)
    prospective.register_prospective(
        csv_path, report_path, state_dir, hmac_key=KEY, now=now
    )
    prospective.create_snapshot(csv_path, report_path, state_dir, hmac_key=KEY, now=now)
    return state_dir, csv_path, draws


def _append_target(draws: list[SSQDraw]) -> SSQDraw:
    return _draw(9, "2026-01-22", 11)


def test_deterministic_snapshot_and_future_prefix_invariance(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    state_left, _, draws_left = _registered_snapshot(left)
    state_right, _, _ = _registered_snapshot(right)

    first = (state_left / "versions/0000/snapshot.json").read_bytes()
    second = (state_right / "versions/0000/snapshot.json").read_bytes()
    assert first == second

    future = _append_target(draws_left)
    assert future.red != draws_left[-1].red
    assert (state_left / "versions/0000/snapshot.json").read_bytes() == first


def test_snapshot_locks_a_b_and_32_matched_cost_controls(tmp_path: Path) -> None:
    state_dir, _, _ = _registered_snapshot(tmp_path)
    snapshot = prospective.load_and_verify_artifact(
        state_dir / "versions/0000/snapshot.json", KEY
    )
    portfolios = snapshot["portfolios"]
    assert isinstance(portfolios, dict)
    for label in ("A", "B"):
        assert portfolios[label]["cost"] == 35
        assert len(portfolios[label]["red7Groups"]) == 5
        assert len(portfolios[label]["tickets"]) == 35
        assert (
            len(
                {
                    (tuple(ticket["red"]), ticket["blue"])
                    for ticket in portfolios[label]["tickets"]
                }
            )
            == 35
        )
    assert len(set(portfolios["A"]["blues"])) == 1
    assert len(set(portfolios["B"]["blues"])) == 5
    assert portfolios["C"]["count"] == 32
    assert len(portfolios["C"]["portfolios"]) == 32
    assert all(item["cost"] == 35 for item in portfolios["C"]["portfolios"])


def test_tamper_and_wrong_key_fail_closed(tmp_path: Path) -> None:
    state_dir, csv_path, _ = _registered_snapshot(tmp_path)
    snapshot_path = state_dir / "versions/0000/snapshot.json"
    document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document["targetIssue"] = "2026999"
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA"):
        prospective.prospective_status(state_dir, canonical_csv=csv_path, hmac_key=KEY)
    with pytest.raises(ValueError, match="HMAC"):
        prospective.load_and_verify_artifact(
            state_dir / "manifest.json", b"wrong-key-that-is-long-enough-000000"
        )


def test_update_rejects_before_draw_and_requires_exactly_one_issue(
    tmp_path: Path,
) -> None:
    state_dir, csv_path, draws = _registered_snapshot(tmp_path)
    target = _append_target(draws)
    _write_csv(csv_path, [*draws, target])
    with pytest.raises(ValueError, match="截止前"):
        prospective.update_prospective(
            csv_path,
            state_dir,
            hmac_key=KEY,
            now=datetime(2026, 1, 22, 20, tzinfo=SHANGHAI),
        )

    skipped = _draw(10, "2026-01-25", 12)
    _write_csv(csv_path, [*draws, target, skipped])
    with pytest.raises(ValueError, match="恰好一期"):
        prospective.update_prospective(
            csv_path,
            state_dir,
            hmac_key=KEY,
            now=datetime(2026, 1, 26, 10, tzinfo=SHANGHAI),
        )


def test_update_observation_status_and_no_formal_activation(tmp_path: Path) -> None:
    state_dir, csv_path, draws = _registered_snapshot(tmp_path)
    target = _append_target(draws)
    _write_csv(csv_path, [*draws, target])
    result = prospective.update_prospective(
        csv_path,
        state_dir,
        hmac_key=KEY,
        now=datetime(2026, 1, 23, 9, tzinfo=SHANGHAI),
    )
    observation = prospective.load_and_verify_artifact(
        state_dir / "versions/0001/observation.json", KEY
    )
    status = prospective.prospective_status(
        state_dir, canonical_csv=csv_path, hmac_key=KEY
    )

    assert result["completed"] == 1
    assert observation["officialResult"] == prospective._draw_payload(target)
    assert observation["cost"] == {
        "A": 35,
        "B": 35,
        "eachC": 35,
        "controlCount": 32,
    }
    assert set(observation["metrics"]) == {"A", "B", "C32Mean", "C32"}
    assert len(observation["metrics"]["C32"]) == 32
    assert status["completed"] == 1
    assert status["horizon"] == 500
    assert set(status["metricsPerIssue"]) == {"A", "B", "C32Mean"}
    assert status["formalRecommendationStatus"] == "uniform_abstain"
    assert status["researchOnly"] is True
    assert status["predictionClaim"] is False
    assert status["formalActivation"] is False
    assert status["autoPromotion"] is False
    serialized = json.dumps(status, ensure_ascii=False)
    assert KEY.decode("ascii") not in serialized


def test_atomic_append_rolls_back_if_current_pointer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir, csv_path, draws = _registered_snapshot(tmp_path)
    _write_csv(csv_path, [*draws, _append_target(draws)])
    original = prospective._atomic_replace_artifact

    def fail_current(
        path: Path,
        payload: dict[str, object],
        key: bytes,
    ) -> dict[str, object]:
        if path.name == "current.json":
            raise OSError("模拟current指针写入失败")
        return original(path, payload, key)

    monkeypatch.setattr(prospective, "_atomic_replace_artifact", fail_current)
    with pytest.raises(OSError, match="模拟"):
        prospective.update_prospective(
            csv_path,
            state_dir,
            hmac_key=KEY,
            now=datetime(2026, 1, 23, 9, tzinfo=SHANGHAI),
        )
    assert not (state_dir / "versions/0001").exists()
    current = prospective.load_and_verify_artifact(state_dir / "current.json", KEY)
    assert current["version"] == 0


def test_register_once_snapshot_once_and_genesis_has_no_result(tmp_path: Path) -> None:
    state_dir, csv_path, draws = _registered_snapshot(tmp_path)
    report_path = tmp_path / "report.json"
    _write_report(report_path, draws)
    genesis = prospective.load_and_verify_artifact(
        state_dir / "versions/0000/observation.json", KEY
    )
    assert genesis["kind"] == "genesis_no_result"
    assert genesis["observed"] is False
    assert "officialResult" not in genesis
    with pytest.raises(FileExistsError):
        prospective.register_prospective(
            csv_path,
            report_path,
            state_dir,
            hmac_key=KEY,
            now=datetime(2026, 1, 21, 12, tzinfo=SHANGHAI),
        )
    with pytest.raises(FileExistsError):
        prospective.create_snapshot(
            csv_path,
            report_path,
            state_dir,
            hmac_key=KEY,
            now=datetime(2026, 1, 21, 12, tzinfo=SHANGHAI),
        )


def test_rebuild_accepts_only_manifest_bound_report_hash_drift() -> None:
    actual = {
        "reportSha256": "old-report-hash",
        "targetIssue": "2026086",
        "portfolios": {"B": {"red": [1, 2, 3]}},
    }
    expected = {
        "reportSha256": "new-report-hash",
        "targetIssue": "2026086",
        "portfolios": {"B": {"red": [1, 2, 3]}},
    }
    manifest = {"verifiedReportSha256": "old-report-hash"}

    prospective._assert_active_snapshot_rebuild_compatible(actual, expected, manifest)

    changed_numbers = {
        **expected,
        "portfolios": {"B": {"red": [1, 2, 4]}},
    }
    with pytest.raises(ValueError, match="确定性重建"):
        prospective._assert_active_snapshot_rebuild_compatible(
            actual, changed_numbers, manifest
        )
    with pytest.raises(ValueError, match="登记报告"):
        prospective._assert_active_snapshot_rebuild_compatible(
            actual, expected, {"verifiedReportSha256": "another-hash"}
        )
