# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.analysis import ssq_8red1blue_v1_prospective as prospective
from src.analysis.ssq_history import SSQDraw

SHANGHAI = ZoneInfo("Asia/Shanghai")
KEY = b"test-only-d8-independent-key-32-bytes"
SOURCE_URL = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?"
    "name=ssq&pageNo=1&pageSize=100&issueCount=&issueStart=&issueEnd=&dayStart="
    "&dayEnd=&week=&systemType=PC"
)


def _draw(issue: int, draw_date: str, offset: int) -> SSQDraw:
    return SSQDraw(
        issue=f"2026{issue:03d}",
        draw_date=draw_date,
        red=tuple(sorted((((index * 5 + offset) % 33) + 1) for index in range(6))),
        blue=offset % 16 + 1,
        source_url=SOURCE_URL,
        raw_hash=f"{issue:064x}"[-64:],
        raw={},
    )


def _draws() -> list[SSQDraw]:
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


def _setup(tmp_path: Path) -> tuple[Path, Path, list[SSQDraw]]:
    draws = _draws()
    csv_path = tmp_path / "ssq.csv"
    report_path = tmp_path / "report.json"
    state_dir = tmp_path / "state"
    _write_csv(csv_path, draws)
    report_path.write_text(
        json.dumps(prospective.build_bound_ensemble_report(draws), ensure_ascii=False),
        encoding="utf-8",
    )
    now = datetime(2026, 1, 21, 12, tzinfo=SHANGHAI)
    prospective.register_prospective(
        csv_path, report_path, state_dir, hmac_key=KEY, now=now
    )
    prospective.create_snapshot(csv_path, report_path, state_dir, hmac_key=KEY, now=now)
    return state_dir, csv_path, draws


def test_register_snapshot_lock_d8_and_zero_overlap(tmp_path: Path) -> None:
    state_dir, _, _ = _setup(tmp_path)
    snapshot = prospective.load_and_verify_artifact(
        state_dir / "versions/0000/snapshot.json", KEY
    )
    assert snapshot["targetIssue"] == "2026009"
    assert snapshot["targetDate"] == "2026-01-22"
    assert len(snapshot["D8"]["red"]) == 8
    assert len(snapshot["D8"]["tickets"]) == 28
    assert snapshot["D8"]["audit"]["overlapWithB"] == 0
    assert snapshot["D8"]["audit"]["combinedUniqueTicketCount"] == 63


def test_tamper_wrong_key_and_catchup_fail_closed(tmp_path: Path) -> None:
    state_dir, csv_path, draws = _setup(tmp_path)
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

    clean = tmp_path / "clean"
    clean.mkdir()
    state_dir, csv_path, draws = _setup(clean)
    target = _draw(9, "2026-01-22", 11)
    skipped = _draw(10, "2026-01-25", 12)
    _write_csv(csv_path, [*draws, target, skipped])
    with pytest.raises(ValueError, match="恰好新增一期"):
        prospective.update_prospective(
            csv_path,
            state_dir,
            hmac_key=KEY,
            now=datetime(2026, 1, 26, 9, tzinfo=SHANGHAI),
        )


def test_exact_one_issue_update_and_no_auto_promotion(tmp_path: Path) -> None:
    state_dir, csv_path, draws = _setup(tmp_path)
    target = _draw(9, "2026-01-22", 11)
    _write_csv(csv_path, [*draws, target])
    result = prospective.update_prospective(
        csv_path, state_dir, hmac_key=KEY, now=datetime(2026, 1, 23, 9, tzinfo=SHANGHAI)
    )
    status = prospective.prospective_status(
        state_dir, canonical_csv=csv_path, hmac_key=KEY
    )
    assert result["completed"] == 1
    assert result["settledIssue"] == "2026009"
    assert result["result"]["overlapWithB"] == 0
    assert status["completed"] == 1
    assert status["horizon"] == 500
    assert status["autoPromotion"] is False
    assert status["formalRecommendationStatus"] == "uniform_abstain"
