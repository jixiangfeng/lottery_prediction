# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from scripts import dlt_fetch_history, dlt_reconcile_history
from src.analysis.dlt_history import build_dlt_source_url, validate_official_record


def _draw():
    return validate_official_record(
        {
            "lotteryDrawNum": "26083",
            "lotteryDrawResult": "14 15 16 23 26 07 09",
            "lotteryDrawTime": "2026-07-25",
            "lotteryGameName": "超级大乐透",
            "lotteryGameNum": "85",
        },
        build_dlt_source_url(1),
    )


def test_fetch_cli_writes_only_validated_atomic_raw_evidence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "history.jsonl"
    monkeypatch.setattr(dlt_fetch_history, "fetch_dlt_history", lambda: [_draw()])

    result = dlt_fetch_history.main(["--output-jsonl", str(output)])

    assert result == 0
    assert output.read_bytes().endswith(b"\n")
    assert "1 条" in capsys.readouterr().out


def test_fetch_cli_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "history.jsonl"
    output.write_bytes(b"existing\n")

    def fail():
        raise ValueError("malformed page")

    monkeypatch.setattr(dlt_fetch_history, "fetch_dlt_history", fail)
    assert dlt_fetch_history.main(["--output-jsonl", str(output)]) == 1
    assert output.read_bytes() == b"existing\n"
    assert "抓取失败" in capsys.readouterr().err


def test_reconcile_cli_generates_csv_and_report(tmp_path: Path) -> None:
    raw = tmp_path / "history.jsonl"
    csv = tmp_path / "official_history.csv"
    report = tmp_path / "reconciliation.json"
    from src.analysis.dlt_history import atomic_write_raw_evidence

    atomic_write_raw_evidence(raw, [_draw()])
    result = dlt_reconcile_history.main(
        [
            "--raw-jsonl",
            str(raw),
            "--output-csv",
            str(csv),
            "--output-report",
            str(report),
        ]
    )

    assert result == 0
    assert csv.exists() and report.exists()
