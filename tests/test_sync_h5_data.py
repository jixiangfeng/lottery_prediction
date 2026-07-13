# -*- coding: utf-8 -*-

import json
from pathlib import Path

from scripts.sync_h5_data import sync_reports


def test_sync_reports_copies_walk_forward_report_when_present(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    data_dir = reports_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "kl8_daily_2026182.json").write_text(json.dumps({"issue": "2026182"}), encoding="utf-8")
    (data_dir / "walk_forward_kl8.json").write_text(json.dumps({"bestStrategy": "omission_mix"}), encoding="utf-8")

    sync_reports(reports_dir, tmp_path / "h5" / "public")

    target = tmp_path / "h5" / "public" / "report-data" / "walk_forward_kl8.json"
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["bestStrategy"] == "omission_mix"
