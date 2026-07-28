# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from src.analysis.ssq_history import SSQDraw

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECONCILE_CLI = _load_script_module(
    "ssq_reconcile_history_cli", "scripts/ssq_reconcile_history.py"
)
ENSEMBLE_CLI = _load_script_module("ssq_ensemble_v1_cli", "scripts/ssq_ensemble_v1.py")


def test_reconcile_missing_data_is_successful_no_action(tmp_path, capsys):
    output = tmp_path / "official_history.csv"

    result = RECONCILE_CLI.main(
        [
            "--raw-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--output-csv",
            str(output),
        ]
    )

    assert result == 0
    assert not output.exists()
    assert "不动作" in capsys.readouterr().out


def test_ensemble_missing_data_is_successful_no_action(tmp_path, capsys):
    output = tmp_path / "report.json"

    result = ENSEMBLE_CLI.main(
        ["--csv", str(tmp_path / "missing.csv"), "--output", str(output)]
    )

    assert result == 0
    assert not output.exists()
    assert "不动作" in capsys.readouterr().out


def test_ensemble_report_hash_binds_audit_metadata():
    report = {
        "diversifiedPortfolioV2": {
            "protocolSha256": "fixture-diversified",
            "groups": [{"group": 1, "red": [1, 2, 3, 4, 5, 6, 7], "blue": 1}],
        },
        "auditMetadata": {
            "orderedRed6Combinations": [
                {
                    "rank": 1,
                    "red": ["01", "02", "03", "04", "05", "06"],
                    "redScore": 1.0,
                }
            ],
            "orderedRed6CombinationCount": 1,
            "researchBlueTop1": 16,
        },
    }
    draws = [
        SSQDraw(
            issue="2026001",
            draw_date="2026-01-01",
            red=(1, 2, 3, 4, 5, 6),
            blue=16,
            source_url="fixture",
            raw_hash="fixture",
            raw={},
        )
    ]

    bound = ENSEMBLE_CLI._bind_report_to_canonical(report, draws)
    unsigned = dict(bound)
    claimed = unsigned.pop("reportSha256")
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert claimed == hashlib.sha256(canonical).hexdigest()
    tampered = dict(unsigned)
    tampered["auditMetadata"] = {
        **unsigned["auditMetadata"],
        "researchBlueTop1": 15,
    }
    assert claimed != ENSEMBLE_CLI._payload_sha256(tampered)
    tampered_diversified = dict(unsigned)
    tampered_diversified["diversifiedPortfolioV2"] = {
        **unsigned["diversifiedPortfolioV2"],
        "protocolSha256": "tampered",
    }
    assert claimed != ENSEMBLE_CLI._payload_sha256(tampered_diversified)
