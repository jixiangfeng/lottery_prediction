# -*- coding: utf-8 -*-
import pytest

from src import common


def test_get_data_run_triggers_download(monkeypatch):
    captured = {}

    def fake_download(code, start=None, end=None, use_sequence_order=False):
        captured["code"] = code
        captured["start"] = start
        captured["end"] = end
        captured["sequence"] = use_sequence_order

    monkeypatch.setattr(common, "download_history", fake_download)
    common.get_data_run("kl8", sequence_mode=True, start_issue=2024001, end_issue=2024002)

    assert captured == {
        "code": "kl8",
        "start": 2024001,
        "end": 2024002,
        "sequence": True,
    }


def test_get_data_run_invalid_code():
    with pytest.raises(ValueError):
        common.get_data_run("ssq")


def test_get_current_number(monkeypatch):
    monkeypatch.setattr(common, "get_current_issue", lambda code: "20241231")
    assert common.get_current_number("kl8") == "20241231"
