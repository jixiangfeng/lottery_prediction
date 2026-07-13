# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src import data_fetcher


@pytest.fixture
def sample_html() -> str:
    numbers = "".join(f"<td>{i:02d}</td>" for i in range(1, 21))
    return f"""
    <div class="wrap_datachart">
      <table>
        <tbody id="tdata">
          <tr>
            <td>2024002</td>
            {numbers}
          </tr>
        </tbody>
      </table>
    </div>
    """


@pytest.fixture
def sample_sequence_text() -> str:
    nums = " ".join(f"{i:02d}" for i in range(1, 21))
    return f"2024001 {nums},其他内容"


def test_http_client_rejects_unknown_domain():
    client = data_fetcher.LotteryHttpClient(timeout=1, retries=1, backoff_factor=0.1, user_agent="test")
    with pytest.raises(ValueError):
        client.get_text("https://example.com/data")


def test_http_client_fetches_allowed_domain(monkeypatch):
    client = data_fetcher.LotteryHttpClient(timeout=1, retries=1, backoff_factor=0.1, user_agent="test")

    class DummyResponse:
        def __init__(self) -> None:
            self.encoding = None
            self.text = "<html>ok</html>"

        def raise_for_status(self) -> None:
            return None

    dummy = DummyResponse()

    def fake_get(self, url, headers, timeout):
        assert "datachart.500.com" in url
        return dummy

    monkeypatch.setattr(requests.Session, "get", fake_get, raising=True)
    text = client.get_text("https://datachart.500.com/kl8/history/echo")
    assert dummy.encoding == "utf-8"
    assert text == dummy.text

def test_parse_issue_list(sample_html):
    cfg = data_fetcher.LOTTERY_CONFIGS["kl8"]
    df = data_fetcher._parse_issue_list(cfg, sample_html)
    assert list(df.columns) == ["期数"] + [f"红球_{i}" for i in range(1, 21)]
    assert df.iloc[0]["期数"] == "2024002"


def test_parse_sequence(sample_sequence_text):
    df = data_fetcher._parse_kl8_sequence(sample_sequence_text)
    assert df.iloc[0]["期数"] == "2024001"
    assert df.iloc[0]["红球_1"] == "01"
    assert len(df.columns) == 21


def test_download_history_and_load(monkeypatch, sample_html, sample_sequence_text, tmp_path):
    output_dir = tmp_path / "data" / "kl8"
    output_dir.mkdir(parents=True, exist_ok=True)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_text(self, url: str) -> str:
            if "917500" in url:
                return sample_sequence_text
            return sample_html

    monkeypatch.setattr(data_fetcher, "LotteryHttpClient", FakeClient)

    result = data_fetcher.download_history("kl8", start=2024001, end=2024002, use_sequence_order=False)
    assert result.total_issues == 1
    csv_path = Path(result.saved_path)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "期数" in df.columns

    seq_result = data_fetcher.download_history("kl8", use_sequence_order=True)
    assert seq_result.total_issues >= 1
    meta_path = Path(seq_result.saved_path).parent / "download_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["code"] == "kl8"

    loaded = data_fetcher.load_history("kl8")
    assert not loaded.empty
