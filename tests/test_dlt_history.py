# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from urllib import error as urllib_error
from urllib.parse import parse_qs, urlparse

import pytest

from src.analysis import dlt_history
from src.analysis.dlt_history import (
    DLT_CSV_FIELDS,
    DLT_REQUIRED_HEADERS,
    atomic_write_raw_evidence,
    build_dlt_source_url,
    fetch_dlt_history,
    read_raw_evidence,
    reconcile_raw_jsonl,
    validate_dlt_source_url,
    validate_official_record,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dlt"


def _page_fixture() -> dict[str, object]:
    return json.loads((FIXTURES / "official_page_1.json").read_text(encoding="utf-8"))


def _row(
    issue: str = "26083",
    *,
    result: str = "14 15 16 23 26 07 09",
    draw_date: str = "2026-07-25",
    game_num: str = "85",
    game_name: str = "超级大乐透",
) -> dict[str, object]:
    return {
        "lotteryDrawNum": issue,
        "lotteryDrawResult": result,
        "lotteryDrawTime": draw_date,
        "lotteryGameNum": game_num,
        "lotteryGameName": game_name,
        "verify": 1,
    }


def test_fixture_page_fetches_validated_newest_first_draws() -> None:
    payload = _page_fixture()

    draws = fetch_dlt_history(request_json=lambda _url: payload)

    assert [draw.issue for draw in draws] == ["26083", "26082"]
    assert draws[0].front == (14, 15, 16, 23, 26)
    assert draws[0].back == (7, 9)
    assert draws[0].draw_date == "2026-07-25"
    assert len(draws[0].raw_hash) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lotteryGameNum": "84"}, "gameNo 不是 85"),
        ({"lotteryGameName": "排列3"}, "名称不是超级大乐透"),
        ({"lotteryDrawNum": "26A83"}, "期号非法"),
        ({"lotteryDrawTime": "2026-02-30"}, "日期非法"),
        ({"lotteryDrawResult": "01 02 03 04 05 06"}, "必须为 7 个"),
        ({"lotteryDrawResult": "01 02 03 04 04 06 07"}, "前区不得重复"),
        ({"lotteryDrawResult": "02 01 03 04 05 06 07"}, "前区必须严格升序"),
        ({"lotteryDrawResult": "01 02 03 04 36 06 07"}, "前区范围"),
        ({"lotteryDrawResult": "01 02 03 04 05 07 07"}, "后区不得重复"),
        ({"lotteryDrawResult": "01 02 03 04 05 08 07"}, "后区必须严格升序"),
        ({"lotteryDrawResult": "01 02 03 04 05 06 13"}, "后区范围"),
    ],
)
def test_record_fails_closed_on_wrong_identity_or_malformed_values(
    changes: dict[str, object], message: str
) -> None:
    raw = _row()
    raw.update(changes)

    with pytest.raises(ValueError, match=message):
        validate_official_record(raw, build_dlt_source_url(1))


def test_record_rejects_future_date() -> None:
    future = f"{date.today().year + 1}-01-01"
    with pytest.raises(ValueError, match="未来日期"):
        validate_official_record(
            _row(draw_date=future), build_dlt_source_url(1), today=date.today()
        )


def test_source_url_and_required_headers_are_fixed() -> None:
    valid = build_dlt_source_url(97)
    validate_dlt_source_url(valid)
    query = parse_qs(urlparse(valid).query)
    assert query == {
        "gameNo": ["85"],
        "provinceId": ["0"],
        "pageSize": ["30"],
        "isVerify": ["1"],
        "pageNo": ["97"],
        "termLimits": ["0"],
    }
    assert DLT_REQUIRED_HEADERS["Referer"] == "https://www.sporttery.cn/"
    assert DLT_REQUIRED_HEADERS["Origin"] == "https://www.sporttery.cn"
    assert DLT_REQUIRED_HEADERS["User-Agent"]

    with pytest.raises(ValueError, match="官方白名单"):
        validate_dlt_source_url(valid.replace("webapi.sporttery.cn", "example.com"))
    with pytest.raises(ValueError, match="固定参数非法"):
        validate_dlt_source_url(valid.replace("gameNo=85", "gameNo=84"))
    with pytest.raises(ValueError, match="参数集合"):
        validate_dlt_source_url(f"{valid}&extra=1")


def test_pagination_validates_page_identity_total_and_cross_page_chronology() -> None:
    newest = [_row(f"26{number:03d}") for number in range(130, 100, -1)]
    oldest = [_row("26100", draw_date="2026-07-22")]

    def request_json(url: str) -> dict[str, object]:
        page_no = int(parse_qs(urlparse(url).query)["pageNo"][0])
        rows = newest if page_no == 1 else oldest
        return {
            "errorCode": "0",
            "errorMessage": "处理成功",
            "success": True,
            "value": {
                "lastPoolDraw": newest[0],
                "list": rows,
                "pageNo": page_no,
                "pageSize": 30,
                "pages": 2,
                "total": 31,
            },
        }

    draws = fetch_dlt_history(request_json=request_json)
    assert len(draws) == 31
    assert draws[0].issue == "26130"
    assert draws[-1].issue == "26100"

    malformed = request_json(build_dlt_source_url(1))
    malformed["value"]["pageNo"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="pageNo 不匹配"):
        fetch_dlt_history(request_json=lambda _url: malformed)

    def duplicate_request(url: str) -> dict[str, object]:
        payload = request_json(url)
        if payload["value"]["pageNo"] == 2:  # type: ignore[index]
            payload["value"]["list"] = [newest[-1]]  # type: ignore[index]
        return payload

    with pytest.raises(ValueError, match="重复期号"):
        fetch_dlt_history(request_json=duplicate_request)


def test_atomic_raw_evidence_reconciles_to_lf_csv_and_stable_report(
    tmp_path: Path,
) -> None:
    draws = fetch_dlt_history(request_json=lambda _url: _page_fixture())
    raw_path = tmp_path / "raw" / "history.jsonl"
    csv_path = tmp_path / "official_history.csv"
    report_path = tmp_path / "reconciliation.json"

    atomic_write_raw_evidence(raw_path, draws, fetched_at="2026-07-27T00:00:00+00:00")
    reconciled, report = reconcile_raw_jsonl(raw_path, csv_path, report_path)

    assert [draw.issue for draw in reconciled] == ["26082", "26083"]
    csv_bytes = csv_path.read_bytes()
    assert b"\r\n" not in csv_bytes and csv_bytes.endswith(b"\n")
    assert csv_bytes.splitlines()[0].decode() == ",".join(DLT_CSV_FIELDS)
    assert report["count"] == 2
    assert report["first"] == {"issue": "26082", "date": "2026-07-22"}
    assert report["latest"] == {"issue": "26083", "date": "2026-07-25"}
    assert report["dataSha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert (
        report["rawEvidenceSha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    )
    assert len(report["sourceFingerprints"]) == 1
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_reconciliation_deduplicates_repeated_evidence_but_rejects_conflict(
    tmp_path: Path,
) -> None:
    draws = fetch_dlt_history(request_json=lambda _url: _page_fixture())
    raw_path = tmp_path / "history.jsonl"
    atomic_write_raw_evidence(raw_path, [draws[0], draws[0]])
    reconciled, _ = reconcile_raw_jsonl(
        raw_path, tmp_path / "history.csv", tmp_path / "report.json"
    )
    assert len(reconciled) == 1

    payloads = [json.loads(line) for line in raw_path.read_text().splitlines()]
    payloads[1]["raw"]["lotteryDrawResult"] = "01 02 03 04 05 06 07"
    payloads[1]["rawHash"] = payloads[0]["rawHash"]
    raw_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="原始哈希不匹配"):
        read_raw_evidence(raw_path)


def test_transport_failure_retries_with_timeout_backoff_and_official_headers(
    monkeypatch,
) -> None:
    payload = _page_fixture()
    calls: list[tuple[object, float]] = []
    sleeps: list[float] = []

    class Headers:
        @staticmethod
        def get_content_type() -> str:
            return "application/json"

    class Response:
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def geturl() -> str:
            return build_dlt_source_url(1)

        @staticmethod
        def read() -> bytes:
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib_error.URLError("temporary")
        return Response()

    monkeypatch.setattr(dlt_history.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dlt_history.time, "sleep", sleeps.append)
    result = dlt_history._default_request_json(build_dlt_source_url(1))

    assert result == payload
    assert len(calls) == 2 and sleeps == [0.5]
    assert calls[0][1] == dlt_history.DLT_HTTP_TIMEOUT_SECONDS
    sent_headers = dict(calls[0][0].header_items())
    assert sent_headers["Referer"] == DLT_REQUIRED_HEADERS["Referer"]
    assert sent_headers["Origin"] == DLT_REQUIRED_HEADERS["Origin"]
    assert sent_headers["User-agent"] == DLT_REQUIRED_HEADERS["User-Agent"]


def test_malformed_page_does_not_replace_existing_raw_file(tmp_path: Path) -> None:
    raw_path = tmp_path / "history.jsonl"
    raw_path.write_bytes(b"keep-me\n")
    payload = _page_fixture()
    payload["value"]["list"][0]["lotteryGameNum"] = "84"  # type: ignore[index]

    with pytest.raises(ValueError):
        draws = fetch_dlt_history(request_json=lambda _url: payload)
        atomic_write_raw_evidence(raw_path, draws)

    assert raw_path.read_bytes() == b"keep-me\n"
