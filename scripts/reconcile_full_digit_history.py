# -*- coding: utf-8 -*-
"""从已保存的多源原始响应重建福彩3D/排列三规范历史 CSV。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Draw:
    """一条跨源对账记录。"""

    issue: str
    number: str
    date: str


def _plain_cell(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def parse_500_history(path: Path, issue_digits: int) -> dict[str, Draw]:
    """解析500历史页保存的 GBK HTML。"""

    text = path.read_bytes().decode("gbk", "replace")
    draws: dict[str, Draw] = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
        cells = [
            _plain_cell(cell)
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)
        ]
        issue = next(
            (value for value in cells if re.fullmatch(rf"\d{{{issue_digits}}}", value)),
            None,
        )
        number = next(
            (value for value in cells if re.fullmatch(r"(?:\d\s+){2}\d", value)),
            None,
        )
        date = next(
            (
                value
                for value in reversed(cells)
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            ),
            None,
        )
        if issue is not None and number is not None and date is not None:
            draws[issue] = Draw(issue, number.replace(" ", ""), date)
    if not draws:
        raise ValueError(f"500历史页未解析到开奖记录：{path}")
    return draws


def parse_17500_history(path: Path, issue_digits: int) -> dict[str, Draw]:
    """解析17500空格分隔历史文本。"""

    draws: dict[str, Draw] = {}
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        cells = line.split()
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        issue = cells[0][-issue_digits:]
        if len(issue) != issue_digits:
            continue
        number = "".join(cells[2:5])
        if re.fullmatch(r"\d{3}", number):
            draws[issue] = Draw(issue, number, cells[1])
    if not draws:
        raise ValueError(f"17500历史文本未解析到开奖记录：{path}")
    return draws


def parse_official_csv(path: Path) -> dict[str, Draw]:
    """读取项目抓取器保存的官方接口原始 CSV。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["期数"]: Draw(row["期数"], row["开奖号码"], row["开奖日期"][:10])
        for row in rows
    }


def _assert_internal_issue_continuity(draws: dict[str, Draw], year_digits: int) -> None:
    grouped: dict[str, list[int]] = {}
    for issue in draws:
        grouped.setdefault(issue[:year_digits], []).append(int(issue[year_digits:]))
    gaps: dict[str, list[int]] = {}
    for year, indexes in grouped.items():
        missing = sorted(set(range(min(indexes), max(indexes) + 1)) - set(indexes))
        if missing:
            gaps[year] = missing
    if gaps:
        raise ValueError(f"规范历史存在年度内部缺号：{gaps}")


def _write_csv(draws: dict[str, Draw], output: Path, source: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("期数", "开奖号码", "开奖日期", "数据来源")
        )
        writer.writeheader()
        for issue in sorted(draws, key=int, reverse=True):
            draw = draws[issue]
            writer.writerow(
                {
                    "期数": draw.issue,
                    "开奖号码": draw.number,
                    "开奖日期": draw.date,
                    "数据来源": source,
                }
            )
    temporary.replace(output)


def reconcile_fc3d(raw: Path, output: Path) -> dict[str, object]:
    """以500全国福彩3D历史为完整骨架，并用福彩官网重叠区校验。"""

    secondary = parse_500_history(raw / "fc3d_500_history.html", 7)
    official = parse_official_csv(raw / "fc3d_cwl_api.csv")
    common = set(secondary) & set(official)
    number_conflicts = sorted(
        issue for issue in common if secondary[issue].number != official[issue].number
    )
    date_conflicts = sorted(
        issue for issue in common if secondary[issue].date != official[issue].date
    )
    if set(official) - set(secondary) or number_conflicts or date_conflicts:
        raise ValueError(
            "福彩3D多源校验失败："
            f"missing={len(set(official) - set(secondary))}, "
            f"number={number_conflicts[:10]}, date={date_conflicts[:10]}"
        )
    _assert_internal_issue_continuity(secondary, 4)
    _write_csv(
        secondary,
        output,
        "500全国历史；2013起经中国福利彩票官网逐期交叉验证",
    )
    issues = sorted(secondary, key=int)
    return {
        "lottery": "fc3d",
        "rows": len(secondary),
        "issueMin": issues[0],
        "issueMax": issues[-1],
        "officialOverlap": len(common),
        "numberConflicts": 0,
        "dateConflicts": 0,
        "caveat": "2004001至2013001仅有500全国历史骨架；未使用17500的早期地方试点口径",
    }


def reconcile_pl3(raw: Path, output: Path) -> dict[str, object]:
    """以500和17500号码共识为准，日期采用三源多数。"""

    source_500 = parse_500_history(raw / "pl3_500_history.html", 5)
    source_17500 = parse_17500_history(raw / "pl3_17500.txt", 5)
    official = parse_official_csv(raw / "pl3_zhcw_api.csv")
    if set(source_500) != set(source_17500) or set(source_500) != set(official):
        raise ValueError("排列三三源期号集合不一致")
    issues = sorted(source_500, key=int)
    canonical: dict[str, Draw] = {}
    official_number_corrections: list[str] = []
    official_date_corrections: list[str] = []
    for issue in issues:
        first = source_500[issue]
        second = source_17500[issue]
        primary = official[issue]
        if first.number != second.number:
            raise ValueError(f"排列三两个独立历史源号码冲突：{issue}")
        if primary.number != first.number:
            official_number_corrections.append(issue)
        dates = Counter((primary.date, first.date, second.date))
        date, votes = dates.most_common(1)[0]
        if votes < 2:
            raise ValueError(f"排列三日期无多数共识：{issue}")
        if primary.date != date:
            official_date_corrections.append(issue)
        canonical[issue] = Draw(issue, first.number, date)
    _assert_internal_issue_continuity(canonical, 2)
    _write_csv(canonical, output, "中彩网、500、17500三源对账后的多数共识")
    return {
        "lottery": "pl3",
        "rows": len(canonical),
        "issueMin": issues[0],
        "issueMax": issues[-1],
        "threeSourceOverlap": len(canonical),
        "officialNumberCorrections": official_number_corrections,
        "officialDateCorrections": official_date_corrections,
        "secondaryNumberConflicts": 0,
    }


def main() -> int:
    """重建两个规范历史文件并写出质量报告。"""

    parser = argparse.ArgumentParser(description="离线重建完整数字彩历史")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/full_history_quality.json")
    )
    args = parser.parse_args()
    reports = [
        reconcile_fc3d(args.raw_dir, args.data_dir / "fc3d/official_history.csv"),
        reconcile_pl3(args.raw_dir, args.data_dir / "pl3/official_history.csv"),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"schemaVersion": 1, "lotteries": reports}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(reports, ensure_ascii=False))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
