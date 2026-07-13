# -*- coding: utf-8 -*-

from pathlib import Path

from src.analysis.html_report import build_html_report, write_html_report


def test_build_html_report_is_self_contained_and_styled():
    markdown = """# 快乐8每日分析报告

- 最新期号：`2026181`
- 主推荐参数：`omission_mix`

## 候选组选十

| 序号 | 号码 | 收益率 |
|---:|---|---:|
| 1 | `01 02 03 04 05 06 07 08 09 10` | -34.35% |

## 风险提示

- 不保证中奖。
"""

    html = build_html_report(markdown, title="快乐8每日分析报告")

    assert "<!doctype html>" in html.lower()
    assert "快乐8每日分析报告" in html
    assert "omission_mix" in html
    assert "<style>" in html
    assert "linear" in html.lower() or "--accent" in html
    assert "<table" in html
    assert "01 02 03" in html
    assert "https://" not in html


def test_write_html_report_creates_html_file(tmp_path):
    markdown = "# 快乐8每日分析报告\n\n- 最新期号：`2026181`\n"

    output = write_html_report(markdown, output_dir=tmp_path, issue="2026181")

    assert output == tmp_path / "html" / "kl8_daily_2026181.html"
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
