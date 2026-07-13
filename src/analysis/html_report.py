# -*- coding: utf-8 -*-
"""快乐8 Markdown 日报转静态 HTML。

生成的 HTML 自包含 CSS，不依赖 CDN 或外部服务，适合本地离线打开或作为文件发送。
视觉风格参考 Linear 深色仪表盘：低亮度背景、半透明卡片、紫色强调色和紧凑表格。
"""

from __future__ import annotations

import html
import re
from pathlib import Path


_STYLE = r"""
:root {
  --bg: #08090a;
  --panel: rgba(255,255,255,0.035);
  --panel-2: rgba(255,255,255,0.055);
  --border: rgba(255,255,255,0.08);
  --border-soft: rgba(255,255,255,0.05);
  --text: #f7f8f8;
  --muted: #8a8f98;
  --muted-2: #62666d;
  --accent: #7170ff;
  --accent-bg: rgba(113,112,255,0.13);
  --good: #10b981;
  --bad: #ef4444;
  --warn: #f59e0b;
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0;
  color: var(--text);
  background:
    radial-gradient(circle at 15% -10%, rgba(113,112,255,0.22), transparent 34rem),
    radial-gradient(circle at 85% 0%, rgba(16,185,129,0.10), transparent 30rem),
    var(--bg);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-feature-settings: "cv01", "ss03";
  line-height: 1.55;
}
.page { max-width: 1180px; margin: 0 auto; padding: 40px 20px 72px; }
.hero {
  padding: 28px;
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  box-shadow: 0 24px 80px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  margin-bottom: 22px;
}
.kicker { color: var(--accent); font-size: 13px; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }
h1 { font-size: clamp(32px, 5vw, 56px); line-height: 1; letter-spacing: -1.2px; margin: 12px 0 10px; font-weight: 650; }
.subtitle { color: var(--muted); margin: 0; max-width: 760px; }
h2 {
  margin: 26px 0 14px;
  padding-top: 18px;
  border-top: 1px solid var(--border-soft);
  font-size: 22px;
  letter-spacing: -.3px;
}
h3 { font-size: 18px; margin: 20px 0 10px; }
p, li { color: #d0d6e0; }
ul { padding-left: 20px; }
code {
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  color: #eef0ff;
  background: rgba(113,112,255,0.12);
  border: 1px solid rgba(113,112,255,0.22);
  padding: 2px 6px;
  border-radius: 7px;
  font-size: .92em;
}
.section {
  margin: 18px 0;
  padding: 20px;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
}
table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 14px;
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  margin: 14px 0 6px;
}
th, td {
  padding: 10px 11px;
  border-bottom: 1px solid rgba(255,255,255,0.055);
  text-align: left;
  vertical-align: top;
  font-size: 14px;
}
th { color: #f7f8f8; background: rgba(255,255,255,0.055); font-weight: 620; }
td { color: #d0d6e0; }
tr:hover td { background: rgba(255,255,255,0.025); }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0; }
.metric { border: 1px solid var(--border); border-radius: 16px; padding: 14px; background: var(--panel-2); }
.metric .label { color: var(--muted); font-size: 12px; }
.metric .value { font-size: 22px; margin-top: 4px; font-weight: 650; }
.bad { color: var(--bad); }
.good { color: var(--good); }
.warn { color: var(--warn); }
.num-line code { word-spacing: .25rem; }
.footer { color: var(--muted-2); font-size: 12px; margin-top: 32px; text-align: center; }
@media (max-width: 760px) {
  .page { padding: 22px 12px 44px; }
  .hero { padding: 20px; border-radius: 18px; }
  table { display: block; overflow-x: auto; white-space: nowrap; }
  th, td { font-size: 13px; }
}
"""


def _inline(text: str) -> str:
    """处理 Markdown 行内代码与基础转义。"""

    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", escaped)


def _format_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        # 跳过 Markdown 分隔行
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    headers = rows[0]
    body = rows[1:]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{_inline(cell)}</th>" for cell in headers)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend(f"<td>{_inline(cell)}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def markdown_to_html(markdown: str) -> str:
    """把项目日报使用的 Markdown 子集转换为 HTML。"""

    lines = markdown.splitlines()
    output: list[str] = []
    in_list = False
    in_section = False
    index = 0

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            close_list()
            index += 1
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            close_list()
            output.append(_format_table(table_lines))
            continue
        if stripped.startswith("# "):
            close_list()
            if in_section:
                output.append("</section>")
                in_section = False
            output.append(f"<h1>{_inline(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            close_list()
            if in_section:
                output.append("</section>")
            output.append(f"<section class=\"section\"><h2>{_inline(stripped[3:])}</h2>")
            in_section = True
        elif stripped.startswith("### "):
            close_list()
            output.append(f"<h3>{_inline(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{_inline(stripped[2:])}</li>")
        else:
            close_list()
            output.append(f"<p>{_inline(stripped)}</p>")
        index += 1
    close_list()
    if in_section:
        output.append("</section>")
    return "\n".join(output)


def _extract_issue(markdown: str) -> str:
    match = re.search(r"最新期号：`([^`]+)`", markdown)
    return match.group(1) if match else "unknown"


def build_html_report(markdown: str, title: str = "快乐8每日分析报告") -> str:
    """生成自包含 HTML 报告。"""

    issue = _extract_issue(markdown)
    body = markdown_to_html(markdown)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - {html.escape(issue)}</title>
  <style>{_STYLE}</style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="kicker">KL8 LOCAL ANALYTICS · HTML REPORT</div>
      <h1>{html.escape(title)}</h1>
      <p class="subtitle">静态离线 HTML，可直接浏览器打开。所有统计和候选仅作历史数据分析与娱乐参考，不保证中奖。</p>
    </header>
    {body}
    <div class="footer">Generated by kl8-lottery-analyzer · self-contained HTML · no external CDN</div>
  </main>
</body>
</html>
"""


def write_html_report(markdown: str, output_dir: Path | str, issue: str | None = None) -> Path:
    """将 Markdown 日报同时写成 HTML 文件。"""

    target_issue = issue or _extract_issue(markdown)
    directory = Path(output_dir) / "html"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"kl8_daily_{target_issue}.html"
    output.write_text(build_html_report(markdown), encoding="utf-8")
    return output
