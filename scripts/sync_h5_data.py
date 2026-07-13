# -*- coding: utf-8 -*-
"""同步结构化日报数据到 Vue H5 public 目录。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_catalog(data_dir: Path) -> dict:
    reports = []
    for path in sorted(data_dir.glob('kl8_daily_*.json')):
        issue = path.stem.split('_')[-1]
        reports.append({
            'issue': issue,
            'label': f'{issue[:4]}-{issue[4:]}',
            'path': f'/report-data/{issue}.json',
        })
    if not reports:
        raise FileNotFoundError(f'未找到结构化日报 JSON：{data_dir}/kl8_daily_*.json')
    return {'latestIssue': reports[-1]['issue'], 'reports': list(reversed(reports))}


def sync_reports(reports_dir: Path, h5_public_dir: Path) -> Path:
    data_dir = reports_dir / 'data'
    catalog = build_catalog(data_dir)
    target_dir = h5_public_dir / 'report-data'
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in catalog['reports']:
        source = data_dir / f"kl8_daily_{item['issue']}.json"
        target = target_dir / f"{item['issue']}.json"
        shutil.copyfile(source, target)

    index_path = target_dir / 'index.json'
    index_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')

    latest_source = data_dir / f"kl8_daily_{catalog['latestIssue']}.json"
    latest_target = target_dir / 'latest.json'
    shutil.copyfile(latest_source, latest_target)

    walk_forward_source = data_dir / 'walk_forward_kl8.json'
    if walk_forward_source.exists():
        shutil.copyfile(walk_forward_source, target_dir / 'walk_forward_kl8.json')
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description='同步快乐8日报 JSON 到 H5 public 目录')
    parser.add_argument('--reports-dir', default='reports')
    parser.add_argument('--h5-public-dir', default='h5/public')
    args = parser.parse_args()
    target = sync_reports(Path(args.reports_dir), Path(args.h5_public_dir))
    print(target)


if __name__ == '__main__':
    main()
