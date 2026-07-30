# 双色球官方逐期奖级奖金采集

`src.analysis.ssq_prizegrades` 和 `scripts/ssq_fetch_prizegrades.py` 只调用中国福利彩票发行管理中心公开接口：

`https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq...`

接口记录的 `prizegrades` 中，奖级 1--6 是双色球六个正式奖级；官网目前还会返回一个金额和中奖注数均为空的 `type=7` 占位项，采集器只忽略该空项。任何缺失、重复或非 1--6 的非空奖级均会失败，不会猜测或补零。

## 小范围采集与续跑

```bash
uv run --python 3.11 --with-requirements requirements-dev.txt \
  python -m scripts.ssq_fetch_prizegrades \
  --start-page 1 --pages 2 \
  --cache-jsonl data/ssq/raw/prizegrades.jsonl \
  --snapshot-json data/ssq/official_prizegrades.json
```

每页最多 100 期。发生中断时以已完成的下一页重新运行，例如已完成第 1--2 页后用 `--start-page 3 --pages 2`。缓存是追加型 JSONL：每行都包含完整的官方 `raw`、完整来源 URL、UTC `fetchedAt` 和由 URL+原文计算的 SHA-256。重复抓取保留为独立审计证据；快照对同一期选择最新抓取时间的已验证记录，同一抓取时间出现冲突则拒绝。

## 校验与覆盖限制

- 仅接受 `https://www.cwl.gov.cn` 的固定 SSQ 分页 URL；校验玩法身份、数字期号和奖级 1--6。
- `typenum`（中奖注数）及 `typemoney`（单注奖金，元）必须为十进制非负整数；金额不接受浮点/带单位/空值。
- `--pages` 默认仅抓 1 页，**不代表全历史覆盖**。快照只包含已成功抓到且通过校验的期号；未抓取、接口未返回或校验失败的期号不会出现，更不会被填造。
- 分页内容会随新开奖向后移动；一次全量覆盖应在固定时间窗口从第 1 页连续采至接口末页，并保留这次缓存和抓取时间作为覆盖边界。
