# 快乐8全量历史预训练与特征发现结果（2026-07-23）

## 结论

福彩官网可返回的快乐8完整历史共2014期。严格隔离最新500期Frozen后，开发区仅有1514期，少于`kl8_pick5_v1`正式协议要求的1550期，因此没有伪造缺失历史或读取Frozen补足。独立v2探索流程使用前300期初始训练、714期Search和最后500期开发区Evaluation。

五个嵌套特征组和十三个独立消融组均未同时满足总体LogLoss/Brier优于均匀基线且五个连续时间块都不变差。最终选择为`uniform`，没有合法候选进入温度校准或真实Evaluation模型评估；Frozen号码未读取，未晋级、未推荐。

## 全量数据审计

| 项目 | 结果 |
|---|---:|
| 来源 | `https://www.cwl.gov.cn/`年度日期窗口 |
| 总期数 | 2014 |
| 起始 | `2020001 / 2020-10-28` |
| 截止 | `2026193 / 2026-07-22` |
| 唯一期号 | 2014 |
| 唯一日期 | 2014 |
| 每期号码 | 恰好20个唯一`1..80`号码 |
| 年内缺失期号 | 0 |
| canonical CSV模式 | `0444` |
| canonical CSV SHA-256 | `cb53ec0e9855bf72a9f7e69aeb23ccce24e199d0e411e04f221dadd32d63efb9` |
| raw JSONL SHA-256 | `7c844602f0298568ae485fe4fe6420e77f095a1d794d3c11531d9fd693d4a318` |

逐年期数：2020年65期，2021年351期，2022年351期，2023年351期，2024年352期，2025年351期，2026年至7月22日193期。

## 数据隔离

| 区域 | 边界 | 期数 | 用途 |
|---|---|---:|---|
| Initial train | 开发区前段 | 300 | 初始预训练 |
| Search | 紧接初始训练 | 714 | 特征组选择 |
| Development Evaluation | `Search`之后至`2025044` | 500 | 只评Search胜者一次 |
| Frozen | `2025045`—`2026193` | 500 | 只读取期号/日期边界，号码不解析 |

开发区canonical SHA-256：`0a7dd88c8f41805d91247281b8154317a3ddc0d8271bc4614867e324e98f6e27`。

## 特征和模型

每个目标期为号码1—80构造prior-only面板。特征只使用目标期之前的数据：

- 频率窗口：5/10/20/40/80/160/320；
- omission：原值、`log1p`、320封顶；
- lag：1/2/3/5/10；
- 趋势：10减80、20减160；
- EWMA半衰期：10/20/80/300；
- 上期上下文：原号、邻号、同尾、同十位区间；
- 最近80期pair context lift，排除self。

LightGBM固定seed和保守小树，每50期只使用此前全部面板扩展重训；每期80个预测裁剪并归一到概率和20。均匀基线固定为0.25。

## 嵌套特征搜索

`Delta = uniform - model`，正值才代表模型更好。

| 特征组 | 特征数 | ΔLogLoss | ΔBrier | Top20命中增量 | 正向时间块 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| frequency | 7 | -0.00015143 | -0.00005601 | +0.07843 | 1/5 | 淘汰 |
| frequency+omission | 10 | -0.00006122 | -0.00002179 | +0.05182 | 1/5 | 淘汰 |
| +lags/trends/EWMA | 21 | -0.00026985 | -0.00010198 | +0.01681 | 1/5 | 淘汰 |
| +previous context | 26 | -0.00027354 | -0.00010322 | -0.04622 | 0/5 | 淘汰 |
| full + pair context | 27 | -0.00018126 | -0.00006857 | +0.06443 | 1/5 | 淘汰 |

所有组的proper scores总体均比均匀基线更差，且没有一组满足五块稳定性。

## 独立消融

| 单独特征组 | ΔLogLoss | ΔBrier | Top20命中增量 | LogLoss正向块 | 结论 |
|---|---:|---:|---:|---:|---|
| frequency320 | -0.00000312 | -0.00000157 | -0.07423 | 3/5 | 最接近零，但无总体增益且排序更差 |
| frequency5 | -0.00001477 | -0.00000583 | +0.01120 | 2/5 | 不稳定，淘汰 |
| frequency80 | -0.00003677 | -0.00001380 | +0.09384 | 2/5 | 排名局部改善但proper scores失败，淘汰 |
| frequency40 | -0.00005309 | -0.00001955 | -0.04762 | 3/5 | 淘汰 |
| omission | -0.00005530 | -0.00002095 | -0.03081 | 1/5 | 淘汰 |
| frequency10 | -0.00007083 | -0.00002599 | -0.09944 | 0/5 | 淘汰 |
| lags | -0.00009421 | -0.00003572 | +0.03221 | 2/5 | 淘汰 |
| frequency160 | -0.00010246 | -0.00003843 | -0.01120 | 0/5 | 淘汰 |
| previous context | -0.00013271 | -0.00005005 | -0.00420 | 0/5 | 淘汰 |
| pair context | -0.00014308 | -0.00005340 | -0.08263 | 0/5 | 淘汰 |
| frequency20 | -0.00017069 | -0.00006304 | -0.07843 | 0/5 | 淘汰 |
| trends | -0.00021612 | -0.00008123 | -0.02661 | 1/5 | 淘汰 |
| EWMA | -0.00022161 | -0.00008328 | +0.04482 | 0/5 | 淘汰 |

`frequency80`和EWMA虽在Top20命中上出现小幅正值，但LogLoss、Brier和跨块稳定性同时失败，不能作为可用预测信号。`frequency320`只能作为近均匀观察项，不应启用。

## Feature gain解释

完整模型内部gain较高的特征包括`frequency320`、`pairContextLift80`、`ewma300`、`frequency160`和两个趋势项。这些只表示模型在树分裂中使用过它们，不代表样本外有效。完整模型总体和四个时间块的proper scores均失败，因此不保留任何特征为生产信号。

## 最终状态

```text
selectedFeatureSet=uniform
selectionReason=no_eligible_feature_set
evidenceStatus=exploratory_feature_discovery_only
frozenRead=false
promotionPassed=false
recommendationEnabled=false
userVisibleCandidates=[]
```

由于Search没有合法胜者，温度校准无候选可执行，Development Evaluation保持均匀基线：500期Top20平均命中`5.022`，LogLoss `0.5623351446`，Brier `0.1875`。

## 产物

- `data/kl8/raw/history.jsonl`
- `data/kl8/kl8.csv`
- `reports/development/kl8_feature_discovery_v2_official_20260723.json`
- `reports/development/kl8_feature_ablation_v2_official_20260723.json`

嵌套报告SHA-256：`d6a0cc6e7c7b5d00ff423300837d05e1cda67bf2d9f6d87661584cb7281a23e8`。

消融报告SHA-256：`85df910a129722c1dcb72368a1c1f19166b71f6b3ae4eb5d1dd1f9253a5d4e08`。

源码指纹：`c111db1e5f3da6ad82c5e167327b5c09f1710849af5629334d441b342e2e464b`。

## 验证

- v2专项测试：7 passed；
- KL8与v2聚焦测试：全部通过；
- 完整`make ci`：297 passed；
- 总覆盖率：86.28%；
- Black、isort、flake8、mypy、compileall全部通过。

本次没有运行正式5000次null，没有读取Frozen号码，没有输出推荐号码。正式v1仍需等待至少36期新增官方数据后才能满足1550开发期+500 Frozen的固定协议。
