# 数字彩模型统一证据总账

> 固定成本直选Top50；历史结果只允许淘汰，不允许按局部高点挑赢家。

## 直选Top50可比证据

| 目标 | 模型 | 彩种 | 期数 | 命中 | 命中率 | 原始p | Holm校正p | 概率质量 | 稳定块 | 联合通过块 | 决策 |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| direct_top50 | behavioral_context_v2 | fc3d | 6500 | 323 | 4.969% | 0.5533 | 1.0000 | worse_than_core | — | 0 | closed |
| direct_top50 | behavioral_context_v2 | pl3 | 6500 | 333 | 5.123% | 0.3322 | 1.0000 | worse_than_core | — | 0 | closed |
| direct_top50 | behavioral_context_v3 | fc3d | 6500 | 333 | 5.123% | 0.3322 | 1.0000 | worse_than_core | 8 | 0 | closed |
| direct_top50 | behavioral_context_v3 | pl3 | 6500 | 331 | 5.092% | 0.3742 | 1.0000 | worse_than_core | 8 | 0 | closed |
| direct_top50 | behavioral_context_v4 | fc3d | 6500 | 314 | 4.831% | 0.7420 | 1.0000 | worse_than_core | 5 | 0 | closed |
| direct_top50 | behavioral_context_v4 | pl3 | 6500 | 362 | 5.569% | 0.0202 | 0.2419 | worse_than_core | 11 | 0 | closed |
| direct_top50 | learned_ranker_v4_historical_blocks | fc3d | 7000 | 346 | 4.943% | 0.5945 | 1.0000 | joint_gate_failed | 0 | 0 | closed |
| direct_top50 | learned_ranker_v4_historical_blocks | pl3 | 7000 | 367 | 5.243% | 0.1824 | 1.0000 | joint_gate_failed | 0 | 0 | closed |
| direct_top50 | lightgbm_three_position | fc3d | 6000 | 282 | 4.700% | 0.8639 | 1.0000 | worse_than_uniform | — | 0 | closed |
| direct_top50 | lightgbm_three_position | pl3 | 6000 | 294 | 4.900% | 0.6471 | 1.0000 | worse_than_uniform | — | 0 | closed |
| direct_top50 | rank_ftrl_v4_1 | fc3d | 7000 | 359 | 5.129% | 0.3183 | 1.0000 | joint_gate_failed | 8 | 1 | closed |
| direct_top50 | rank_ftrl_v4_1 | pl3 | 7000 | 368 | 5.257% | 0.1684 | 1.0000 | joint_gate_failed | 9 | 0 | closed |

## 覆盖不足或事后形成、不得进入可比排名的证据

| 模型 | 彩种 | 期数 | 命中 | 原因 | 决策 |
|---|---|---:|---:|---|---|
| behavioral_context_v1 | fc3d | 500 | 28 | 只覆盖单个复用开发块，未覆盖全部固定块 | closed |
| behavioral_context_v1 | pl3 | 500 | 32 | 只覆盖单个复用开发块，未覆盖全部固定块 | closed |
| rank_ftrl_v4_1_drop_sum_distribution | fc3d | 7000 | 379 | 事后归因形成的单彩种候选，pl3没有共享删除候选 | future_only_not_activated |

## 不同成本、不可与直选Top50混排的诊断

| 目标 | 彩种 | 期数 | 命中 | 基线率 | p值 | 决策 |
|---|---|---:|---:|---:|---:|---|
| group_projection_top50 | fc3d | 7000 | 1349 | 18.730% | 0.1242 | closed |
| group_projection_top50 | pl3 | 7000 | 1215 | 17.350% | 0.4990 | closed |
| independent_group_top10 | fc3d | 7000 | 445 | 6.000% | 0.1095 | closed |
| independent_group_top10 | pl3 | 7000 | 418 | 6.000% | 0.5472 | closed |

## 最终决策

- **无模型入选**：全部模型未共同通过LogLoss、Brier、Top50与时间稳定性。
- 生产模式：`uniform_abstain`。
- 行为v1～v4：封存为负证据，不再在同一历史调参。
- 核心前瞻检查点：50/100/200期；50期只允许提前淘汰。
- 用户可见候选保持为空。
