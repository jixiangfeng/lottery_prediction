# 双色球 Challenger E2 冻结有限选择研究

## 研究边界

E2 是独立研究挑战器，不修改 E1、B、D8、`ssq_ensemble_v1`、真实状态、密钥或定时任务。它不自动晋级，所有报告固定声明：

- `researchOnly=true`
- `formalGate=false`
- `formalRecommendationStatus=uniform_abstain`
- `automaticPromotion=false`
- 即使 Validation 合格，也必须另建未来前瞻链并经人工决定；本阶段不实现 HMAC。

## 导入即冻结的协议

历史必须是按数值期号排序、无重复的官方 2043 期：

| 分区 | 固定期数 | Python 索引 |
|---|---:|---:|
| Warmup | 120 | `[0, 120)` |
| Search | 923 | `[120, 1043)` |
| Validation | 500 | `[1043, 1543)` |
| Diagnostic | 500 | `[1543, 2043)` |

候选恰好八个，CLI 不开放特征、L2、分区、重试或候选覆盖：

- `F0_L001`, `F0_L010`: EWMA30、EWMA120
- `F1_L001`, `F1_L010`: F0 + gap、prior-repeat
- `F2_L001`, `F2_L010`: F1 + pair-affinity
- `F3_L001`, `F3_L010`: F2 + `trend=x_EWMA30-x_EWMA120`
- 每组的 L2 分别固定为 0.01 和 0.10。

全部候选复用 E1 的先验特征方程、33 个零均值固定效应、逐期基数校准、AdaGrad `eta=0.05`、累加器初值 `1e-6` 和 `epsilon=1e-8`。每期严格执行 `predict-lock-score-update`，红球概率和为 6，概率并列按球号升序。

## Validation 选择

每个候选持续在线训练至 Validation 结束。每个 Validation 期都用严格前序 `FixedEnsembleState` 重建 incumbent 红蓝概率、B 与同成本 D8，并构建候选的 8 红 + 1 蓝、28 注组合。

候选只有同时满足以下条件才合格：

1. 逐球平均 LogLoss 严格小于 uniform；
2. Brier 不大于 uniform；
3. 候选减 D8 的实际红 8 命中均值严格大于 0；
4. 五个连续 100 期块至少四块差值非负；
5. 任一块差值均不低于 -0.05。

红 5 命中率只作 secondary，红 6 只记录，二者绝不参与选择。合格者按 Validation LogLoss 升序、Brier 升序、红 8 差值降序、候选 ID 升序确定唯一候选。没有合格候选时必须 `rejected`，不允许事后放宽。

只有选中候选才报告最新 500 期 Diagnostic，且明确 `promotionEvidence=false`。拒绝时当前报告仍安全生成，但 `currentTargetGroup=null`，不会生成目标组或前瞻状态。

## 运行与产物

```bash
make ssq-challenger-e2-v1
```

固定产物：

- `reports/retrospective/ssq_challenger_e2_selection_v1.json`
- `reports/research/ssq_challenger_e2_v1.json`

选择报告包含八候选完整 Validation 指标、五块硬闸、实际分区期号、数据/输入/协议/逐期审计摘要和报告哈希。当前报告只在已选时从第一期重放该候选至第 2043 期，并复用 incumbent 当前蓝球与 B 构建零完整票重叠的独立 8+1；运行前后校验 ensemble 文件字节和内嵌报告哈希不变。
