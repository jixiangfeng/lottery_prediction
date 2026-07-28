# 双色球 Challenger E1 独立诊断

## 边界

- E1 只用于回溯与当前研究，不修改 `ssq_ensemble_v1`、B、D8 的状态、协议或既有报告。
- 不开放学习率、正则、预热期、特征、随机对照数量等 CLI 调参项。
- 所有输出固定为 `formalGate=false`、`selection=null`、`formalRecommendationStatus=uniform_abstain`，不做历史晋级或自动推荐。

## 严格历史顺序

1. 将 canonical 双色球历史按数值期号升序排序并拒绝重复期号。
2. 每期更新前同时取得 E 红球概率与 incumbent 红蓝概率，并断言 E 的 33 项概率和为 6。
3. 用 incumbent 概率构造同期 B；用 incumbent 红球、同一蓝球和 B 构造 D8；用 E 红球、同一蓝球和 B 构造 E 8+1。
4. 前 120 期只预热；其后逐期评分 E/incumbent/uniform 的逐球平均红球 LogLoss/Brier，以及 E、D8、32 组固定随机 8+1 的红8交集、阈值、蓝球和红6模式。
5. 完成当期记录与指纹后，才分别更新 E 和 incumbent 状态。

## 运行

```bash
make ssq-challenger-e-v1
```

默认产物：

- `reports/research/ssq_challenger_e_v1.json`
- `reports/retrospective/ssq_challenger_e_v1_full_history.json`

历史报告包含逐期输入/构造哈希、预测与评分前缀指纹、E-D8 红8和完整票交集、E-D8 与 E-随机32差值。它们是描述性诊断，不是预测有效性或中奖概率声明。
