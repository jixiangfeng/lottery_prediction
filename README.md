# 双色球 D8 / D8+7 研究工具

本仓库仅保留双色球的两个固定研究链：

- **D8**：8红+1蓝，展开28注，固定成本56元；
- **D8+7**：D8的28注加7张固定边界单式，共35注、70元。

D8 内部使用 B35 作为确定性去重参照，不是独立策略。

所有输出仅用于研究：不承诺中奖、不自动下单、不追号或加码。未通过独立前瞻门禁的结论保持 `uniform_abstain`。

## 常用入口

```bash
make ssq-fetch
make ssq-reconcile
make ssq-evaluate
make ssq-d8-history
make ssq-d8-official-backtest
make ci
```

## 审计原则

1. 仅使用固定官方来源；
2. 严格先预测、后开奖、再更新；
3. 使用固定成本随机基线与 proper score；
4. 不因单期结果反向调参；
5. 前瞻状态使用 SHA-256/HMAC、原子写入与逐期结算。
