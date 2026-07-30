# 彩票研究工具：双色球D8、超级大乐透

本仓库保留两个相互隔离的研究域：

- `ssq`：双色球D8（8红+1蓝）、官方历史、严格前序评估与独立HMAC前瞻链；另有不创建前瞻状态的D8 v2 ML challenger历史诊断；
- `dlt`：超级大乐透，7+2固定成本Search/Validation研究。

D8内部生成B35参照票以执行零完整票重叠约束；该参照不是独立策略、没有独立前瞻链或命令入口。

所有输出均为研究用途，不承诺中奖，不自动下单，不自动提高预算。模型未通过固定闸门时统一保持`uniform_abstain`。

## 环境

```bash
uv run --python 3.11 --with-requirements requirements-dev.txt python -V
make setup
make ci
```

## 常用入口

```bash
# 双色球D8
make ssq-fetch
make ssq-reconcile
make ssq-evaluate
make ssq-d8-history

# 超级大乐透
make dlt-fetch
make dlt-reconcile
make dlt-search
make dlt-validation
```

## 数据目录

```text
data/ssq/
data/dlt/
```

## 审计原则

1. 仅使用固定官方来源；
2. 严格先预测、后开奖、再更新；
3. 固定成本随机基线与proper score共同否决；
4. 失败不放宽闸门，不用历史稀有命中倒推模型；
5. 前瞻状态使用SHA-256/HMAC、原子写入和逐期结算。
