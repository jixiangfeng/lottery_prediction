# 彩票研究工具：双色球、超级大乐透、快乐8

本仓库只保留三个相互隔离的研究域：

- `ssq`：双色球，官方历史、严格前序评估、B/D8/E影子链；
- `dlt`：超级大乐透，官方历史、7+2固定成本Search/Validation研究；
- `kl8`：快乐8，官方历史、选4同成本A/B前瞻链及选5研究工具。

所有输出均为研究用途，不承诺中奖，不自动下单，不自动提高预算。模型未通过固定闸门时统一保持`uniform_abstain`。

## 环境

```bash
uv run --python 3.11 --with-requirements requirements-dev.txt python -V
make setup
make ci
```

## 常用入口

```bash
# 双色球
make ssq-fetch
make ssq-reconcile
make ssq-evaluate

# 超级大乐透
make dlt-fetch
make dlt-reconcile
make dlt-search
make dlt-validation

# 快乐8
make kl8-fetch
make kl8-pick4-joint-status
```

## 数据目录

```text
data/ssq/
data/dlt/
data/kl8/
```

## 审计原则

1. 仅使用固定官方来源；
2. 严格先预测、后开奖、再更新；
3. 固定成本随机基线与proper score共同否决；
4. 失败不放宽闸门，不用历史稀有命中倒推模型；
5. 前瞻状态使用SHA-256/HMAC、原子写入和逐期结算。
