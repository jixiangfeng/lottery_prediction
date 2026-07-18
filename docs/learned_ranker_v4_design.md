# learned ranker 设计

## 目标

只保留一套可审计的三位彩研究架构：完整枚举 `000–999`，使用目标期以前的数据生成特征，以固定参数评分，并严格隔离 Search、Validation 与 Frozen Test。

历史 v1、概率 v2、在线概率 v3 不再是兼容层、专家特征或运行入口。

## 支持范围

- 支持：福彩3D、排列三；
- 拒绝：排列五；
- 所有模式只读本地 CSV，抓取必须通过独立显式命令。

## 数据边界

预测目标索引 `t` 时，只允许读取 `[0, t)`：

```text
Search      参数探索
Validation  参数选择
Frozen Test 锁定后一次性评估
```

Frozen Test 不能参与：

- 特征增删；
- 窗口与权重；
- half-life、alpha、omission cap、temperature；
- 目标 profile；
- 候选预算；
- gate 或激活阈值。

## 特征

当前原生特征包括：

- 位置、二位组合、和值、跨度、形态与奇偶大小；
- 多窗口频率和遗漏；
- 30–300、50–all 与 30/300 趋势差；
- 50/100/150 相对全历史的 regime gap；
- 最近一期距离、重复关系和遗漏回弹；
- 可审计的 soft constraint penalty。

全历史仅作为弱长期先验；不再包含任何 legacy ensemble 特征。

## 参数与搜索

参数产物覆盖：

- 完整特征权重；
- 特征窗口、窗口权重、alpha、half-life、omission cap；
- temperature；
- 直选/组选/位置池固定成本；
- group aggregation；
- 数据、源码、参数和产物指纹；
- 切分边界和 `testSegmentUsedForSelection=false`。

开发期可使用命中率 profile，但最终 Frozen Test 前必须锁定。

## 候选预算曲线

同一套排序可同时统计：

- 直选 Top10/20/50/100/250/500/700/900/990/1000；
- 组选 Top10/20/50/100/150/220；
- 位置池 3/5/7/10。

命中率必须与同成本随机基线一起报告；扩大候选数本身不构成模型提升。

## 冻结证据

评估产物必须验证：

- rule code；
- params fingerprint；
- params artifact fingerprint；
- source fingerprint；
- frozen canonical data fingerprint；
- report fingerprint；
- test segment 未用于选择。

允许在 `split.testEnd` 之后追加新开奖，但修改冻结前缀必须使证据失效。

## 激活

直选和组选分项 gate 独立；公共概率/稳定性 gate 失败时全部保持研究模式。未通过分项不得进入主推荐。

## 风险

彩票开奖结果高度随机。增加特征、参数或模型复杂度不等于提高未来命中率；任何历史结果都不保证中奖或盈利。
