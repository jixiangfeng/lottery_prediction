# 大乐透固定基数分布与 7+2 构造协议 v1

## 范围

本版本只实现无开奖结果依赖的数学核心和确定性组合构造，不包含模型训练、候选选择、Validation、Frozen 读取或成绩评估。

## 固定基数分布

对大小恰为 `K` 的集合 `S`：

```text
P(S) = (1-epsilon) * exp(sum(score_i / tau)) / Z + epsilon / C(N,K)
```

- `tau` 必须有限且大于零；`epsilon` 必须位于 `[0, 1]`。
- 所有分数必须有限。
- `Z` 使用 log-domain 基本对称多项式动态规划计算。
- 边际概率由前缀/后缀 DP 精确计算，和约束为 `K`。
- 通用数学 API 使用零基索引；DLT builder 对外使用 1–35、1–12 的彩票号码。

## 冻结构造

- 分区计数固定为 `600 / 1301 / 500 / 500`，合计 2901。
- 候选 ID 固定且恰为 4 个：`C1_LONG_RIDGE`、`C2_MULTISCALE_RIDGE`、`C3_PAIR_GRAPH_RIDGE`、`C4_EQUAL_LOGPOOL`。
- 前区按边际概率降序、号码升序打破平局，取前 7；后区同规则取前 2。
- 选中号码按升序保存；按字典序展开全部 `C(7,5)=21` 个前区组合，所有票共享同一后区 2 号码。
- 只允许基本投注，固定 21 注、42 元；不含追加。
- 输出固定声明：`researchOnly=true`、`predictionClaim=false`、`equalChanceNoEdge=true`、`formalRecommendationStatus=uniform_abstain`。
- 协议、输入边际、选中集合和完整票分别使用规范 JSON SHA-256 绑定；校验失败时关闭。

冻结协议 SHA-256：

```text
8d4fd071f6036bfb298332f90ae2413a5349a8520a46b146587bafb13ecada4e
```

该摘要变化意味着协议内容发生变化，必须创建新版本，不得覆盖 v1。
