# 快乐8 Pick4 同成本联合概率 A/B v1

## 研究边界

`kl8_pick4_joint_portfolio_v1` 与 `kl8_pick4_joint_ab_v1` 是完全隔离的研究挑战器。它们不修改 `kl8_pick4_rank_challenger` v2 的模型、参数、报告或状态，不改变当前 prospective 状态、生产输出或其他彩票，也不会自动联网、抓取或初始化真实状态。

控制 A 精确复用当前 v2 预测流程与 `ranked_pick4_portfolio(scores)`：五张互不重叠的 Pick4 票覆盖分数排序 Top20。挑战 B 锁定相同 Top20 并集、相同五票成本和每票四个号码，只允许在两张票之间交换一个号码。

## 联合概率模型

输入是 `kl8_pick4_prospective` 现有审计流程生成的 `probabilities80`，要求逐项严格 `0<p<1` 且总和为 20。先转换为赔率：

```text
w_i = p_i / (1 - p_i)
```

随后定义恰选 20 个号码的条件 Poisson 分布：

```text
P(S) = product(w_i, i in S) / e_20(w_1, ..., w_80),  |S| = 20
```

对任意四号码票 `T`，恰中 `k` 个号码的概率为：

```text
P(K=k) = e_k(w_T) * e_(20-k)(w_not_T) / e_20(w_all),  k=0..4
```

实现使用 log-domain 初等对称多项式动态规划，避免直接乘积溢出/下溢，并对每张票验证 `P(0)+...+P(4)=1`。均匀赔率时结果精确退化为 `Hypergeometric(N=80, K=20, n=4)`。

**重要限制**：输入 `probabilities80` 是已审计边际；条件化后的联合模型不保证逐项保持这些输入边际。快照同时保存 `auditedMarginals80` 和 A/B 每张票的联合模型命中 PMF，不作边际保持声明。

## 固定挑战目标

从控制票组开始，枚举所有两票之间的一对号码交换。每轮选择目标严格最佳的候选，目标按以下顺序作字典序最大化：

1. 五票 `P(exact4)` 之和；
2. 五票 `P(atLeast3)` 之和；
3. 五票 `P(atLeast2)` 之和。

目标完全固定且无权重。相同目标候选按规范化排序票组 tuple 决胜。只有目标严格改善才接受交换，直到不存在更优交换。原子进度事件只在接受一次交换后输出。该目标只是 payout surrogate，不是官方奖金、收益率或盈利声明。

## 500 期前瞻协议

- 固定状态根：`state/kl8_pick4_joint_ab_v1`。
- 固定恰好 500 个未来已结算开奖，分成五个固定 100 期块。
- 初始化目标只能是 canonical 最新可见开奖的下一期，并必须在目标日 `21:30+08:00` 前完成。
- `observed=0` 只表示已创建本地文件，不是任何前瞻证据；真实初始化必须面对真正未开奖的未来目标。
- 每期只允许 `step` 一次；目标尚不可用时返回 no-op，重复调用幂等。
- 若 canonical 一次出现超过一个尚未逐期锁定的开奖，拒绝 backfill/catch-up。
- 每个版本只追加，使用临时目录、文件 `fsync`、原子 rename 与目录 `fsync` 发布。
- JSON artifact 使用稳定 SHA-256 和 HMAC-SHA256；模型字节另行绑定 SHA/HMAC。CLI 要求密钥位于 state 外且权限为 `0600` 或更严格。
- 第 500 期只结案，不生成第 501 期快照；禁止调参、重试、重置或自动生产激活。

每个预开奖快照保存完整分数排名、`probabilities80`、Top20 并集、A/B 锁定票、逐票精确 PMF、联合目标及差值、最新可见期号与结果、数据/源码/模型/代码/协议哈希、前一快照哈希、创建时间和 32 个预先锁定的随机同成本基准票组。`formalCandidates=[]`、`productionActivation=false` 恒定。

每个结算记录保存 A/B 逐票命中向量、最佳票 `atLeast2/atLeast3/exact4`、总命中数、exact4 票数、固定 realized payout surrogate（exact2=1、exact3=2、exact4=3）及 B-A 配对差。因为并集相同，A/B 总命中数必须完全相等。

## 状态与闸门

`status` 报告：

- A/B 的 `atLeast2`、`atLeast3`、`exact4` 率、平均 exact4 票数和平均 payout surrogate；
- B-A 配对差；
- 三个二元终点的单侧精确 McNemar 等价二项检验；
- payout surrogate 非零配对差的单侧精确 sign test；
- 四项配对 p 值的 Holm 校正；
- 32 个同并集、同票数、同票面大小的固定随机基准；
- 五个固定 100 期块。

恰好 500 期后，预声明 gate 同时要求：B 的 exact4 与 atLeast3 严格改善、atLeast2 不恶化、平均配对 payout surrogate 为正、四项 Holm 校正 p 值均不超过 0.05，且五个块中四项配对差均非负。即使全部通过，也只返回 `pending-human-review`，绝不产生正式候选或自动激活。

## 操作命令

以下命令仅供未来人工、开奖前操作；本次实现任务不得执行真实初始化：

```bash
make kl8-pick4-joint-ab-initialize KL8_PICK4_JOINT_AB_GENERATE_KEY=1
make kl8-pick4-joint-ab-step
make kl8-pick4-joint-ab-status
```

CLI 只有 `initialize`、`step`、`status` 三个动作，以及 CSV、state、密钥三个路径选项；初始化额外允许首次生成本地密钥。不存在目标期、模型、目标函数、票组、权重或搜索网格覆盖参数。

CLI 会在 state 同级固定写入 `<state-dir>.progress.json`。每次更新都通过同目录临时文件、文件 `fsync`、`os.replace` 与目录 `fsync` 原子发布，记录命令开始/失败/完成以及票组优化开始、每次接受换票和优化完成事件。
