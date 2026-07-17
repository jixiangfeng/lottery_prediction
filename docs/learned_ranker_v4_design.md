# 三位彩固定评分算法 v4 设计文档

## 1. 背景

当前项目已有三类三位彩候选生成方式：

- `ensemble`：多个人工统计模型做分位集成，输出候选排序。
- `probability` v2：完整枚举 `000-999`，对单个统计剖面做概率校准，守门失败则回退均匀分布。
- `online_probability` v3：均匀基线加 14 个统计专家，每期开奖后按真实号概率更新专家权重。

v3 的工程约束比较严谨，能够做到“先预测、后开奖、再反馈”，但它仍然是专家权重调节：系统先定义多个专家，再学习各专家的权重。实际 500 期评估中，福彩3D和排列三的 Log Loss 均未优于均匀基线，最终约 `96.5%` 权重回到均匀专家，说明现有专家信号没有形成稳定预测优势。

v4 的目标不是继续调专家权重，而是固定一套可解释评分公式，让历史开奖数据反推公式参数。后续每日预测时使用同一套固定算法结构，只更新开奖历史和滚动统计状态。

## 2. 目标

v4 目标：构建一个“固定评分公式 + 历史反推参数 + 严格前向验证”的三位彩候选排序算法。

具体目标：

1. 每期只使用该期之前的开奖号，禁止读取未来数据。
2. 每次预测完整枚举 `000-999`，对所有候选号码打分排序。
3. 固定评分公式结构，只训练公式参数、窗口参数、时间衰减参数。
4. 支持最近几十期更高权重，但保留中长期统计作为稳定约束。
5. 输出直选 TopK、组选 TopK、复式位置池和组选数字池。
6. 通过严格前向回测判断是否跑赢随机基线。
7. 若前向闸门失败，报告必须明确标记为“研究模式，不接入主推荐”。

非目标：

- 不承诺提高中奖概率或保证盈利。
- 不允许根据完整历史事后调参后直接宣称有效。
- 不做每期临时换规则的解释型拟合。
- 第一版不引入神经网络，避免小样本过拟合。

## 3. 核心思想

v4 不再让多个专家投票，而是固定一个候选号码评分公式：

```text
score(candidate, history_state) =
  w_position_frequency       * 位置频率分
+ w_position_omission        * 位置遗漏分
+ w_pair_frequency           * 两位组合分
+ w_shape_distribution       * 形态分布分
+ w_sum_distribution         * 和值分布分
+ w_span_distribution        * 跨度分布分
+ w_parity_bigsmall          * 奇偶大小分
+ w_recent_trend             * 短长窗口趋势分
+ w_latest_distance          * 上期距离分
+ w_repeat_latest            * 重复上期分
+ w_omission_rebound         * 遗漏回补分
+ w_constraint_penalty       * 过度集中惩罚
```

公式结构固定，训练只搜索：

- 各项权重 `w_*`
- 窗口组合
- 时间衰减半衰期
- 分数归一化方式
- 候选 TopK 和复式池生成参数

这样可以满足“根据开奖结果反推一个固定算法”的需求，同时避免每期临时生成一套解释。

## 4. 数据流

### 4.1 单期预测流程

以预测第 `T` 期为例：

1. 截断历史，只取 `1 ... T-1` 期。
2. 基于截断历史生成多窗口统计状态。
3. 枚举 `000 ... 999` 共 1000 个候选。
4. 为每个候选计算特征分量。
5. 用固定评分公式得到总分。
6. 按总分排序，输出直选 TopK。
7. 按组选数字集合汇总排列分数，输出组选 TopK。
8. 保存开奖前预测快照。
9. 等第 `T` 期开奖后，用真实号评估本期表现。

### 4.2 滚动训练流程

从第 `min_train_size + 1` 期开始做前向滚动：

```text
用 1-100 期状态预测 101 期
读取 101 期开奖结果，记录真实号排名和命中

用 1-101 期状态预测 102 期
读取 102 期开奖结果，记录真实号排名和命中

...

用 1-(N-1) 期状态预测 N 期
读取 N 期开奖结果，记录真实号排名和命中
```

训练参数时可以在训练段反复执行上述流程，但验证段和测试段必须冻结参数。

## 5. 历史窗口与近期权重

最近几十期应当更重要，但不能只看最近几十期。v4 使用两种机制表达近期权重。

### 5.1 多窗口特征

默认窗口：

```text
10, 20, 30, 50, 100, 300, all
```

每个窗口计算：

- 百、十、个位数字频率
- 百十、百个、十个组合频率
- 和值分布
- 跨度分布
- 组三、组六、豹子分布
- 奇偶比分布
- 大小比分布
- 质合比分布
- 和值尾分布
- 镜像、连号、重号分布
- 当前遗漏与窗口内最大遗漏

### 5.2 时间衰减

对历史样本引入指数衰减：

```text
sample_weight(age) = exp(-age / half_life)
```

其中 `age=1` 表示上一期，`age=30` 表示距离当前预测期 30 期。

候选半衰期：

```text
none, 20, 30, 50, 80, 100, 150, 200
```

参数选择原则：

- 如果短期确有延续性，较小 `half_life` 会在验证段胜出。
- 如果短期只是噪声，验证段会倾向较大 `half_life` 或不衰减。
- 不允许人工指定“最近 30 期一定更准”后直接上线。

## 6. 特征设计

### 6.1 候选基础特征

对候选号 `abc` 计算：

- `digit_0=a`、`digit_1=b`、`digit_2=c`
- `sum=a+b+c`
- `span=max(a,b,c)-min(a,b,c)`
- `shape=豹子/组三/组六`
- `parity_pattern=奇偶比`
- `big_small_pattern=大小比`
- `prime_composite_pattern=质合比`
- `sum_tail=sum % 10`
- `has_consecutive` 是否有连号
- `has_mirror` 是否有镜像关系
- `repeat_latest_count` 与上一期重复数字数
- `same_position_repeat_count` 与上一期同位置重复数
- `latest_l1_distance` 与上一期绝对距离和

### 6.2 多窗口统计特征

对每个窗口 `W`：

- `pos_freq_W_i_digit`：候选第 `i` 位数字在窗口内频率
- `pos_omission_W_i_digit`：候选第 `i` 位数字当前遗漏
- `pair_freq_W_ij_digits`：两位组合频率
- `sum_freq_W`：候选和值频率
- `span_freq_W`：候选跨度频率
- `shape_freq_W`：候选形态频率
- `parity_freq_W`：候选奇偶比频率
- `big_small_freq_W`：候选大小比频率
- `sum_tail_freq_W`：候选和值尾频率

频率建议做贝叶斯平滑：

```text
smoothed_rate = (count + alpha * prior) / (window_size + alpha)
```

### 6.3 短长趋势特征

趋势特征用于表达“最近是否变热”：

```text
trend_30_300 = rate_30 - rate_300
trend_50_all = rate_50 - rate_all
trend_ratio_30_300 = rate_30 / max(rate_300, epsilon)
```

对位置频率、两位组合、和值、跨度、形态都计算短长差。

### 6.4 遗漏回补特征

遗漏不直接等于应该出现，需要做分位化：

```text
omission_percentile = current_omission / historical_max_omission
omission_zscore = (current_omission - mean_omission) / std_omission
```

可设置非线性回补函数：

```text
rebound_score = log1p(current_omission) / log1p(omission_cap)
```

遗漏上限 `omission_cap` 作为可搜索参数，候选值：

```text
20, 30, 50, 80
```

## 7. 固定评分公式

第一版采用线性可解释公式：

```text
raw_score = Σ w_k * feature_k
```

为了避免某一类特征尺度过大，所有特征进入公式前做归一化：

- 频率类：转成 log probability 或 z-score。
- 遗漏类：压缩到 `[0, 1]`。
- 趋势类：按历史分布做 robust z-score。
- 惩罚类：非负分数，最后乘负权重。

候选最终概率可选：

```text
probability = softmax(raw_score / temperature)
```

其中 `temperature` 只用于排序分数转概率，不改变排序。候选值：

```text
0.1, 0.2, 0.5, 1.0, 2.0
```

## 8. 参数搜索

### 8.1 搜索参数

第一版参数空间：

- 特征权重：`w_position_frequency` 等 12-20 个权重。
- 窗口权重：`10/20/30/50/100/300/all`。
- 时间衰减半衰期：`none/20/30/50/80/100/150/200`。
- 遗漏上限：`20/30/50/80`。
- softmax 温度：`0.1/0.2/0.5/1.0/2.0`。
- 组选分数聚合方式：`sum_prob`、`max_perm`、`mean_top_perm`。

### 8.2 搜索算法

建议分两步：

1. 粗搜索：随机搜索或遗传算法，快速探索参数空间。
2. 精搜索：贝叶斯优化或局部扰动搜索，优化候选参数。

不建议第一版使用高自由度神经网络，因为历史期数少，容易拟合噪声。

### 8.3 目标函数

主目标不只看 Top10 命中，建议综合：

```text
objective =
  - 0.35 * mean_log_rank
  - 0.25 * mean_rank_percentile
  + 0.20 * direct_top10_hit_rate
  + 0.15 * group_top10_hit_rate
  + 0.05 * box_pool_coverage_rate
  - penalty_for_block_instability
```

其中：

- `mean_log_rank`：真实号排名的 log 均值，越小越好。
- `mean_rank_percentile`：真实号排名分位，越小越好。
- `direct_top10_hit_rate`：直选 Top10 命中率。
- `group_top10_hit_rate`：组选 Top10 命中率。
- `box_pool_coverage_rate`：复式数字池覆盖真实号码的比例。
- `penalty_for_block_instability`：分块表现不稳定惩罚。

## 9. 训练、验证与测试切分

不能在全部历史上反复调参后直接上线。必须分段。

默认 1000 期切分：

```text
1-500    参数搜索段
501-750  验证选择段
751-1000 冻结测试段
```

规则：

1. 参数只能在 `1-500` 上搜索。
2. 用 `501-750` 选择最终参数。
3. `751-1000` 只允许运行冻结参数，不允许再调。
4. 测试段结果是判断能否进入日报主推荐的主要依据。

如果历史期数不足 1000，可按比例切分：

```text
50% search / 25% validation / 25% frozen test
```

## 10. 前向评估闸门

v4 必须同时通过概率质量、命中率和稳定性闸门。

### 10.1 随机基线

三位彩直选 Top10 随机命中期望：

```text
10 / 1000 = 1%
```

组选 Top10 随机命中期望取决于组选候选排列数，报告中必须逐期精确计算。

### 10.2 通过条件

建议第一版闸门：

1. 测试段平均真实号排名优于均匀随机中位排名。
2. 测试段 Log Loss 不差于均匀分布，最好显著优于 `log(1000)=6.907755`。
3. 直选 Top10 命中率高于随机期望，单侧 p 值 `< 0.05`。
4. 组选 Top10 命中率高于逐期随机期望，单侧 p 值 `< 0.05`。
5. 分成至少 3 个时间块后，不能只靠一个时间块贡献全部优势。
6. 福彩3D和排列三至少一个彩种通过完整闸门；若只通过单彩种，只能对该彩种启用。

### 10.3 失败处理

如果闸门失败：

- 报告中明确写“未建立稳定预测优势”。
- 日报仍可生成研究候选，但不得标为主推荐。
- 不允许根据测试段结果继续调整参数后重新宣称通过。

## 11. 日报输出设计

新增模式：

```text
DIGIT_RANKING_MODE=learned_ranker_v4
```

日报输出目录：

```text
reports/learned_ranker_v4_daily/
```

日报内容：

- 最新开奖期号和号码。
- 使用的冻结参数指纹。
- 训练历史截止期。
- 直选 Top10：号码、总分、归一化概率、关键贡献项。
- 组选 Top10：数字集合、排列概率和、排列数。
- 复式建议：位置池、组选数字池、注数和成本。
- 近期回测摘要：最近 50/100/300 期命中情况。
- 风险提示：是否通过可行性闸门。

推荐快照：

```text
reports/picks/digit/<彩种>_learned_ranker_v4_<实验ID>_<参数指纹前缀>_<源期号>.json
```

评估报告：

```text
reports/evaluations/learned_ranker_v4_fc3d.md
reports/evaluations/learned_ranker_v4_pl3.md
reports/evaluations/learned_ranker_v4_pl5.md
```

## 12. 状态与可复现性

v4 必须保存以下内容：

- 输入 CSV SHA-256。
- 源码指纹。
- 参数搜索空间。
- 最终参数 JSON。
- 训练/验证/测试切分边界。
- 每期预测前的历史截止期。
- 每期直选候选、组选候选、真实开奖号、真实号排名。
- 随机基线和 p 值。

参数文件建议：

```text
reports/state/learned_ranker_v4/<彩种>_params.json
```

每期轨迹建议：

```text
reports/evaluations/learned_ranker_v4_<彩种>.json
```

## 13. 模块设计

建议新增模块：

```text
src/analysis/digit_learned_features.py
src/analysis/digit_learned_ranker.py
src/analysis/digit_learned_ranker_search.py
src/analysis/digit_learned_ranker_walk_forward.py
scripts/digit_learned_ranker.py
```

### 13.1 digit_learned_features.py

职责：

- 构建多窗口统计。
- 为 `000-999` 候选生成特征矩阵。
- 支持时间衰减和贝叶斯平滑。
- 禁止使用目标期及之后数据。

核心函数：

```python
build_history_state(history, rule, config) -> LearnedHistoryState
build_candidate_features(state, rule, candidates) -> DataFrame
```

### 13.2 digit_learned_ranker.py

职责：

- 固定评分公式。
- 参数加载和指纹计算。
- 直选/组选候选选择。
- 分数转概率。

核心函数：

```python
score_candidates(features, params) -> np.ndarray
select_direct_candidates(scores, top_k) -> list[DigitCandidate]
select_group_candidates(probabilities, top_k) -> list[DigitGroupCandidate]
```

### 13.3 digit_learned_ranker_search.py

职责：

- 参数搜索。
- 搜索段前向模拟。
- 验证段选择最终参数。

核心函数：

```python
search_learned_ranker_params(history, rule, search_config) -> LearnedRankerParams
```

### 13.4 digit_learned_ranker_walk_forward.py

职责：

- 冻结参数前向测试。
- 计算 Log Loss、Brier、排名、TopK 命中、组选命中。
- 生成 Markdown/JSON 评估报告。

核心函数：

```python
run_learned_ranker_walk_forward(history, rule, params, split) -> LearnedRankerReport
```

## 14. 实施计划

### 阶段一：特征矩阵与固定公式

1. 新增 `digit_learned_features.py`。
2. 实现完整 `000-999` 候选特征。
3. 新增 `digit_learned_ranker.py`。
4. 用手工默认参数生成日报候选。
5. 增加单元测试，验证无未来数据泄漏。

### 阶段二：前向回测

1. 新增 `digit_learned_ranker_walk_forward.py`。
2. 对福彩3D、排列三运行冻结参数回测。
3. 输出 Markdown/JSON 评估报告。
4. 接入随机基线和分块稳定性闸门。

### 阶段三：参数搜索

1. 实现随机搜索或遗传算法。
2. 支持训练段搜索、验证段选择、测试段冻结。
3. 输出最终参数指纹。
4. 明确记录搜索空间，保证可复现。

### 阶段四：日报集成

1. 在 `digit-report` 中新增 `learned_ranker_v4` 模式。
2. 输出日报、JSON 和推荐快照。
3. 若闸门未通过，日报显示研究模式。
4. 重复运行同一输入必须得到同一推荐快照。

## 15. 测试要求

必须新增测试：

- 特征生成只读取目标期之前历史。
- 候选空间完整覆盖 `000-999`。
- 分数排序稳定，平分时使用确定性 tie-break。
- 组选概率等于全部排列概率求和。
- 时间衰减权重随 `age` 单调下降。
- 参数指纹对配置变化敏感。
- 前向回测不会在测试段调参。
- 同一输入重复运行输出一致。

建议测试文件：

```text
tests/test_digit_learned_features.py
tests/test_digit_learned_ranker.py
tests/test_digit_learned_ranker_walk_forward.py
```

## 16. 风险与约束

### 16.1 过拟合风险

历史开奖接近随机，参数搜索很容易找到历史上好看的噪声组合。必须依赖冻结测试段和分块稳定性控制。

### 16.2 小样本风险

三位彩 1000 期历史并不算多，虽然每期有 1000 个候选，但真实正样本每期只有 1 个。不能把候选数误认为独立样本数。

### 16.3 概率解释风险

评分公式输出的概率只是归一化排序概率，不等于真实开奖概率。只有通过长期前向验证后，才可谨慎解释为模型概率。

### 16.4 策略变更风险

如果用测试段结果继续调参，测试段就变成训练数据，必须重新划分新的未来区间。文档和报告必须记录这一点。

## 17. 推荐默认配置

第一版默认配置建议：

```json
{
  "minTrainSize": 100,
  "windows": [10, 20, 30, 50, 100, 300, "all"],
  "halfLifeCandidates": [null, 20, 30, 50, 80, 100, 150, 200],
  "omissionCaps": [20, 30, 50, 80],
  "temperatures": [0.1, 0.2, 0.5, 1.0, 2.0],
  "directTopK": 10,
  "groupTopK": 10,
  "split": {
    "searchRatio": 0.50,
    "validationRatio": 0.25,
    "testRatio": 0.25
  },
  "passGate": {
    "directPValueMax": 0.05,
    "groupPValueMax": 0.05,
    "requireLogLossNotWorseThanUniform": true,
    "minStableBlocks": 2,
    "blockCount": 3
  }
}
```

## 18. 最终判断标准

v4 能否接入主推荐，不看训练段表现，只看冻结测试段：

- 冻结测试段通过概率质量和命中闸门：可作为主推荐候选。
- 只通过组选或只通过某彩种：只对通过部分启用。
- 未通过：保留为研究报告，不替代当前日报主流程。

最重要的原则：

```text
允许用历史开奖结果反推固定算法参数，
不允许用全部历史事后拟合后直接宣称未来有效。
```
