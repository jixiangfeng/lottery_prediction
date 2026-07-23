# 快乐8选5概率预测设计方案

> 适用范围：快乐8选5。  
> 当前状态：`kl8_pick5_v1` 已实现为开发挑战器，不接入日报、正式推荐或任何历史 Frozen Test。  
> 证据边界：历史数据只能用于淘汰方案，不能直接证明新模型有效。

## 1. 结论先行

快乐8选5不复用福彩3D/排列三的三位数模型。首版采用以下最小方案：

```text
80个号码边际概率专家
  -> 在线自适应指数加权
  -> 边际概率校准
  -> 小规模Top5组合生成器
  -> 单号概率指标 + 选5命中指标双口径评估
  -> 严格统计闸门
  -> 开发挑战器模式
```

首版专家固定为：

1. `UniformExpert`：每个号码入选概率固定为 `20/80=0.25`，永久基线。
2. `EwmaShortExpert`：单号短期 EWMA，固定半衰期 `h=20`。
3. `EwmaMidExpert`：单号中期 EWMA，固定半衰期 `h=80`。
4. `EwmaLongExpert`：单号长期 EWMA，固定半衰期 `h=300`。
5. `OmissionShrinkExpert`：遗漏回补专家，强收缩、独立验证其是否有效。
6. `PairwiseAdjustedExpert`：基于二号共现的轻量边际修正专家，不直接建五号组合模型。

首版明确不做：

- 不直接枚举 `C(80,5)=24,040,016` 个五号组合训练排序模型。
- 不迁移福彩3D/排列三的和值、跨度、形态、位置等特征。
- 不上 LightGBM、深度学习或大规模窗口搜索作为首版。
- 不根据历史单块漂亮结果调整窗口、权重或组合规则。
- 不输出正式推荐；未通过准入时只能输出空候选或研究观察。
- 不把 smoke、小样本回测或随机模拟不足次数写成已通过。

## 2. 问题定义

快乐8每期从 `1..80` 中开出 20 个号码。选5任务不是 80 选 1，也不是 1000 类单标签分类，而是固定 20 个正例的多标签问题。

每期目标记为：

\[
Y_t \subset \{1,\ldots,80\},\quad |Y_t|=20
\]

模型在开奖前输出每个号码的边际入选概率：

\[
p_t(i)=P(i\in Y_t),\quad i\in\{1,\ldots,80\}
\]

并约束：

\[
0\le p_t(i)\le1,\qquad \sum_i p_t(i)\approx20
\]

选5候选组合由边际概率和轻量 pairwise 调整生成，不把五号组合作为直接训练类别。

## 3. 专家设计

### 3.1 UniformExpert

Uniform 是永久随机基线：

\[
p_t(i)=0.25
\]

如果复杂专家不能稳定超过 Uniform，模型必须回退为空候选，不允许用固定数字顺序或随机顺序伪装成预测。

### 3.2 EWMA单号专家

每个号码维护指数衰减出现率：

\[
S_{i,t}=\rho S_{i,t-1}+(1-\rho)I(i\in Y_t),\quad \rho=2^{-1/h}
\]

首版只固定三个半衰期：

```text
h=20 / 80 / 300
```

每个 EWMA 专家输出 80 个边际概率，并做均值校准，使概率总量接近 20。

### 3.3 OmissionShrinkExpert

遗漏专家使用号码距离上次出现的期数 `gap_i`，但必须强收缩到 Uniform：

```text
raw_score_i = bounded_transform(gap_i)
p_i = shrink_to_total_20(raw_score_i, shrinkage >= 0.8)
```

遗漏不是默认有效信号。该专家的目的主要是让系统有机会证明“遗漏无效”，而不是把遗漏作为先验收益来源。

### 3.4 PairwiseAdjustedExpert

维护号码 pair 的 EWMA 共现强度：

\[
C_{i,j,t}=\rho C_{i,j,t-1}+(1-\rho)I(i\in Y_t,j\in Y_t)
\]

Pairwise 只允许做轻量边际修正：

```text
base_i = EWMA80单号边际
selected_context = 开奖前EWMA80最高的20个号码
lift_ij = clip(C_ij / max(base_i * base_j, epsilon) - 1)
pair_bonus_i = clip(mean(lift_ij for j in selected_context if j != i))
p_i = normalize_sum20(base_i * exp(pair_marginal_weight * pair_bonus_i))
```

上下文、边际和pair矩阵都只能来自上一期更新结束后的状态，不读取当前期开奖。该定义按关联对象身份响应，不使用固定20-of-80下会退化为边际EWMA仿射变换的整行均值。首版不建立三号、四号或五号联合记忆，避免在巨大组合空间追逐噪声。

## 4. 在线聚合

每个专家在开奖前输出 80 个边际概率。聚合器使用在线 Hedge：

```text
expert probabilities -> clipped Bernoulli losses -> exponential weights -> mixed probabilities
```

损失使用开奖前的 80 维 Bernoulli LogLoss 或 Brier，并按每期 20 个正例统一计算。专家权重只允许由目标期之前的损失更新。

聚合后必须重新校准：

```text
clip to [epsilon, 1-epsilon]
normalize expected positives to 20
```

## 5. Top5组合生成

组合生成分两层：

1. 从 80 个号码中取边际概率 TopN 作为候选池，首版固定 `N=20`。
2. 在候选池内搜索 Top5 组合，评分为：

```text
sum(logit(p_i))
  + pairwise_weight * sum(pair_score_ij)
  - concentration_penalty
```

首版固定：

```text
候选池N=20
输出Top5组合数量=5
pairwise_weight很小且预注册
concentration_penalty只用于风险分散，不写成预测增益证据
```

组合评分只用于生成选5候选；概率质量评估仍使用完整 80 个边际概率。

## 6. 评估指标

### 6.1 边际概率指标

每期对 80 个号码计算：

- Bernoulli LogLoss；
- Brier；
- 相对 Uniform 的 Delta LogLoss / Delta Brier；
- 校准后的期望正例数偏差。

### 6.2 选5业务指标

随机选5命中数服从超几何分布：

\[
X\sim Hypergeometric(N=80,K=20,n=5)
\]

随机期望：

\[
E[X]=5\times\frac{20}{80}=1.25
\]

首版至少报告：

- 平均命中数；
- 命中 `0/1/2/3/4/5` 的频率；
- 命中 `>=3`、`>=4`、`=5` 的频率；
- 相对超几何基线的 p 值；
- 分块稳定性。

边际概率指标和选5命中指标必须分开计算、联合否决。任何一项硬门槛失败都不能晋级。

## 7. 数据分段

首版开发协议建议：

```text
最早300期             仅初始化专家
随后历史              严格逐期预测和机械更新
开发Search 500期      固定结构压力测试，不继续挑专家
Calibration 250期     只选择边际概率校准参数
开发Evaluation 500期  最终开发区压力测试
最新Frozen 500期      完全排除，不读取
```

历史开发区只能标记为：

```text
evidenceStatus=exploratory_reused_development
validationOpened=false
promotionPassed=false
recommendationEnabled=false
```

## 8. 随机模拟

正式晋级前必须做全流程均匀随机模拟：

```text
随机生成快乐8历史
  -> 完整Search/Calibration/Evaluation
  -> Top5组合生成
  -> 概率与命中联合闸门
```

绝对下限固定为源码字面常量`FORMAL_MIN_ITERATIONS=5000`，配置构造同时拒绝`required_null_iterations<5000`。正式协议、报告与null只接受逐字段等于默认值的唯一规范配置，并固定排除500期Frozen。报告至少包含：

- 随机数据通过全部开发门槛的比例；
- 达到真实 Delta LogLoss / Delta Brier 的比例；
- 达到真实每票平均命中、五票投资组合平均总命中，以及最佳票`>=3`、`>=4`和`=5`比例的次数；
- UniformExpert 和复杂专家最终权重分布；
- 全流程经验 p 值。

每个目标期的五张票必须记录长度5的命中数组、总命中和最佳票命中。匹配成本基线按80个号码在五票中的重数`0..5`分组，用动态规划精确抽取20个号码得到单期总命中PMF，再跨期卷积计算右尾；组合重叠不得被当作五次独立单票。

`nullSimulationPassed`要求联合开发闸门经验假阳性率不高于`alpha`，且Delta LogLoss、Delta Brier、每票平均命中、五票投资组合平均总命中、最佳票`>=3`、`>=4`、`=5`七项观测经验p值全部不高于`alpha`。smoke只能验证执行链，不能作为算法证据；无论结果如何，当前`promotionPassed`仍为false。

## 9. 先预测、后更新时序

每个目标期严格执行：

```text
1. 读取目标期之前的在线状态
2. 各专家输出80个号码边际概率
3. 使用上一期结束后的专家权重生成混合概率
4. 生成研究Top5组合
5. 再读取当期开奖并计算损失/命中
6. 更新EWMA、遗漏、pairwise状态
7. 更新下一期使用的专家权重
```

第 5 步之前不得让当前开奖号进入概率、权重、校准或组合生成。

## 10. 报告边界

开发报告固定包含：

```text
schemaVersion=kl8_pick5_development_v1
evaluationKind=development_prequential_challenger
evidenceStatus=exploratory_reused_development
frozenRead=false
validationOpened=false
promotionPassed=false
recommendationEnabled=false
formalRecommendation=null
```

未通过准入时：

- `userVisibleCandidates=[]`；
- `researchCandidates` 只允许在显式审计模式输出；
- 文案必须标注“研究观察，未通过准入，不是正式推荐”。

## 11. 建议代码结构

首版建议新增：

```text
docs/kl8_pick5_design.md
src/analysis/kl8_pick5_probability_v1.py
src/analysis/kl8_pick5_null.py
scripts/kl8_fetch_history.py
scripts/kl8_pick5_predict_today.py
scripts/kl8_pick5_development.py
tests/test_kl8_pick5_probability_v1.py
tests/test_kl8_pick5_null.py
```

## 12. 完成与激活条件

`kl8_pick5_v1` 从开发挑战器进入可收集独立 Validation 的最低条件：

- 核心、CLI、泄漏和双口径测试通过；
- 默认开发协议完整运行并保留不可覆盖报告；
- 5000 次全流程随机模拟完成且经验假阳性率满足预注册阈值；
- 源码、配置、数据截止期和未来更新规则全部锁定；
- 新的独立 500 期 Validation 一次性通过前，不展示正式候选。

该设计的目标不是保证快乐8选5命中提高，而是让系统能正确区分“有可复验证据的边际概率改善”和“历史噪声中的漂亮排序”。
