# KL8 算法原理深度解析（2025.10）

## 1. 最新优化摘要
1. **修正贝叶斯分布**：基于 Beta-Binomial 共轭先验，修正早期边际概率计算误差。
2. **马尔可夫链**：引入 1~3 阶转移概率建模，捕捉开奖序列依赖。
3. **信息熵与互信息**：衡量号码间的独立性与关联度，辅助组合筛选。
4. **统计显著性检验**：卡方检验/KS 检验用于验证约束差异的统计意义。
5. **自适应阈值管理**：基于动量的阈值调整器，自动放松/收紧约束。
6. **遗传算法**：多目标适应度（热号覆盖、冷号多样性、趋势约束等）。
7. **深度特征提取**：PCA + MLP 混合模型提炼高级特征向量。
8. **高级号码生成策略**：贝叶斯 + 马尔可夫 + 遗传 + 深度特征的集成框架。
9. **特征增强引擎（新增）**：利用近期动量和共现谱的综合得分，支持 `--feature_mode` 精细化选择。

## 2. 多维概率约束模型

给定历史开奖集合 $D = \{d_1, d_2, ..., d_n\}$，目标是在特征向量 $f_i$ 上约束：

$$|P_{current}(f_i) - P_{historical}(f_i)| \leq \epsilon_i$$

各特征（重复率、冷热号比例、奇偶比、分组占比、和值区间）均拥有对应的容差 $\epsilon_i$，阈值按运行结果自适应调整。

## 3. 重复率与动量

### 3.1 重复率概率

$$R(A, B) = |A \cap B|, \quad P(R=k) = \frac{\#\{(i,j) : R(d_i,d_j)=k\}}{\binom{n}{2}}$$

### 3.2 动量定义（新增）

在特征增强引擎中，使用近期窗口（如 40 期）与长期窗口（如 160 期）的频率差构建动量：

$$M(b) = \text{freq}_{recent}(b) - \frac{|\text{recent}|}{|\text{reference}|} \times \text{freq}_{reference}(b)$$

经标准化后得到动量得分，用于 `feature_mode=momentum`。

## 4. 共现谱分析（新增）

构建共现矩阵 $C$：

$$C_{ij} = \sum_{t=1}^{T} \omega_t \cdot \mathbf{1}[i,j \in d_t]$$

其中 $\omega_t = \text{decay}^{t}$ 表示时间衰减权重。通过主特征向量 $\mathbf{v}_{max}$ 得到号码的谱中心性：

$$S_{co}(i) = \frac{|\mathbf{v}_{max}(i)| - \min}{\max - \min}$$

`feature_mode=cooccurrence` 直接使用谱得分排序。

## 5. 混合特征得分

特征增强引擎默认使用混合权重：

$$S_{hybrid}(i) = w_1 \cdot S_{recency}(i) + w_2 \cdot S_{momentum}(i) + w_3 \cdot S_{co}(i)$$

当前权重设置为 $(w_1, w_2, w_3) = (0.45, 0.25, 0.30)$，可根据需要调整。

## 6. 贝叶斯后验

采用 Beta-Binomial 共轭先验：

$$P(b \mid D) = \frac{\alpha_b + c_b - 1}{\sum_{k=1}^{80} (\alpha_k + c_k - 1)}$$

其中 $\alpha_b$ 为先验，$c_b$ 为历史计数。贝叶斯后验在遗传算法和启发式生成中用于排序。

## 7. 马尔可夫链

使用高阶状态（默认 2 阶）：

$$P(X_t \mid X_{t-1}, X_{t-2}) = \frac{\text{count}(X_{t-2}, X_{t-1}, X_t)}{\sum_{X_t'} \text{count}(X_{t-2}, X_{t-1}, X_t')}$$

结合 `np.random.choice` 按概率采样候选号码。

## 8. 遗传算法概述

- 个体表示：排序后的号码组合。
- 适应度：重复率差、奇偶差、分组差、特征增强得分等综合。
- 交叉：局部交叉 + 集合填充。
- 变异：随机替换与交换。

## 9. 自适应阈值管理

当某约束在最近窗口内失败次数过多时：

$$\epsilon_i^{new} = \epsilon_i^{old} + \max(0.01, \epsilon_i^{old} \cdot r)$$

其中 $r \in [0.1, 0.3]$。持续成功则逐步收紧阈值。

## 10. 深度特征提取

使用滑动窗口构造高维特征，经 `StandardScaler` 标准化、`PCA` 降维、`MLPRegressor` 学习热号预测。结果用于指导候选筛选与遗传算法初始种群。

## 11. 综合评分选拔

最终候选组合通过以下评分函数筛选：

$$\text{Score}(X) = \lambda_1 \cdot \text{Repeat}(X) + \lambda_2 \cdot \text{Parity}(X) + \lambda_3 \cdot \text{FeatureEnhancer}(X)$$

其中 `FeatureEnhancer(X)` 为组合中号码平均的混合特征得分。若评分无法区分，则退回热/冷号补全策略。

---

引入特征增强后，`--advanced_mode 1/2` 与 `--feature_mode` 的组合可显著提升高级模式的命中率与稳定性。建议在生产环境下保留混合模式，必要时根据趋势切换到动量或共现模式。
