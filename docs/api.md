# 公共 API

## 数据

- `load_digit_csv(path, rule)`：读取并标准化历史 CSV。
- `normalize_digit_dataframe(df, rule)`：统一期号和位置列。
- `sort_digit_dataframe_by_issue(df, ascending=True)`：按数值期号排序。
- `canonical_digit_data_sha256(df, rule)`：计算不受 CSV 行顺序与文本格式影响的 canonical 指纹。
- `fetch_digit_history(...)`：显式从固定白名单抓取福彩3D或排列三历史。

## 统计

- `analyze_digit_history(df, rule, ...)`：计算通用历史统计。
- `analyze_digit_history_with_snapshot(...)`：增量统计及快照元数据。
- `get_digit_theoretical_probabilities(rule)`：精确枚举理论基线。

## learned ranker

- `LearnedFeatureConfig`：窗口、窗口权重、alpha、half-life 与 omission cap。
- `build_history_state(history, rule, config, target_issue=...)`：只构建目标期以前的状态。
- `build_candidate_features(state, rule, candidates=None)`：生成完整 `000–999` 原生特征矩阵。
- `LearnedRankerParams`：固定权重、温度和候选成本。
- `score_candidates(features, params)`：执行固定线性评分。
- `build_learned_ranker_plan(features, params, rule)`：生成直选、组选与位置池研究方案。
- `search_learned_ranker_params(history, rule, config)`：只使用 Search/Validation 选参。
- `run_learned_ranker_walk_forward(history, rule, params, split)`：锁定参数后评估 Frozen Test。
- `generate_learned_ranker_daily(...)`：生成研究日报和不可覆盖快照。
- `build_candidate_budget_curve(periods)`：直选预算曲线。
- `build_group_budget_curve(periods)`：组选预算曲线。
- `build_position_pool_budget_curve(periods)`：位置池预算曲线。
- `resolve_activation(...)`：生成 common/direct/group 分项激活语义。

## CLI

唯一预测 CLI：

```text
scripts/digit_learned_ranker.py train|evaluate|daily
```

历史 v1/v2/v3 API 已删除，不提供兼容导入。

## 语义约束

- `sum_prob` 才能称为组选概率；
- `max_perm` 与 `mean_top_perm` 只能称为 score/aggregation；
- softmax 数值是排序归一化值，不应解释为真实开奖概率；
- 所有报告仅用于历史研究，不保证预测有效、中奖或盈利。
