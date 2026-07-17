# 公共 API

## 玩法规则

- get_lottery_rule(code)：返回 fc3d、pl3 或 pl5 的 LotteryRule。
- list_lottery_rules()：返回当前三种玩法。
- validate_numbers(rule, numbers)：校验号码数量、位置范围和重复规则。

## 数据与统计

- load_digit_csv(path, rule)：读取并标准化数字彩 CSV。
- fetch_digit_history(lottery, ...)：显式从固定白名单抓取福彩3D或排列三历史，不被日报/回测隐式调用。
- write_digit_history_csv(draws, path)：保留期号、号码、日期、来源并原子写入 CSV。
- normalize_digit_dataframe(df, rule)：统一期号和位置列。
- analyze_digit_history(df, rule, ...)：计算多窗口经验统计。
- get_digit_theoretical_probabilities(rule)：精确枚举理论数学基线。
- analyze_digit_history_with_snapshot(...)：增量统计并返回更新元数据。

## 候选与报告

- DigitCandidateConfig：候选数量、窗口权重、排序模式和结构约束。
- DigitCandidateConfig.exclude_latest：`None` 表示玩法默认值；三位彩默认不排除上期原号，排列五默认排除；显式 `True/False` 会覆盖默认值。
- generate_digit_candidates(...)：生成兼容的直选候选结果。
- generate_digit_betting_candidates(...)：生成直选和三位彩组选候选。
- build_advanced_model_scores(...)：生成蒙特卡洛和逻辑回归外部票。
- generate_digit_report_from_csv(..., freeze_pick_snapshot=False)：生成 Markdown，可选同时生成 JSON；显式冻结时拒绝覆盖同源期不同推荐。
- fit_digit_probability_calibration(...)：用严格历史前2/3选参、后1/3守门；失败返回学习权重0。
- build_digit_probability_plan(...)：构建完整1000状态概率、直选纯TopK和组选排列概率和。
- update_online_weights(...)：根据当期各专家给真实号的概率更新权重，并向固定先验收缩。

## 验证

- run_digit_walk_forward_backtest(...)：严格前推并比较统计、集成和随机策略。
- write_digit_walk_forward_reports(...)：写入前推 Markdown 和 JSON。
- save_digit_pick_snapshot(..., immutable=False, experiment_id=None)：保存开奖前候选；schema v3 包含完整候选配置、推荐指纹和不可变标记，不同实验同源期可并存。
- process_digit_pick_evaluations(...)：复盘已开奖快照并更新累计表现。
- poisson_binomial_right_tail(...)：计算每期随机概率可变时的精确单侧右尾概率。
- calculate_group_random_probability(...)：按组选覆盖的有序排列数计算随机命中概率。
- evaluate_viability_metric(...)：执行样本量、显著性、提升、置信下界和分块稳定性闸门。
- build_prediction_viability_report(...)：合并直选与组选闸门；三位彩要求两项均通过。
- run_digit_probability_walk_forward(...)：在最早目标期前冻结一次概率校准，再评估Log Loss、Brier、排名和命中。
- write_digit_probability_walk_forward_reports(...)：原子写入概率 v2 开发评估 Markdown/JSON。
- run_digit_online_probability_walk_forward(...)：前段在线预训练，目标段逐期先预测后反馈，并保存完整权重轨迹。
- write_digit_online_probability_reports(...)：原子写入在线概率 v3 Markdown 和逐期 JSON。
- build_digit_online_probability_plan(...)：消费在线状态并生成下一期完整概率、直选和组选候选；首次运行全量重放，追加数据只更新新增期。
- DigitOnlineProbabilityState：保存规则、配置指纹、历史前缀指纹、处理期数和当前专家权重。
- DigitOnlineProbabilityStateUpdate：报告 `full_rebuild`、`incremental` 或 `cache_hit` 状态更新结果。

严格前推报告 schema v5 在 `strategyViability` 下按策略输出 `directGate`、`groupGate`、`viable` 和 `reason`。每个目标期同时保存 `directRandomProbability` 与 `groupRandomProbability`，便于复核零假设。

所有分数与报告仅用于历史研究，不保证中奖。
