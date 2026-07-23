# 快乐8 v2 探索性特征发现

## 边界

- 输入仅为两遍加载器隔离后的开发区 DataFrame；Frozen 500期只读取期号和日期边界，不解析号码。
- 实际开发区固定1514期：前300期初始训练、随后714期Search、最后500期Evaluation。
- Search选择完成后才对唯一胜者运行Evaluation；Evaluation结果不能改变特征集。
- 该流程不修改`kl8_pick5_v1`，不运行正式5000次null，也不产生推荐号码。

## 特征与模型

- 每期为1..80号生成prior-only面板，包含多窗口频率、遗漏、滞后、趋势、EWMA、上期上下文和过去80期pair lift。
- 五个候选集严格嵌套；LightGBM使用固定seed、小树、扩展训练集并每50期重训。
- 每期概率裁剪后复用`normalize_sum20`归一到期望正例数20；均匀基线固定为0.25。

## 选择与输出

- 候选需同时满足总体LogLoss/Brier不差于均匀基线，且五个时间块的两项差值均非负。
- 合格候选依次按LogLoss改善、Brier改善、较少特征、名称排序；无合格候选时选择`uniform`。
- 报告原子不可覆盖写入，并固定包含`exploratory_feature_discovery_only`、`frozenRead=false`、`promotionPassed=false`、`recommendationEnabled=false`和空`userVisibleCandidates`。

## 2026-07-23全量执行结果

- 福彩官网年度窗口全量共2014期；隔离500期Frozen后开发区1514期。
- 五个嵌套特征组和十三个独立消融组均未通过总体LogLoss/Brier与五块稳定性联合门槛，固定退回`uniform/no-signal`。
- Frozen号码未读取，未运行正式5000次null，未生成推荐。
- 完整定量结果见[`kl8_feature_discovery_v2_results_20260723.md`](kl8_feature_discovery_v2_results_20260723.md)。
