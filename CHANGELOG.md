# Changelog

## Unreleased

### Removed

- 删除历史 v1 集成候选、回测、报告、蒙特卡洛和 ML ranker 实现。
- 删除概率 v2 与在线概率 v3 的实现、CLI、测试、文档和历史报告。
- 删除 v4 中的 `legacy_ensemble_score`、`legacy_ensemble_votes`、`legacy_ensemble_rank_mean` 特征与默认权重。
- 删除旧版兼容入口；项目只保留 learned ranker 训练、冻结评估和研究日报流程。

### Changed

- Makefile、README、API 与运维文档改为 learned ranker 单架构。
- Frozen Test 统计新增直选、组选和位置池候选预算曲线，并同时报告随机基线与 lift。
- 全历史窗口维持弱先验，近期/中期/长期与 regime 特征保留。

### Safety

- 彩票开奖结果高度随机；本项目不保证预测有效、中奖或盈利。
