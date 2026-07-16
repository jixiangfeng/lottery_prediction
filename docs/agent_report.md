# 本次自动执行报告 | Automation Execution Report

## 2026-07-16 独立质量审查剩余项修复
- 严格 TDD：先增加复合模型术语、精确形态预算、默认质量门槛、PL5/三位彩嵌套调参和 Markdown 指标降级测试，并确认 RED。
- 候选 `score` 明确为重叠特征加权的启发式复合模型分，不是规范联合概率；JSON 新增 `modelWeight` / `compositeModelWeight`，旧 `jointProbability` / `probabilityMass` 仅作 deprecated 兼容。
- 统计直选、随机直选、统计组选、随机组选统一使用共享形态预算：三位彩默认 80% 组六 + 20% 组三，单形态配置全部使用指定形态；排列五主流/三重号防守保持同口径。
- 嵌套调参对三位彩与排列五均调用最终完整候选生成，以实际号码是否进入 `direct_candidates` / `group_candidates` 判定命中，只比较命中数与过滤空间归一化排名分位。
- 默认 `score_floor` 调整为 `2.0`，质量池不足预算时回退到对应形态的纯分数 TopK，且多样性选择不低于纯分数 TopK 最低分。
- 本轮按要求不运行耗时真实报告；数字彩聚焦测试 `55 passed`，仓库 `make test` 为 `179 passed, 2 warnings`，相关模块 `py_compile` 与 `git diff --check` 通过。
- 全仓 `py_compile` 被既有 `src/analysis/kl8_analysis_plus.py:994` 的 `global rule_filter` 语法错误阻断；该文件不属于本轮范围，未修改。

## 2026-07-15 数字彩第一轮可验证优化
- 需求摘要：为福彩3D、排列三、排列五增加无未来数据污染的严格逐期前推回测，并升级形态过滤、候选生成和统计特征；快乐8链路不改动。
- 方案概览：每个目标期单独统计此前历史；当前统计策略与 `uniform_random` 使用相同候选数、过滤器和形态配额；候选核心枚举完整空间后执行确定性多样性选择。
- TDD RED：新增测试首次执行因 `src.analysis.digit_walk_forward` 尚不存在而收集失败，随后按测试契约实现。
- TDD GREEN：数字彩聚焦回归 `29 passed`；真实数据使用每彩种最近 30 个目标期、最小训练 100 期、每期 10 注候选。
- 真实结果：福彩3D 当前策略直选 `1/30`、随机 `0/30`，但平均位置覆盖率低 `4.44` 个百分点；排列三双方直选均 `0/30`，当前策略组选 `2/30`、随机 `1/30`，位置覆盖率低 `8.89` 个百分点；排列五双方直选均 `0/30`，当前策略位置覆盖率低 `30.00` 个百分点。
- 结论：本轮没有证据证明统计策略稳定超过随机。福彩3D 直选与排列三组选仅是小样本局部领先，排列五位置覆盖明显落后，应继续调整权重和多样性，但不得据此承诺提升中奖概率。
- 风险：小窗口命中极稀疏，随机基线方差较大；历史领先或落后都不能推导未来中奖概率。

## 2025-10-14 Copula 采样与图嵌入扩展
- 在高级候选生成中引入 Copula 多样性采样与互信息惩罚（`src/analysis/copula_sampler.py`、`src/analysis/mutual_information.py`），提升号码相关性建模。
- 新增图嵌入训练脚本 `scripts/train_graph_embeddings.py`（PyTorch Node2Vec），自动适配 CPU / GPU / AMD ROCm，并生成 `graph_embedding_scores`。
- `feature_enhancer` 追加图嵌入调试字段与缓存清理方法，综合得分权重大幅增强。
- 配置更新：`config.analysis.copula`、`config.analysis.graph_embedding`、Makefile `train-graph` 目标，以及完整的 `--copula_*` CLI 覆盖参数。

## 需求摘要 | Requirement Summary
- 实施文档《docs/kl8_algorithm_extension_report.md》中列出的短期与中期任务，落地 Dirichlet 平滑、频繁项集挖掘、Copula 采样、图嵌入与互信息惩罚。
- 维持 CLI 配置灵活，同时兼容 CPU/GPU 环境，并将实现细节同步到 README、使用指南与运维文档。

## 方案概览 | Solution Overview
- 架构模块：
  - Copula 模块拟合 80 维相关矩阵，提供候选组合与诊断信息，最终由高级模式整合。
  - 图嵌入训练脚本独立运行，分析阶段仅依赖 NumPy，避免在生产环境安装 PyTorch。
  - 互信息矩阵在高级评分阶段扣减高相关组合，与 Copula 采样共同提升多样性。
- 选型与权衡：
  - 采用自实现高斯 Copula + 特征值截断，避免额外依赖且易于调参。
  - Node2Vec（随机游走 + Skip-gram）在 CPU 环境即可训练，可选 GPU 加速。
  - 互信息默认保守权重，防止惩罚过大，可通过 YAML 统一调整。

## 实现与自测 | Implementation & Self-testing
- 代码实现：`src/analysis/copula_sampler.py`、`src/analysis/mutual_information.py`、`scripts/train_graph_embeddings.py`、`feature_enhancer` 图嵌入通道、`kl8_analysis.py` Copula/互信息集成、Makefile `train-graph`。
- 单元测试：`tests/test_copula_sampler.py`、`tests/test_mutual_information.py`、`tests/test_feature_enhancer.py`。
- 自测命令：
  ```bash
  pytest tests/test_copula_sampler.py tests/test_mutual_information.py tests/test_feature_enhancer.py -v
  python scripts/train_graph_embeddings.py --lottery kl8 --epochs 5 --device auto
  python src/analysis/kl8_analysis.py \
    --cal_nums 20 --total_create 240 --limit_line 240 \
    --advanced_mode 2 --feature_mode hybrid \
    --rule_filter soft --rule_support 0.08 --rule_confidence 0.7 \
    --copula_mode force --copula_samples 64 --copula_shrinkage 0.1
  ```

## 风险与后续改进 | Risks & Next Steps
- 样本量敏感：Copula 需足够历史期，建议监控 `effective_draws` 与数据完整性。
- 权重调节：互信息与图嵌入权重应结合回测持续调整，可研发自动化调参脚本。
- 图嵌入优化：可探索 Node2Vec 的 `p/q` 偏置、长游走及多嵌入投票。
- 运维支持：在 `docs/ops.md` 中补充训练耗时、设备占用与缓存更新频率的监控指引。
