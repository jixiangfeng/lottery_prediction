# 本次自动执行报告 | Automation Execution Report

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
