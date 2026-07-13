# KL8 算法扩展研究报告（2025.10 更新）

## 1. 现状概览
- **核心入口**：`src/analysis/kl8_analysis.py` 负责参数解析、历史数据加载、候选组合生成与写盘，支持基础与高级模式。
- **特征增强**：`src/analysis/feature_enhancer.py` 整合近期频率、动量、共现谱、Dirichlet 平滑、PCA 与图嵌入，并输出结构化调试信息。
- **规则挖掘**：`src/analysis/rule_miner.py` 基于 FP-Growth 挖掘频繁项集，支持软/硬模式过滤组合。
- **候选多样性**：`src/analysis/copula_sampler.py` 与互信息惩罚提供 80 维依赖建模和多样性控制。
- **支撑脚本**：`scripts/train_graph_embeddings.py` 训练 Node2Vec 嵌入；`config/config.yaml` 集中管理 Dirichlet / 规则 / Copula / 图嵌入参数；`tests/` 下的 Pytest 套件覆盖关键模块。

## 2. 新增算法状态
| 模型/机制                    | 阶段      | 状态 | 代码入口 | 说明 |
|-----------------------------|-----------|------|----------|------|
| Dirichlet-Multinomial 平滑  | 短期迭代  | ✅ 已落地 | `feature_enhancer._compute_dirichlet_scores` | 提供窗口自适应平滑与调试信息，参数由 YAML 配置 |
| FP-Growth 关联规则          | 短期迭代  | ✅ 已落地 | `src/analysis/rule_miner.py` | 支持缓存与 CLI 覆盖阈值，集成在高级候选阶段 |
| Copula 多样性采样           | 中期迭代  | ✅ 新增 | `src/analysis/copula_sampler.py` + `advanced_number_generation` | 估计 80 维相关结构，`--copula_*` 参数可调 |
| Node2Vec 图嵌入特征         | 中期迭代  | ✅ 新增 | `scripts/train_graph_embeddings.py` + `feature_enhancer` | Skip-gram 训练，自动识别 CPU/GPU/AMD ROCm，结果写入 `data_cache/` |
| 互信息多样性惩罚            | 自主扩展  | ✅ 新增 | `src/analysis/mutual_information.py` | 计算 80×80 互信息矩阵，在高级评分阶段扣减高关联组合 |

## 3. 实施记录
### 3.1 Dirichlet 平滑与规则挖掘
- 保持 Dirichlet 通道，融合在综合得分中；`config.analysis.dirichlet` 暴露可调参数。
- 规则挖掘新增缓存与软/硬模式，`--rule_filter`、`--rule_support`、`--rule_confidence` 可在 CLI 覆盖。
- 测试：`tests/test_feature_enhancer.py`、`tests/test_rule_miner.py` 覆盖主要逻辑。

### 3.2 Copula 采样
- `CopulaSampler` 使用 NumPy 实现高斯 Copula，特征值截断保证正定性，支持 shrinkage、samples、min_draws 等配置。
- 高级模式整合 Copula 候选，自动去重并记录条件数/样本量等诊断信息。
- 测试：`tests/test_copula_sampler.py` 验证组合长度、去重、随机种子稳定性与异常分支。

### 3.3 图嵌入特征
- `scripts/train_graph_embeddings.py` 采用随机游走 + Skip-gram（负采样），支持 `--device auto`，输出 `npz` 缓存与训练元信息。
- `feature_enhancer` 增加 `graph_embedding_scores` 及缓存清理函数，综合得分权重可在 YAML 配置。
- Makefile 新增 `train-graph` 目标，README/使用指南同步更新。

### 3.4 互信息惩罚
- `src/analysis/mutual_information.py` 基于指示矩阵快速计算互信息，默认平滑避免零概率。
- 高级候选阶段通过线性扣分抑制高相关组合，可调节 `analysis.graph_embedding.weight` 控制影响。
- 测试：`tests/test_mutual_information.py` 检查对称性、对角线归零及空输入回退。

## 4. 自测与验证
- **单元测试**：`pytest tests/test_copula_sampler.py tests/test_mutual_information.py tests/test_feature_enhancer.py -v`
- **训练脚本**：`python scripts/train_graph_embeddings.py --lottery kl8 --epochs 5 --device auto`
- **全算法命令**：
  ```bash
  python src/analysis/kl8_analysis.py \
    --cal_nums 20 \
    --total_create 240 \
    --limit_line 240 \
    --advanced_mode 2 \
    --feature_mode hybrid \
    --rule_filter soft \
    --rule_support 0.08 \
    --rule_confidence 0.7 \
    --copula_mode force \
    --copula_samples 64 \
    --copula_shrinkage 0.1
  ```
- Copula 拟合会输出 `effective_draws`、条件数等诊断信息；图嵌入训练脚本逐 epoch 打印平均损失与耗时。

## 5. 风险与后续建议
1. **Copula 样本量**：确保 `limit_line ≥ analysis.copula.min_draws`（默认 180），不足时会自动降级；建议增加数据质量监控。
2. **权重调节**：互信息惩罚、图嵌入权重仍需长期回测验证，可引入自动化调参脚本。
3. **图嵌入参数**：后续可引入 Node2Vec 的 `p/q` 偏置与更长随机游走，尝试多份嵌入做 A/B。
4. **回测体系**：Contextual Bandit / VAE 等长期规划依赖更完善的收益记录与回测流水线。
5. **运维支持**：在 `docs/ops.md` 补充 Copula 与图嵌入训练的监控指标（耗时、设备占用、缓存更新时间）。

> 已同步更新 `docs/decision_record.md`、`CHANGELOG.md`、`docs/agent_report.md`、`ASSUMPTIONS.md` 与 README，确保全算法命令和配置说明在文档中一致。
