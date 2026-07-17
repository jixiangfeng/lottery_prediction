# 本次自动执行报告 | Automation Execution Report

## 2026-07-16 数字彩第三轮算法闭环完善
- 每期快照新增实际启用子模型各自 TopK；开奖后自动生成 `modelHits/modelPerformance/recommendedModelWeights`，样本少于 5 期不调权，之后加一平滑且单模型最多浮动 20%。
- 严格前推同步输出逐模型 Top 候选命中；ML 未训练或蒙特卡洛关闭时不会生成伪模型复盘样本。
- 遗漏升级为与频率窗口一致的多窗口截断特征；日报 JSON/Markdown 显式输出 `omissionWindows`。
- 新增奇偶/大小/质合历史平滑概率约束，支持 `off|soft|hard`；默认 `soft` 只施加小惩罚，避免过度筛号。
- 组三/组六分别在各自无序数字集空间重新计算 16 模型分位；蒙特卡洛升级为多窗口边际+位置对条件+形态接受的联合抽样。
- 数字彩聚焦回归 `62 passed`；8 个核心模块合计覆盖率 `92%` （1767 语句，138 未覆盖）。`black --check`、项目 flake8、相关 `py_compile`、两个 CLI 帮助和 Makefile 命令展开均通过。
- 仓库仍没有福彩3D/排列三/排列五真实 CSV，未生成新的真实效果结论；全仓 `make ci` 仍受既有 `src/analysis/kl8_analysis_plus.py:994` 语法错误阻断。

## 2026-07-16 数字彩集成投票与真实推荐闭环
- 在既有全空间评分上补齐质合、连号、镜像、和值尾、上期距离和同位置重号，并接入蒙特卡洛、sklearn 逻辑回归排序器，共 16 个投票器统一转换为过滤空间排名分位后融合。
- 严格前推新增 `ensemble_voting`，与旧 `current_statistics`、多次 `uniform_random` 使用相同过滤条件、候选数和形态预算进行比较。
- 新增 `digit_pick_tracking.py`：保存开奖前候选快照，数据更新后自动复盘源期之后第一期开奖，并输出单期 JSON/Markdown 与累计真实命中摘要。
- 机器学习训练样本严格按时间构建：每个历史正样本和负样本仅使用该目标期之前的统计特征；蒙特卡洛使用相同过滤条件。两者都只参与排序，不作为真实概率。
- 严格前推新增 30/50/100/300/全历史独立窗口比较；Makefile 默认启用高级模型与窗口比较，可通过变量关闭以控制排列五耗时。
- 聚焦回归使用隔离 Python 3.12.3、`scikit-learn 1.8.0` 和真实 sklearn 训练执行：`57 passed`，8 个数字彩核心模块合计覆盖率 `93%` （1540 语句，109 未覆盖）。为避免仓库 `.coveragerc` 对 `src/analysis/*` 的既有排除，本次覆盖率命令使用空配置采集。
- 聚焦 `black --check`、按项目 `.flake8` 执行的 flake8、相关模块 `py_compile`、两个 CLI `--help` 和 `make -n digit-report/digit-walk-forward` 均通过；日报 JSON/Markdown 已抽查 16 模型分位、高级模型证据和推荐留痕字段。
- 当前环境没有 Conda/Python 3.11，仓库也没有福彩3D/排列三/排列五真实 CSV；因此未生成新的真实前推结论。全仓 `make ci` 仍会被既有 `src/analysis/kl8_analysis_plus.py:994` 的 `global rule_filter` 语法错误阻断，该文件不属于本需求，未修改。

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

## 2026-07-16 数字彩增量统计快照

### 需求摘要
- 为福彩3D、排列三、排列五日报统计层实现首次全量、之后增量；理论数学基线与历史经验统计分层；逐期前推保持独立截止期计算。

### 关键假设
- 日报允许 O(n) 计算已处理前缀摘要验证历史修正，但 cache hit/追加路径不对完整历史重算统计。
- 基础近期状态维护 10/30/50/100/300，动态全历史窗口从聚合计数生成。

### 方案概览
- `digit_statistics_snapshot.py` 保存版本、规则/稳定窗口签名、前缀摘要、全历史与窗口聚合、近期队列、遗漏和最新开奖，采用进程内 `RLock`、跨进程 `flock` 与原子替换。
- `digit_statistics.py` 按规则签名缓存精确理论枚举；日报新增数学基线摘要和 `statisticsUpdate`。
- 快照概率直接使用 Counter 与样本量计算 Dirichlet 平滑；日报 Markdown/JSON 使用临时文件、`fsync` 与 `os.replace` 原子落盘。
- hindsight 全历史回放改为显式开关，默认使用开奖前 prediction snapshot 复盘。

### 自测范围
- 覆盖理论表与零合法组合、首次全量、cache hit、单期/多期追加、固定/全历史窗口碰撞、窗口淘汰、遗漏、转移、空历史 FC3D/PL5、历史修正/删减、损坏与配置失配、显式重建、Counter 不展开闸门、候选一致性、同/跨进程并发原子写及 walk-forward 快照隔离。
- 最终复审新增确定性跨进程屏障：新进程先持有锁，旧进程读取 41 期文件身份后真实阻塞，新进程写完 42 期才释放；覆盖相同配置、不同固定窗口、不同先验和 `rebuild=True`，并增加窗口失配路径禁止调用 `_atomic_write_json` 的纯函数回归测试。

### 验证结果
- 聚焦测试：`.venv/bin/python -m pytest tests/test_digit_statistics_snapshot.py tests/test_digit_report.py -q` -> `45 passed`；其中快照模块 `30 passed`。
- 完整测试：`make test RUN=.venv/bin/python` -> `253 passed, 2 warnings`；warning 为既有 sklearn PCA 小样本运行时告警。
- 构建校验：`.venv/bin/python -m compileall -q src scripts` 通过；`git diff --check` 通过。
- 本次 3 个 Python 文件使用 `isort --profile black`、Black、flake8 均通过；对 `digit_statistics_snapshot.py`、`digit_report.py` 使用 `mypy --follow-imports=skip` 检查无错误。常规导入追踪仅报告既有 `digit_candidates.py:507` 类型债务，本次未新增 mypy 错误。
- `/tmp/fc3d_official_20260716.csv` 共 2255 期：冷启动 `full_rebuild/processedRows=2255/0.508615s`，热命中 `cache_hit/processedRows=0/0.052659s`，追加合法模拟期 `2026187,123` 后为 `incremental/processedRows=1/0.059558s`。
- 对应全量分析耗时分别为 `0.236099s`、`0.235295s`、`0.233885s`。
- 冷/热/追加结果均与同一完整 DataFrame 的 `analyze_digit_history(...)` 深度等价。模拟期仅写入系统临时目录 `fc3d_incremental_review_20260716_yxwrhte6/`，未进入正式报告或仓库数据。
