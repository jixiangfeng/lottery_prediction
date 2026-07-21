# Changelog

## Unreleased

### Added

- 新增 raw JSONL 原始证据层、多源冲突对账和 `digit-reconcile-jsonl` 标准 CSV 生成入口。
- 新增 uniform、position frequency、shape transition 独立基线层，以及 LogLoss、Brier、排名、TopK、校准和 ECE 统一指标。
- 新增 `research/observation/active/demoted/retired` 实战策略状态机；Frozen 未评估时禁止激活。
- 新增三层离线验收脚本、数据/源码指纹和 fc3d/pl3 验收产物。
- 新增20/50/150/all位置频率、形态先验/转移、和值跨度固定基线矩阵。
- 新增150期滚动历史状态、Search目标NPZ断点恢复和Top-1 ECE校准报告。
- 新增direct/group/position独立闸门、策略注册表、状态迁移、淘汰和回滚记录。
- 新增多JSONL provider编排及标准化冲突/来源不足报告。
- 新增v4均匀概率收缩`λ`、`research_calibrated` proper-scoring目标和`λ=0`强制放弃主推荐。
- 新增开发区stride=1在线自适应模拟：每期更新状态、默认每10期Inner Search/Validation重选参数，严格止于Frozen边界。
- 新增开发区可预测性审计：18项时序置换检验、7种简单逐期基线、块配对置换和FDR多重校正。
- 新增v4逐期特征归因和正则化在线梯度：20个学习率/收缩候选并行更新，每10期使用prior-only Search/Validation重校准。
- 将在线梯度默认升级为稀疏收缩v4：有害特征组5—10倍L2、形态权重归零、学习率扩展到0.02/0.05，并显式标记复用开发区的探索性证据。
- 锁定稀疏v4后一次性运行fc3d/pl3最后500期Frozen；两个彩种均未通过LogLoss、Brier、Top50显著性和时间稳定性的联合闸门，正式策略保持`research`，防重跑标记永久保留。
- 新增全历史稀疏v4影子预训练：从第151期连续更新到最新期，加入300/500期低权重窗口与300期权重半衰，原子锁定最终状态，并固定只用未来500期作新独立验证。
- 新增全部连续500期历史区块回测，统一使用影子模型权重半衰并禁止挑块；14块/7000期结果未发现跨阶段稳定优势。
- 新增LightGBM三位置多分类挑战模型：521个prior-only特征、内部Validation选参、1000号码联合概率和12个完整500期块；两个彩种均低于5%且概率指标失败，模型关闭为负基线。
- 新增rank-aware FTRL v4.1：Top50边界排序、FTRL稀疏自适应更新、三专家Hedge融合和连续均匀收缩；历史排名略升至`5.13%/5.26%`，但总体不显著且稳定性门槛失败，保持研究关闭。
- 新增v4.1排名距离、Top50边界贡献、预测时零化和完整重训消融；只有fc3d删除`sum_distribution`升至`5.41%`，但`p=0.0604`且联合块不足，只保留为前瞻影子候选。
- 修复近均匀模型仍向用户展示病态Top50的问题：新增历史闸门、最低λ和豹子集中度三重准入，失败时强制`abstained`并清空用户可见候选；排列三10/50豹子快照已撤回。
- 日常锁定影子候选新增组合安全策略：排除上期原号，Top50最多保留1个豹子，并按原始模型顺序补足固定预算；新增策略与全历史影子集成回归测试。
- 新增v4.1直选投影组选和独立组选Top10的7000期评估，使用1/3/6排列数加权基线与Poisson-binomial；fc3d弱改善不显著，pl3无改善，组选推荐保持关闭。
- 新增独立`behavioral_context_v1` A/B挑战器：6项prior-only近期行为特征、零初始权重、10倍L2、配对块置换、Top50/时间块/形态联合闸门；两个彩种完整7000期均失败，保持当前日常模型不变。

### Removed

- 删除历史 v1 集成候选、回测、报告、蒙特卡洛和 ML ranker 实现。
- 删除概率 v2 与在线概率 v3 的实现、CLI、测试、文档和历史报告。
- 删除 v4 中的 `legacy_ensemble_score`、`legacy_ensemble_votes`、`legacy_ensemble_rank_mean` 特征与默认权重。
- 删除旧版兼容入口；项目只保留 learned ranker 训练、冻结评估和研究日报流程。

### Changed

- v4主窗口固定为目标期之前最近20/50/150期，默认half-life=50；全量历史用于生成更多walk-forward目标，而不是每个目标期全量扫描。
- `000-999` 候选核心特征改为 NumPy 批量计算，开发基准由2目标约5.266秒降至约0.066秒。
- Search独立选参，Validation只评估最终Search胜者；不同冻结报告内容按指纹版本化且不覆盖旧产物。
- 抓取入口不再直接覆盖标准CSV；训练、Frozen评估和日报不再提供一键串联入口。

- `pool_hit_only` 改为固定 `position_pool_size` 概率质量池，不再使用直选 Top50 派生宽池。
- 新增 `all_hit_only` 批量开发模式，direct/group/pool 共用 Search/Validation 特征缓存并分别保存参数。
- Search 产物直接记录直选、组选和位置池多预算曲线、随机基线与 lift。
- 新增可配置的直选/组选/位置池搜索目标预算，保证搜索目标与待锁定成本一致。
- 最新500期固定为 Frozen Test；此前全量历史按时间切为 Search/Validation，并保留跨彩种最差折闸门。
- Makefile、README、API 与运维文档改为 learned ranker 单架构。
- Frozen Test 统计新增直选、组选和位置池候选预算曲线，并同时报告随机基线与 lift。
- 长历史用于walk-forward稳定性验证；单目标特征只读取最近150期，删除与主窗口不一致的旧regime字段。

### Safety

- 彩票开奖结果高度随机；本项目不保证预测有效、中奖或盈利。
