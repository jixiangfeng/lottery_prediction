# 福彩3D / 排列三 learned ranker

本仓库只保留当前 learned ranker 架构。历史 v1、概率 v2、在线概率 v3 的实现、CLI、测试、文档和报告已删除，不再提供兼容入口。

> 彩票开奖结果高度随机。本项目仅用于可复核的历史研究，不保证预测有效、中奖或盈利。

## 支持范围

- 福彩3D：`fc3d`
- 排列三：`pl3`
- 排列五：当前 learned ranker 不支持并明确拒绝

## 核心流程

```text
本地官方历史 CSV
  → Search 参数探索
  → Validation 参数选择
  → 锁定参数与源码/数据指纹
  → Frozen Test 一次性评估
  → 未通过闸门时仅输出研究结果
```

最后冻结测试段不参与窗口、权重、候选预算、温度、目标函数或闸门选择。

## 安装与质量检查

要求 Python 3.11；推荐使用项目锁定的 `uv` 命令：

```bash
make ci
```

## 获取与对账历史

```bash
make digit-fetch \
  DIGIT_LOTTERY=fc3d \
  DIGIT_FETCH_PERIODS=0 \
  DIGIT_RAW_JSONL=data/fc3d/raw/history.jsonl

make digit-reconcile-jsonl \
  DIGIT_LOTTERY=fc3d \
  DIGIT_RAW_JSONL=data/fc3d/raw/history.jsonl \
  DIGIT_CSV=data/fc3d/official_history.csv
```

抓取仅允许项目内固定白名单来源，并且只追加原始 JSONL，不直接覆盖标准 CSV。对账命令支持多个 `--raw-jsonl` provider；会先写出 `.reconciliation.json` 冲突/来源不足清单，只有多源号码和日期一致后才生成 CSV。训练、评估和日报不联网。

固定基线矩阵包括 `uniform`、`shape_prior`、`shape_transition_150`、`sum_span_150`、`position_20/50/150/all`，统一报告 LogLoss、Brier、排名、TopK、校准和 ECE。

## 训练

```bash
make digit-learned-ranker-train \
  DIGIT_LOTTERY=fc3d \
  DIGIT_CSV=data/fc3d/official_history.csv \
  DIGIT_V4_FROZEN_TEST_PERIODS=500 \
  DIGIT_V4_OBJECTIVE_PROFILE=research_calibrated
```

快速冒烟可设置 `DIGIT_V4_SMOKE=1`；冒烟结果不能用于效果结论。普通训练默认使用 `research_calibrated`，以相对均匀基线的 LogLoss、Brier、排名、ECE 和时间稳定性作为平滑研究目标；`all_hit_only` 仅保留为显式对照。

## 在线自适应开发模拟

```bash
make digit-learned-ranker-adaptive \
  DIGIT_LOTTERY=fc3d \
  DIGIT_CSV=data/fc3d/official_history.csv \
  DIGIT_V4_FROZEN_TEST_PERIODS=500
```

该命令只运行到 Frozen Test 之前：外层默认连续预测开发区最后500期（stride=1），每期更新20/50/150滚动状态，每10期只用块起点之前最近500期重新执行Inner Search/Validation。参数包含特征权重、temperature和均匀收缩系数 `λ∈{0,0.25,0.5,0.75,1}`。若Inner Search/Validation proper-scoring、LogLoss、Brier、ECE或时间块稳定性任一不通过，后续参数块使用严格均匀概率并标记 `abstained=true`，不进入主推荐。

## 可预测性审计

```bash
make digit-predictability-audit DIGIT_LOTTERY=fc3d
```

审计只加载Frozen之前的号码，执行18项lag=1/2/5时序置换检验，以及uniform、位置频率20/50/150、遗漏、Markov、形态转移、和值跨度的逐期基线对照。默认499次置换、10期配对块，并对全部检验使用Benjamini-Hochberg FDR 5%校正。只有某个简单基线在LogLoss和Brier上同时正向且通过FDR，报告才标记`predictableSignalFound=true`。

## 逐期特征归因与在线梯度

```bash
make digit-online-gradient DIGIT_LOTTERY=fc3d
```

该开发实验并行维护`learning_rate∈{0,0.01,0.02,0.05}`与`λ∈{0,0.25,0.5,0.75,1}`共20个候选学习器。`position_frequency`使用10倍L2，和值、数字对及趋势组使用5倍L2，两个形态特征固定为0。每期先预测，开奖后按最终收缩概率的LogLoss解析梯度更新下一期权重；使用梯度裁剪、L2收缩和权重边界。每10期仅用此前300期Search选择候选，再用紧邻此前100期确认。报告记录真实号码相对Top50边界的逐特征贡献、梯度和更新前后权重；未通过时部署概率保持均匀，并标记`evidenceStatus=exploratory_reused_development`。

稀疏v4已按锁定协议一次性运行最后500期Frozen，`fc3d`与`pl3`均未通过联合闸门；正式策略保持`research`。锁、只读防重跑标记和报告位于`state/learned_ranker_v4/`及`reports/frozen/`，不得重新运行Frozen或根据结果修改后回测同一测试段。

Frozen消费后可用全部已开奖历史初始化前瞻影子状态：

```bash
python scripts/digit_full_history_shadow.py --lottery fc3d --csv data/fc3d/official_history.csv --output state/learned_ranker_v4/full_history_shadow_fc3d.json
python scripts/digit_full_history_shadow.py --lottery pl3 --csv data/pl3/official_history.csv --output state/learned_ranker_v4/full_history_shadow_pl3.json
```

该状态从第151期开始连续更新全部候选，使用20/50/150/300/500期窗口（长期窗口低权重）和300期权重半衰，仅保存最终权重、最近400期校准指标及下一期研究Top50。输出以原子只写方式锁定，正式推荐保持关闭；新的独立验证固定为锁定之后未来500期，历史中不得另挑500期冒充Frozen。

三个命中率开发目标可在一次进程中共享目标期特征：

```bash
uv run --python 3.11 --with-requirements requirements-dev.txt \
  python scripts/digit_learned_ranker.py train \
  --lottery fc3d \
  --csv data/fc3d/official_history.csv \
  --output-dir /tmp/lottery-dev \
  --frozen-test-periods 500 \
  --objective-profile all_hit_only
```

该模式分别写出 direct/group/pool 参数，并在 Search/Validation 元数据中记录直选、组选和固定位置池预算曲线。预算只能由 Search 选择，Validation 只验证已选预算。Search 只选择唯一紧凑 v4 特征配置的权重，Validation 仅对最终配置运行一次。相邻目标期使用150期滚动状态；已完成目标矩阵以 canonical数据指纹、配置和目标索引为键保存为安全NPZ，进程重启后可恢复，不使用pickle。

v4形态特征使用组六、组三、豹子的理论先验 `72%/27%/1%` 作为基线，并计算形态转移与最近30期形态相对该先验的收缩偏离。理论比例本身不构成预测优势；形态权重只有在 Search/Validation 跨时间块稳定时才可采用。

全量历史开发时，最新500期固定为 Frozen Test，前面的全部历史再按时间切为 Search 和 Validation；不得把最新500期用于调参。


参数搜索目标必须和待锁定预算一致，例如：

```bash
--direct-objective-top-k 50
--position-objective-pool-size 3
```

## 冻结评估

```bash
make digit-learned-ranker-evaluate \
  DIGIT_LOTTERY=fc3d \
  DIGIT_CSV=data/fc3d/official_history.csv
```

评估会校验：

- 彩种；
- 参数指纹；
- 参数产物指纹；
- 源码指纹；
- 冻结数据 canonical 指纹；
- `testSegmentUsedForSelection=false`；
- direct/group/position独立随机基线和单侧 p 值；
- Top-1校准分箱与 ECE。

## 研究日报

```bash
make digit-learned-ranker-daily \
  DIGIT_LOTTERY=fc3d \
  DIGIT_CSV=data/fc3d/official_history.csv
```

冻结证据缺失、损坏、不匹配或闸门失败时，日报保持研究模式。日报固定写入 `reports/daily/<lottery>/`；direct、group、position分别显示正式或研究分区，并在 `reports/state/strategy_registry.json` 记录状态迁移、原因、指纹和回滚目标。

## 直接使用 CLI

```bash
uv run --python 3.11 --with-requirements requirements-dev.txt \
  python scripts/digit_learned_ranker.py train --help
```

CLI 子命令：

- `train`：只用 Search 选择参数，Validation确认唯一胜者；默认平滑校准目标；
- `adaptive`：开发区stride=1逐期预测、每10期重选参数、无信号放弃；
- `evaluate`：只用锁定参数评估 Frozen Test；
- `daily`：生成研究日报和不可覆盖快照。

## 代码结构

```text
src/analysis/digit_data.py                         数据标准化与 canonical 指纹
src/analysis/digit_raw_evidence.py                 raw JSONL provider、多源对账和冲突清单
src/analysis/digit_baselines.py                    固定基线矩阵
src/analysis/digit_evaluation.py                   LogLoss/Brier/排名/校准统一指标
src/analysis/digit_strategy_gate.py                实战准入与降级状态机
src/analysis/digit_strategy_registry.py            策略注册表和状态迁移历史
src/analysis/digit_statistics.py                   通用历史统计
src/analysis/digit_statistics_snapshot.py          增量统计快照
src/analysis/digit_learned_features.py             learned ranker 特征
src/analysis/digit_learned_ranker.py               参数、收缩概率、评分、日报与指纹
src/analysis/digit_learned_ranker_adaptive.py      在线自适应逐期模拟、定期重训与放弃
src/analysis/digit_learned_ranker_search.py        Search/Validation搜索与平滑研究目标
src/analysis/digit_learned_ranker_walk_forward.py  Frozen Test 与候选预算曲线
src/analysis/prediction_viability.py                显著性与可行性工具
scripts/digit_learned_ranker.py                     唯一预测 CLI
scripts/fetch_digit_history.py                      显式历史抓取入口
```

详细边界见：

- `docs/learned_ranker_v4_design.md`
- `docs/api.md`
- `docs/ops.md`
