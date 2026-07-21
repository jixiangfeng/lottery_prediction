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

日常预测入口会先从官方白名单抓取最近期开奖，再临时合并到本地CSV和锁定影子状态上做增量更新；不会覆盖`state/learned_ranker_v4/full_history_shadow_*.json`：

```bash
make digit-predict-today DIGIT_LOTTERY=fc3d
make digit-predict-today DIGIT_LOTTERY=pl3
```

也可以直接运行脚本并输出JSON：

```bash
python scripts/digit_predict_today.py --lottery fc3d --json
python scripts/digit_predict_today.py --lottery pl3 --json
```

默认终端只展示最新开奖、准入状态、放弃原因和确定性中文说明。正式策略未激活时不展示号码；`researchTop50`、排序权重、相对均匀基线和前三项特征贡献只保留在`--json`审计输出中。排序权重不是真实开奖概率。

可选DeepSeek文案层默认关闭，且不参与选号或排序。仓库提供`config/ai.example.json`结构；实际密钥放在被Git忽略的`config/ai.local.json`中：

```powershell
.\.venv\Scripts\python.exe scripts\digit_predict_today.py --lottery fc3d --ai
```

默认模型为`deepseek-v4-flash`，接口固定为`https://api.deepseek.com/chat/completions`且关闭思考模式。AI只接收彩种、最新期号、增量开奖、放弃原因、信号指标和验证进度，不接收内部研究号码；请求失败时自动回退到确定性说明。

正式策略未激活但需要人工查看研究排序时，必须显式开启审计视图。该视图只展示Top10，并解释前三名的主要特征贡献，不会把号码写入`userVisibleCandidates`：

```powershell
.\.venv\Scripts\python.exe scripts\digit_predict_today.py --lottery fc3d --ai --show-research
```

历史稳健性可使用固定规则扫描全部完整非重叠500期块：

```bash
python scripts/digit_historical_blocks.py --lottery fc3d --csv data/fc3d/official_history.csv --output reports/retrospective/historical_blocks_v4_fc3d.json
```

该回测从首次具备400期Search/Validation历史后开始，覆盖所有完整块，不允许挑块。当前14块/7000期结果：`fc3d` Top50为`346/7000=4.94%`（`p=0.5945`），`pl3`为`367/7000=5.24%`（`p=0.1824`）；两个彩种均为`0/14`块同时通过LogLoss、Brier与Top50显著性，只能作为回溯稳健性证据。

非线性挑战模型使用三个LightGBM 10分类器，不使用MLForecast连续回归：

```bash
brew install libomp  # macOS首次运行
python scripts/digit_lightgbm_challenger.py --lottery fc3d --csv data/fc3d/official_history.csv --output reports/challenger/lightgbm_position_v1_fc3d.json
```

模型使用521个严格prior-only特征（10期One-Hot滞后、20/50/150/300/500期位置频率、遗漏、上一期和值/跨度/形态），每个外层500期块只用此前500期内部Validation从3个强正则树配置和`0.25/0.5/0.75`均匀收缩中选一组。12块/6000期结果：`fc3d` Top50为`282/6000=4.70%`（`p=0.8639`），`pl3`为`294/6000=4.90%`（`p=0.6471`）；两者LogLoss/Brier均差于均匀，联合通过块均为`0/12`，因此关闭该挑战模型，不继续用同一历史调参。

稀疏v4.1使用`30% LogLoss + 70% Top50边界排序`、FTRL-Proximal逐特征自适应更新和三个在线Hedge专家：

```bash
python scripts/digit_rank_ftrl.py --lottery fc3d --csv data/fc3d/official_history.csv --output reports/challenger/rank_ftrl_v4_1_fc3d.json
```

14块/7000期结果：`fc3d`为`359/7000=5.13%`（`p=0.3183`，稳定块`8/14`，联合通过`1/14`），`pl3`为`368/7000=5.26%`（`p=0.1684`，稳定块`9/14`，联合通过`0/14`）。相较旧稀疏v4排名略有改善，但未达到预注册`10/14`稳定块或总体显著性；专家权重接近均分且动态λ均值仅约`0.31%/0.06%`，模型几乎退回均匀，因此保持研究关闭状态。

v4.1归因报告同时给出实际号码排名区间、与Top50边界的精确线性贡献、预测时单特征零化和从头重训消融。预测时零化没有任何跨彩种共同正候选；完整重训进一步证明`position_trend`（fc3d）和`position_omission`（pl3）的表面副作用不成立。仅fc3d删除`sum_distribution`从`359`升至`379/7000=5.41%`，9块改善、4块下降、1块持平，LogLoss/Brier略好，但总体`p=0.0604`且联合通过仅`1/14`，只能作为未来前瞻影子候选；pl3没有通过完整重训的删除候选，共享模型不删除任何特征。

每日输出现在执行强制准入：历史联合闸门失败、动态λ低于5%或Top50豹子超过1个时，设置`abstained=true`和空`userVisibleCandidates`；内部`researchTop50`只保留审计，不得对用户展示。2026-07-20真实重跑中，fc3d因前两项放弃，pl3还触发`triple_concentration`（豹子10/50）并放弃。日常锁定影子CLI另执行固定组合策略：排除最新一期完全相同号码，Top50按原模型顺序最多保留1个豹子并从完整排名补足50个；这是风险分散规则，不是预测增益证据。

v4.1现同时报告直选Top50投影组选和独立组选Top10，随机基线按每组选键的排列数（豹子1、组三3、组六6）逐期加权并用Poisson-binomial检验。7000期中，fc3d投影组选`1349/7000=19.27%`对加权基线`18.73%`（`p=0.1242`），独立Top10为`445/7000=6.36%`对基线`6.00%`（`p=0.1095`）；pl3投影组选`1215/7000=17.36%`对基线`17.35%`（`p=0.4990`），独立Top10为`418/7000=5.97%`对基线`6.00%`（`p=0.5472`）。两个彩种组选联合通过块均为0，组选推荐保持关闭。

`behavioral_context_v2`是v4内部的标准化行为挑战器，不改变现有11维日常模型或锁定影子状态。旧v1累计频率压力已经停用；v2改为完整号码最近间隔、同组选其他排列最近间隔、上期同位重合、同数字异位重合、形态近期超理论占比和形态连开超理论期望6项风险。全部行为列按每期1000候选标准化，新权重从0开始并使用2倍L2。

协议固定为A/B/C：A是核心v4，B允许标准化行为权重自由学习，C要求全部行为风险权重非正并预先指定为唯一主挑战组；禁止根据结果从B/C中挑赢家。三组复用同一日常Top50策略。准入除LogLoss、Brier、随机基线、时间块和形态约束外，还要求C相对A的配对Top50增量达到`p<0.01`，限制Top50平均形态分布相对`72%/27%/1%`的总变差不超过10%，并要求行为特征相对Top50边界的合计贡献为正且全部固定块稳定。

```bash
python scripts/digit_behavioral_context.py --lottery fc3d --csv data/fc3d/official_history.csv --output reports/development/behavioral_context_v2_fc3d.json --frozen-test-periods 500 --outer-periods 500
python scripts/digit_behavioral_context.py --lottery fc3d --csv data/fc3d/official_history.csv --output reports/development/behavioral_context_v2_fc3d_all_blocks.json --all-development-blocks
```

最终定义的500期快速诊断中，fc3d为A/B/C=`32/33/33`，C相对A新增13期、丢失12期，配对Top50 `p=0.5000`；pl3为`31/39/42`，C相对A新增22期、丢失11期，单组对随机`p=0.00086`，但配对增量`p=0.0401`、LogLoss/Brier均变差。两个彩种均有大量放弃期，fc3d C组形态总变差`14.95%`，因此都不准入。

第一条500期命令只用于快速开发诊断，不能创建影子状态；只有第二条覆盖Frozen以前全部完整500期块的运行才具备晋级资格，且仍须通过其余全部闸门。两条命令都不读取或重跑Frozen。试机号历史数据仍不可用，仅保留`trial_data_unavailable`状态。

完整开发区已按最早可训练点对齐扫描13个固定500期块，共6500期。fc3d A/B/C=`314/323/323`，C命中率`4.97%`，相对A新增163期、丢失154期，配对`p=0.3266`；pl3为`350/323/333`，C命中率`5.12%`，相对A新增171期、丢失188期，配对`p=0.8289`。两者LogLoss/Brier均未改善，固定块存在低于随机、放弃率超过90%；pl3快速诊断的`42/500`没有跨时间复现。逐特征反推还显示两个彩种6项行为风险的平均边界贡献全部为负且13块方向一致，最大伤害来自同数字异位和同位重合。因此v2行为模型保持关闭，不创建影子状态、不接入日常预测。

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
