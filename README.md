# KL8 (快乐 8) 数据分析工具集

本仓库围绕 `src/analysis` 目录的脚本提供快乐 8 历史数据下载、统计分析、候选组合生成与收益回测能力。当前版本聚焦离线分析，深度训练流程已拆分。

> **环境建议**：请始终在名为 `python311` 的 Conda 环境或等效的 Python 3.11 虚拟环境中执行命令，以保持依赖一致。

## 功能清单
- 🔄 `scripts/get_data.py`：下载快乐 8 历史数据，支持顺序出球模式。
- 📝 `scripts/daily_report.py`：生成快乐 8 每日 Markdown 统计报告，包含热冷号、遗漏、分布、选十候选组、固定候选回测、策略横向对比、滑动窗口稳定性回测与参数自动搜索。
- 📦 `src/common.py`：封装数据下载、期号查询与历史数据加载。
- 🌐 `src/data_fetcher.py`：带域名白名单、超时与重试机制的抓取器。
- 🎯 `src/analysis/feature_enhancer.py`：整合近期动量、共现谱、Dirichlet 平滑、PCA 与图嵌入的综合特征评分。
- 🧮 `src/analysis/rule_miner.py`：基于 FP-Growth 的频繁项集与关联规则筛选，支持软/硬模式。
- 🎲 `src/analysis/copula_sampler.py`：高斯 Copula 多样性采样，结合互信息惩罚提升组合相关性建模。
- 🧠 `scripts/train_graph_embeddings.py`：Node2Vec 图嵌入训练脚本，自动识别 CPU / NVIDIA GPU / AMD ROCm 设备。
- 🧩 `src/lotteries/`：多彩种规则注册表，已定义快乐8、福彩3D、排列三、排列五的号码结构与校验规则。
- 🔢 `src/analysis/digit_statistics.py`：数字彩通用统计模块，支持位置、位置对、和值、跨度、形态、奇偶/大小/质合、连号、镜像、和值尾、上期距离、同位重号和遗漏统计。
- 🧹 `src/analysis/digit_data.py`：数字彩 CSV/DataFrame 标准化模块，支持期号/开奖号码列名识别、合并号码拆位、范围校验和按期号排序。
- 📄 `src/analysis/digit_report.py` / `scripts/digit_report.py`：从本地 CSV 生成福彩3D、排列三、排列五 Markdown 分析日报。
- 🎯 `src/analysis/digit_candidates.py`：以 NumPy 紧凑数组复用三位 1000 种/五位 100000 种静态空间和特征，只为选择所需候选构造对象；支持启发式复合分与最多 16 个投票器集成排序，再执行形态配额和确定性多样性选择。
- 🧠 `src/analysis/digit_advanced_models.py`：统一接入多窗口蒙特卡洛模拟和 sklearn 逻辑回归候选排序，两者均只作排序票。
- 📈 `src/analysis/digit_backtest.py`：数字彩候选回测模块，支持三位直选/组选命中和排列五直选命中统计。
- 🧪 `src/analysis/digit_walk_forward.py` / `scripts/digit_walk_forward.py`：严格逐期前推回测，每个目标期只使用此前历史，并与 `uniform_random` 基线对比。
- 🧪 `tests/`：覆盖配置、特征增强、规则挖掘、Copula 采样等模块的 Pytest 套件。

## 快速开始
```bash
conda activate python311
make setup
make download-data             # 可重复执行，获取最新历史数据
make daily                     # 更新数据并生成日报/推荐快照/复盘/累计汇总
make train-graph               # 可选：训练/更新图嵌入缓存
make run                       # 运行基础示例
```

## 生成每日统计报告
```bash
make daily
```

可通过变量调整候选组数量、每组号码数和输出目录：

```bash
make daily REPORT_COUNT=20 GROUP_SIZE=10 OUTPUT_DIR=reports
```

可通过策略模式控制主推荐来源：

```bash
make daily MODE=auto                                # 默认：自动选择参数搜索第一名
make daily MODE=manual STRATEGY=omission_mix        # 固定使用指定参数，便于长期观察
make daily MODE=stable                              # 若已有实盘累计统计，则使用最佳参数，否则回退 auto
make daily BATCH_TRIALS=50                          # 生成多批候选并选择组合总评分最高的一批
```

报告会写入：

```text
reports/kl8_daily_<最新期号>.md
reports/html/kl8_daily_<最新期号>.html
reports/data/kl8_daily_<最新期号>.json
reports/data_source.json
```

报告内容包括最新开奖、数据质量状态、数据源状态、实盘参数加权状态、最近窗口热号/冷号、当前遗漏、区间/尾数分布、选十候选组、每组结构评分、候选组覆盖率/相似度、候选组合总评分，以及把当前候选组固定回放到最近 100 期历史开奖上的命中分布、投入、返奖和收益率。报告会先搜索若干热号/冷号/遗漏/随机权重组合，并把综合评分最高的参数作为主推荐候选；若 `reports/live_summary.md` 已有参数累计收益率，会按 `historical_score + 0.25 * live_roi` 对参数搜索结果做实盘加权排序。主参数确定后，系统默认生成 30 批候选组合并选择组合总评分最高的一批作为最终推荐，可通过 `BATCH_TRIALS` 调整。组合总评分会综合单组质量、整体覆盖率、平均低重合、最大低重合、区间覆盖和尾数覆盖。同时横向比较 `random`、`hot`、`cold`、`balanced`、`hybrid` 五类策略，并用多个 100 期滑动窗口查看策略稳定性；参数搜索结果还会做“训练窗口/测试窗口”前推验证，输出泛化差距、测试收益率和过拟合风险。每次生成报告还会在 `reports/picks/` 保存下一期推荐快照；后续开奖数据更新后，已开奖快照会自动在 `reports/evaluations/` 生成复盘，并在存在至少一期复盘时生成 `reports/live_summary.md` 累计实盘表现。下载元信息会写入 `data/kl8/download_meta.json`，当官方下载失败且本地已有 CSV 时会回退本地缓存并标记 `usedCache=true`。HTML 报告为纯静态自包含文件，无外部 CDN 依赖，可直接用浏览器打开。JSON 报告是 Vue/App 友好的结构化数据，包含 `latestNumbers`、`candidateGroups`、`candidateCoverage`、`candidatePortfolioScore`、`candidateBatchOptimization`、`walkForwardValidation`、`backtest`、`strategyComparison`、`slidingWindow`、`parameterSearch`、`dataQuality`、`dataSource`、`liveParameterWeights` 与产物路径。报告只做历史数据统计和娱乐参考，不保证中奖。

## 多彩种规则架构
当前核心日报仍以快乐8为主，但已经新增统一玩法规则层：

```text
src/lotteries/
├── base.py      # BallSpec / LotteryRule / validate_numbers
├── kl8.py       # 快乐8：1-80 开 20，默认选十
├── fc3d.py      # 福彩3D：百十个位 0-9，可重复
├── pl3.py       # 排列三：百十个位 0-9，可重复
└── pl5.py       # 排列五：万千百十个位 0-9，可重复
```

最小调用示例：

```python
from src.lotteries import get_lottery_rule, validate_numbers

rule = get_lottery_rule("fc3d")
validate_numbers(rule, [1, 1, 1])
```

这层只定义玩法元数据和号码校验，不直接承诺预测效果；数字彩统计、候选、回测与报告链路通过该规则层共享玩法约束。

## 数字彩通用统计
福彩3D、排列三、排列五已经具备通用统计入口：

```python
import pandas as pd
from src.analysis.digit_statistics import analyze_digit_history
from src.lotteries import get_lottery_rule

rule = get_lottery_rule("fc3d")
df = pd.DataFrame([
    {"期数": "2026003", "百位": 1, "十位": 2, "个位": 3},
    {"期数": "2026002", "百位": 1, "十位": 1, "个位": 2},
])
stats = analyze_digit_history(df, rule)
```

当前统计结果包含：

```text
位置频率
30/50/100/300 期多窗口位置频率与贝叶斯平滑概率（日报额外加入当前全历史窗口）
所有 i<j 位置对的多窗口联合频率与贝叶斯平滑概率
形态、和值、跨度、奇偶、大小、质合、连号、镜像、和值尾、上期距离、同位重号的多窗口平滑概率
当前位置遗漏
与 30/50/100/300/全历史频率窗口一致的截断遗漏
和值分布
跨度分布
形态分布：福彩3D/排列三支持 豹子/组三/组六；排列五支持 五同/四一/三二/三一一/二二一/二一一一/全不同
奇偶比
大小比
最新期号与最新号码
按玩法规则精确枚举的理论概率数学基线：形态、和值、跨度、奇偶比、大小比
```

理论概率与经验统计分层展示：理论层按规则枚举全部等可能号码，三位彩精确为豹子 1%、组三 27%、组六 72%；经验层来自已开奖历史及贝叶斯平滑。两层都不是下一期预测，也不能保证预测命中。当前数字彩使用本地 CSV，候选生成、回测和报告已接入，自动下载与 H5 展示仍未接入。

### 数字彩日常增量统计

`digit-report` 默认使用“首次全量、之后增量”的 JSON 快照，默认路径为 `reports/state/<lottery>_statistics_snapshot.json`。首次运行会处理全部历史；数据不变时直接 `cache_hit`；只追加新期时仅更新新增行、全历史聚合、遗漏和有界近期窗口。若调用方在等待锁期间持有的旧历史是当前较新快照的严格短前缀，则返回内存全量计算的 `stale_view`，但绝不降级覆盖快照；此时 `processedRows` 为实际遍历行数、`rebuildReason=stale_view_not_persisted`、`persisted=false`、`snapshotWritten=false`，窗口/先验/显式重建等原请求原因保存在 `requestedRebuildReason`。正常 `full_rebuild`/`incremental` 的两个持久化字段均为 `true`；`cache_hit` 为 `persisted=true`、`snapshotWritten=false`。

```bash
.venv/bin/python scripts/digit_report.py --lottery fc3d --csv data/fc3d.csv --json
.venv/bin/python scripts/digit_report.py --lottery fc3d --csv data/fc3d.csv --json --rebuild-stats
```

快照在损坏、版本/规则/窗口/先验不匹配、历史删减、已处理号码修正或非追加期号时自动全量重建；`--rebuild-stats` 可手动强制重建，`--no-incremental-stats` 仅用于诊断全量口径。固定窗口与动态全历史使用稳定的 `allHistory` 签名，因此总期数碰到 10/30/50/100/300 或自定义固定窗口时不会误判配置变化。全历史窗口从聚合计数重建，近期队列默认最多保留 300 期，不会把全部历史塞入近期队列。

cache hit 和追加仍会对输入前缀做 O(n) 的轻量 SHA-256 完整性校验，用于发现历史修正；该校验不调用全量统计。概率重建直接使用 Counter 与样本量套用 Dirichlet 公式，时间和额外内存只依赖特征域大小。macOS/Linux 使用固定 `.lock` 文件上的 `fcntl.flock` 覆盖“读取、校验、增量、原子替换”整个事务；日报 Markdown/JSON 同样使用临时文件、`fsync` 与 `os.replace` 原子落盘。

日报默认关闭“把今天候选回放全部历史”的 hindsight 回放，因为它会重复全扫且不是有效预测证据；如需迁移诊断可显式传 `--hindsight-backtest`。日常复盘优先使用开奖前已保存的 prediction snapshot。严格逐期前推回测继续按每个目标期的历史截止点独立调用 `analyze_digit_history(...)`，不读取日报最新快照，避免未来数据泄漏。

第二轮候选评分使用可配置的启发式复合对数分：位置边际、位置对、形态、和值、跨度为主，遗漏仅保留小权重辅助。由于特征存在重叠，该分数不是规范联合概率，也不是实际开奖概率。三位彩额外提供 `directCandidates` 与按无序数字集合聚合过滤空间归一化模型质量的 `groupCandidates`；旧 `candidates` 字段继续表示直选候选，排列五的 `groupCandidates` 固定为空。候选 JSON 新增 `modelWeight` / `compositeModelWeight`；旧 `jointProbability` 与 `probabilityMass` 仅作 deprecated 兼容保留。

集成排序固定提供 14 个统计子模型（位置、位置对、形态、和值、跨度、奇偶、大小、质合、连号、镜像、和值尾、上期距离、同位重号、遗漏），并保留蒙特卡洛与 sklearn 逻辑回归排序器槽位，完整配置为 16 个槽位。各槽位在同一过滤空间转为并列中位排名分位，再按固定权重和固定分母融合；未产出结果的蒙特卡洛/ML 槽位使用中性 `0.5` 占位以保持兼容，因此会改变 `ensembleScore` 的绝对尺度，但不会提供候选间的相对排序信号。Markdown/JSON 会输出 `activeModelNames`、`activeModelCount`、`availableModelNames`、`availableModelCount`；其中 active 表示模型提供了非中性结果或模型候选信号，不表示只有 active 槽位参与数学公式。`ensembleScore`、分类器输出与 `compositeModelWeight` 都只用于排序或过滤空间内相对质量，不是实际开奖概率。

每期快照会额外保存实际启用子模型各自的 TopK 候选；开奖后累计 `modelPerformance` 并产出建议权重。单模型样本满 5 期后，日报才使用加一平滑结果保守调权，每个模型相对基础权重最多浮动 20%。旧快照没有逐模型字段时保持基础权重。

结构约束默认使用 `soft`：对奇偶、大小、质合的多窗口平滑概率低于 2% 的结构施加小权重惩罚；可通过 `DIGIT_CONSTRAINT_MODE=off|soft|hard`、`DIGIT_CONSTRAINT_PROBABILITY_FLOOR` 和 `DIGIT_CONSTRAINT_PENALTY_WEIGHT` 调整。`hard` 会直接移出低于阈值的候选，应先做严格前推。

### 数字彩 CSV 标准化
数字彩来源字段不统一时，可以先用 `digit_data` 标准化：

```python
from src.analysis.digit_data import load_digit_csv, normalize_digit_dataframe
from src.lotteries import get_lottery_rule

rule = get_lottery_rule("pl5")
df = load_digit_csv("data/pl5/data.csv", rule)
```

支持两类输入：

```text
期号,开奖号码
2026001,01234
```

或：

```text
期数,万位,千位,百位,十位,个位
2026001,0,1,2,3,4
```

标准化输出统一为：

```text
福彩3D/排列三：期数, 百位, 十位, 个位
排列五：期数, 万位, 千位, 百位, 十位, 个位
```

### 生成数字彩分析日报
当本地已有 CSV 后，可以生成数字彩 Markdown 日报；加 `--json` 会同时输出结构化 JSON：

```bash
python scripts/digit_report.py --lottery fc3d --csv data/fc3d/data.csv --json
python scripts/digit_report.py --lottery pl3 --csv data/pl3/data.csv --json
python scripts/digit_report.py --lottery pl5 --csv data/pl5/data.csv --json
```

或使用 Makefile，默认 `DIGIT_JSON=1` 会输出 JSON：

```bash
make digit-report DIGIT_LOTTERY=pl5 DIGIT_CSV=data/pl5/data.csv
make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_CANDIDATE_COUNT=20
make digit-report DIGIT_LOTTERY=pl3 DIGIT_CSV=data/pl3/data.csv DIGIT_RANKING_MODE=composite
```

输出路径：

```text
reports/fc3d_daily_<最新期号>.md
reports/pl3_daily_<最新期号>.md
reports/pl5_daily_<最新期号>.md
reports/data/fc3d_daily_<最新期号>.json
reports/data/pl3_daily_<最新期号>.json
reports/data/pl5_daily_<最新期号>.json
reports/picks/digit/<彩种>_<源期号>.json
reports/evaluations/<彩种>_<开奖期号>.md
reports/evaluations/<彩种>_live_summary.md
```

报告包含最新开奖、位置频率 Top、当前位置遗漏 Top、和值/跨度/形态、奇偶/大小分布、集成候选、候选回测和理性提示。每次生成日报会先复盘此前已开奖的推荐快照，再保存当前候选供下一期开奖后核验；累计汇总只统计开奖前已留痕推荐。当前数字彩报告只依赖本地 CSV，不自动联网下载；候选生成默认排除上期原号、排除三位豹子、限制常用和值/跨度区间和高重复形态，仍不保证命中。固定候选回放只适合查看候选覆盖，不能代表未来表现；策略比较应使用下方严格逐期前推命令。
默认日报会执行 20000 次联合蒙特卡洛模拟：先融合多窗口位置边际分布，再按位置对条件分布逐位抽样，最后使用形态平滑概率接受采样。轻量二分类排序器使用最近 60 个可进入当期候选空间的训练目标；历史不足时 ML 会自动记为未训练。两者均使用固定种子，不直接预测下期号码，也不作概率校准。

三位彩组选不再直接平均直选集成分：组三和组六分别在各自无序数字集合空间内重新计算 16 个模型分位，再执行形态配额。排列五仍只提供直选。

当前候选核心已改为全空间确定性评分，不再依赖随机抽样：三位彩默认允许跨度 1 的组三，但组六因三个数字互异自然至少跨度 2；排列五允许少量“三一一/三二”作为防守形态，并通过配额和排序保持主流形态占多数。遗漏项采用对数压缩和封顶，多窗口权重、频率权重、遗漏权重及多样性权重均可通过 `DigitCandidateConfig` 配置。

### 数字彩严格逐期前推回测

固定“今天的候选”回放整段历史会使用未来统计信息，不能作为策略验证。严格逐期前推会对每个目标期重新训练，且训练集截止到目标期前一期：

```bash
make digit-walk-forward DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_WF_PERIODS=50
make digit-walk-forward DIGIT_LOTTERY=pl3 DIGIT_CSV=data/pl3/data.csv DIGIT_WF_PERIODS=50
make digit-walk-forward DIGIT_LOTTERY=pl5 DIGIT_CSV=data/pl5/data.csv DIGIT_WF_PERIODS=30
```

Makefile 默认开启高级模型、30/50/100/300/全历史独立窗口比较和多随机基线；需要真正外层未见验证时再开启嵌套调参：

```bash
make digit-walk-forward \
  DIGIT_LOTTERY=fc3d \
  DIGIT_CSV=data/fc3d/data.csv \
  DIGIT_WF_PERIODS=30 \
  DIGIT_WF_BASELINE_RUNS=20 \
  DIGIT_WF_NESTED_TUNING=1 \
  DIGIT_WF_INNER_VALIDATION_PERIODS=10
```

CLI 对应参数为 `--baseline-runs`、`--nested-tuning`、`--inner-validation-periods`、`--advanced-models`、`--monte-carlo-simulations`、`--ml-training-periods`、`--ml-negative-samples`、`--compare-windows`、`--constraint-mode`、`--constraint-probability-floor`、`--constraint-penalty-weight` 与 `--report-prefix`。注意：`make digit-walk-forward` 默认传入 `--advanced-models`，直接运行 `scripts/digit_walk_forward.py` 则需显式添加该开关；数字彩日报 CLI/API 继续保持默认启用高级模型，可用 `--no-monte-carlo` / `--no-ml` 关闭。报告以直选/组选命中及其随机基线百分位为主，同时输出实际启用模型、逐模型 TopK 命中、真实开奖号的模型排名分位桶与独立窗口稳定分；位置覆盖与 `candidateScorePercentile` 仅是选择器内部诊断。后者采用 mid-rank，所有随机运行分数打平时可能显示 50%，不代表预测优势。任何历史回测结果都不能保证未来中奖，也不得根据外层目标期开奖结果反向选择配置。

也可直接运行：

```bash
python scripts/digit_walk_forward.py \
  --lottery fc3d \
  --csv data/fc3d/data.csv \
  --periods 50 \
  --min-train-size 100 \
  --candidate-count 10 \
  --output-dir reports/evaluations
```

每次输出 JSON 与 Markdown，包含目标期数、候选数、直选命中、三位彩组选命中、各位置覆盖、最大连续未中，以及 `current_statistics`、`ensemble_voting` 相对 `uniform_random` 的差异。期号必须为纯数字且数值唯一；系统按数值而非字符串排序，因此 `8、9、10、11` 这类非零填充期号也不会把未来期混入训练集，`01` 与 `1` 会被视为重复并拒绝。Makefile 默认写入 `reports/evaluations`，可通过 `DIGIT_WF_OUTPUT_DIR` 覆盖。随机基线使用固定种子保证可复现，但单个小窗口波动很大；任何历史领先都不代表未来有效，更不能保证提高中奖概率或盈利。

## Vue3 H5 用户端
项目提供独立的 Vue3 + Vite H5 用户端，目录为 `h5/`。它不是静态 HTML 模板，而是运行时读取 `public/report-data/latest.json` 与 `public/report-data/index.json` 的动态前端工程，适合手机浏览器或微信内 H5 访问。首版采用蓝白用户端风格，并通过底部 Tab 拆分为首页、推荐、回测、策略四个页面；回测与策略页使用 ECharts 展示命中分布和策略收益率；历史期号可点击进入 `/report/:issue` 单独分享页。

首次安装依赖：

```bash
make h5-install
```

启动开发服务：

```bash
make h5-dev
```

构建生产产物：

```bash
make h5-build
```

流程说明：`make h5-dev` / `make h5-build` 会先执行 `make daily` 生成最新 `reports/data/kl8_daily_<期号>.json`，再同步到 `h5/public/report-data/latest.json` 与 `h5/public/report-data/index.json`；Vue 页面会按当前路由动态加载最新或指定期号的 JSON。

## 一键启用全算法
以下命令会同时启用高级模式（遗传 + 贝叶斯 + Copula + 互信息惩罚）、特征增强、关联规则过滤以及可调的 Copula 采样参数：

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

> 建议在执行前运行 `make train-graph` 以生成最新的图嵌入缓存，确保特征增强通道生效。

## 特征增强与模式组合示例
```bash
# 混合模式：综合近期动量与共现谱
python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 120 --limit_line 200 --advanced_mode 2 --feature_mode hybrid

# 并行模式下强调趋势：使用 Plus 版本并设定线程池
python src/analysis/kl8_analysis_plus.py --cal_nums 15 --total_create 500 --max_workers 6 --advanced_mode 1 --feature_mode momentum

# 仅基于共现谱评分挑选候选
python src/analysis/kl8_analysis.py --advanced_mode 1 --feature_mode cooccurrence --limit_line 150
```

`--feature_mode` 选项：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `hybrid`（默认） | 动量、共现谱、Dirichlet、PCA、图嵌入综合排序 | 通用推荐 |
| `momentum` | 强调近期与长期窗口频率差 | 捕捉短期热点时 |
| `cooccurrence` | 聚焦号码共现谱主特征向量 | 注重号码协同关系时 |

## 关联规则筛选
```bash
python src/analysis/kl8_analysis.py \
  --cal_nums 10 --total_create 200 --limit_line 200 \
  --advanced_mode 2 --feature_mode hybrid \
  --rule_filter soft --rule_support 0.6 --rule_confidence 0.8
```
- `--rule_filter hard`：违反高置信度规则的组合将被直接剔除。
- `--rule_filter soft`：违规组合接受罚分，但可保留以丰富多样性。
- 相关阈值可在 `config.analysis.rules` 中配置，也可通过 CLI 覆盖。

## Copula / 图嵌入 / 互信息
- Copula 采样通过 `--copula_mode auto|off|force` 控制，`--copula_samples`、`--copula_shrinkage`、`--copula_min_draws`、`--copula_multiplier` 可调精细权重。
- 图嵌入缓存默认为 `data_cache/graph_embeddings.npz`，生成命令：`make train-graph`。
- 互信息惩罚默认在高级模式开启，如需调节可调整 `analysis.graph_embedding.weight`。

## 目录结构
```
.
├── config/                # 配置文件（config.yaml）
├── data/kl8/              # 随仓库提供的最小示例数据
├── docs/                  # 架构 / API / 运维文档
├── examples/              # 高频号码统计示例
├── scripts/               # 数据下载与图嵌入训练脚本
├── src/
│   ├── analysis/          # 核心分析脚本与工具
│   ├── common.py          # 公共接口
│   ├── config.py          # 快乐 8 配置入口
│   └── data_fetcher.py    # 历史数据抓取
├── tests/                 # Pytest 测试套件
└── Makefile               # 一键任务入口
```

## 常见问题 FAQ
1. **提示找不到数据文件？**  
   先执行 `make setup` 创建目录，再运行 `make download-data` 或 `python scripts/get_data.py --name kl8`。当前默认数据源是中国福彩网官方接口 `https://www.cwl.gov.cn/`，请确认网络可访问该域名。
2. **如何启用高级算法组合？**  
   使用前述“一键启用全算法”命令或将 `--advanced_mode 2` 与 `--copula_mode auto/force`、`--feature_mode hybrid`、`--rule_filter soft` 联合使用。
3. **Plus 版本与单线程版本如何选择？**  
   - `kl8_analysis.py`：轻量、易调试，适合快速验证。  
   - `kl8_analysis_plus.py`：线程池并行，适合批量生成，需合理设置 `--max_workers`。
4. **批量任务调度**  
   使用 `src/analysis/kl8_running.py` 可遍历参数组合，并通过 `--running_mode` 控制执行策略。
5. **Copula / 图嵌入如何维护？**  
   - 建议定期运行 `make train-graph` 更新嵌入。  
   - Copula 采样需确保 `limit_line` 不小于 `analysis.copula.min_draws`（默认 180），不足时会自动退回传统策略。

## `kl8_running.py` 资源提示
批量运行会按参数列表笛卡尔展开生成大量任务（每个期号 × cal_nums × total_create），容易造成内存和文件句柄压力。请从小规模参数起步，确认后再逐步放大，同时监控 `max_workers`、内存及磁盘空间。

## 测试与质量
- `make ci`：依次执行 `fmt`、`lint`、`test`、`build`。
- `pytest --cov=src`：查看覆盖率（核心模块覆盖率 ≥ 80%）。
- `make clean`：清理缓存、编译产物与 `__pycache__`。

## 协作建议
- 新增接口或配置时同步更新 `docs/api.md`、`docs/architecture.md`、`ASSUMPTIONS.md`。
- 如果扩展其他玩法，请在 `docs/decision_record.md` 记录设计取舍，并在 `config/` 提供新的配置段。
- 若需恢复深度模型训练，可在独立分支重建 pipeline 后再合入主线。

## 版本历史
- **v1.4.0**：新增 Copula 多样性采样、互信息惩罚、图嵌入训练脚本及全算法命令。
- **v1.3.0**：引入 Dirichlet 平滑与关联规则筛选。
- **v1.1.0**：新增特征融合选项、优化数据下载脚本。
- **v1.0.0**：初始发布，包含基本数据下载与分析能力。
