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
- 🔢 `src/analysis/digit_statistics.py`：数字彩通用统计模块，支持福彩3D、排列三、排列五的位置频率、和值、跨度、形态、奇偶大小和遗漏统计。
- 🧹 `src/analysis/digit_data.py`：数字彩 CSV/DataFrame 标准化模块，支持期号/开奖号码列名识别、合并号码拆位、范围校验和按期号排序。
- 📄 `src/analysis/digit_report.py` / `scripts/digit_report.py`：从本地 CSV 生成福彩3D、排列三、排列五 Markdown 分析日报。
- 🎯 `src/analysis/digit_candidates.py`：数字彩统计候选生成器，支持和值、跨度、形态过滤和按位置权重采样。
- 📈 `src/analysis/digit_backtest.py`：数字彩候选回测模块，支持三位直选/组选命中和排列五直选命中统计。
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

这层只定义玩法元数据和号码校验，不直接承诺预测效果；后续福彩3D、排列三、排列五的数据下载、统计指标和回测奖表会基于该规则层逐步接入。

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
当前位置遗漏
和值分布
跨度分布
形态分布：福彩3D/排列三支持 豹子/组三/组六；排列五支持 五同/四一/三二/三一一/二二一/二一一一/全不同
奇偶比
大小比
最新期号与最新号码
```

这一步仍是历史统计基础设施，不能保证预测命中；后续可在此基础上继续接数字彩数据下载、候选生成、奖表回测和 H5 展示。

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
```

输出路径：

```text
reports/fc3d_daily_<最新期号>.md
reports/pl3_daily_<最新期号>.md
reports/pl5_daily_<最新期号>.md
reports/data/fc3d_daily_<最新期号>.json
reports/data/pl3_daily_<最新期号>.json
reports/data/pl5_daily_<最新期号>.json
```

报告包含最新开奖、位置频率 Top、当前位置遗漏 Top、和值/跨度/形态、奇偶/大小分布、统计候选、候选回测和理性提示。当前数字彩报告只依赖本地 CSV，不自动联网下载；候选生成基于位置频率、当前遗漏和随机扰动，并默认排除上期原号、排除三位豹子、限制常用和值/跨度区间和高重复形态，仍不保证命中。回测只是把当前候选放回历史开奖中检查直选/组选命中情况，不能代表未来表现。

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
