# 福彩3D、排列三、排列五分析工具

本项目面向三位和五位数字彩的离线历史研究，提供 CSV 标准化、统计分析、候选排序、推荐留痕和严格逐期前推回测。所有评分只表示历史排序信号，不是开奖概率，也不保证中奖或盈利。

## 功能清单

- 福彩3D：位置统计、直选/组选候选、形态分析和严格前推。
- 排列三：与福彩3D共享三位数字分析能力和独立玩法规则。
- 排列五：五位置统计、10 万号码空间向量化评分和直选前推。
- 多窗口统计：30/50/100/300 期及当前全历史窗口。
- 增量快照：首次全量、无变化缓存命中、纯追加更新和历史修正检测。
- 候选排序：14 个统计模型加蒙特卡洛、逻辑回归两个可选模型。
- 推荐闭环：开奖前保存候选，数据更新后自动复盘并累计模型表现。
- 严格前推：每个目标期只使用此前历史，并与多次均匀随机基线比较。
- 可行性闸门：直选、组选分别检验，只有长期、显著、稳定地高于精确随机基线才标记为可行。
- 概率 v2：完整1000状态归一化、严格历史校准、独立守门和失败时均匀回退。
- 在线概率 v3：每期先预测再按开奖结果更新14个统计模型权重，完整记录权重轨迹；可作为独立日报模式运行。
- 固定评分 v4：完整 `000-999` 多窗口特征、贝叶斯平滑、可选半衰期衰减、固定线性权重、严格 search/validation/frozen-test 隔离和研究日报。

## 快速开始

    conda activate python311
    make setup
    make ci

运行不依赖外部数据的最小示例：

    make run

## 数据格式

日报和回测不会自动访问网络。可以提供自己的本地 CSV，也可以显式抓取福彩3D或排列三公开历史：

    make digit-fetch DIGIT_LOTTERY=fc3d DIGIT_FETCH_PERIODS=1000
    make digit-fetch DIGIT_LOTTERY=pl3 DIGIT_FETCH_PERIODS=1000

抓取只访问代码内固定的 `www.cwl.gov.cn` 和 `jc.zhcw.com` 白名单，带超时、重试、字段校验和原子写入。默认写入 `data/<彩种>/data.csv`，该目录不提交到 Git。排列五继续由调用方提供本地 CSV。

支持合并号码：

    期号,开奖号码
    2026001,012

也支持分位置列：

    期数,百位,十位,个位
    2026001,0,1,2

排列五分位置列为 万位、千位、百位、十位、个位。期号必须是数值唯一的纯数字；系统按数值排序，避免字符串顺序造成未来数据混入。

## 生成日报

    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv
    make digit-report DIGIT_LOTTERY=pl3 DIGIT_CSV=data/pl3/data.csv
    make digit-report DIGIT_LOTTERY=pl5 DIGIT_CSV=data/pl5/data.csv

默认生成 Markdown 和 JSON：

    reports/<彩种>_daily_<期号>.md
    reports/data/<彩种>_daily_<期号>.json
    reports/state/<彩种>_statistics_snapshot.json
    reports/picks/digit/<彩种>_<源期号>.json

日报默认启用集成排序、蒙特卡洛和轻量逻辑回归。可通过 config/.env.example 中的变量调整，复制到项目根目录 .env 后 Makefile 会自动读取。

普通日报允许同一期重新生成；正式开奖前实验必须显式冻结，已冻结的同源期快照不能被后续普通运行覆盖：

    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_FREEZE_PICK=1

## 固定评分算法 v4

v4 首版只支持 `fc3d` 和 `pl3`；`pl5` 会明确拒绝并保持旧路径。完整设计见 `docs/learned_ranker_v4_design.md`。正式流程：

    make digit-learned-ranker-train DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
    make digit-learned-ranker-evaluate DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
    make digit-learned-ranker-daily DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv

完整参数搜索较慢时可先做流水线冒烟；它只减少 trial 和目标期，不会把核心算法替换为占位实现：

    make digit-learned-ranker-v4 DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv DIGIT_V4_SMOKE=1

也可通过原日报入口生成 v4 研究日报：

    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv DIGIT_RANKING_MODE=learned_ranker_v4

主要产物：

- `reports/state/learned_ranker_v4/<彩种>_params.json`：参数、搜索证据、CSV/源码/参数指纹。
- `reports/evaluations/learned_ranker_v4_<彩种>.md/.json`：冻结测试 LogLoss、Brier、排名、TopK、精确组选随机基线、p 值和分块闸门。
- `reports/learned_ranker_v4_daily/<彩种>_<实验ID>_<参数指纹前缀>_daily_<期号>.md/.json`：直选/组选、位置池、组选数字池和分项激活状态。
- `reports/picks/digit/<彩种>_learned_ranker_v4_<实验ID>_<参数指纹前缀>_<源期号>.json`：按实验与参数隔离、内容不同即拒绝覆盖的冻结快照。

公共概率/稳定性闸门通过后，直选和组选按各自命中闸门独立启用；未通过部分必须留在研究分区。归一化概率只是评分转化，不是实际开奖概率，不保证中奖或盈利。

## 严格前推

    make digit-walk-forward DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_WF_PERIODS=50
    make digit-walk-forward DIGIT_LOTTERY=pl3 DIGIT_CSV=data/pl3/data.csv DIGIT_WF_PERIODS=50
    make digit-walk-forward DIGIT_LOTTERY=pl5 DIGIT_CSV=data/pl5/data.csv DIGIT_WF_PERIODS=30

前推报告写入 reports/evaluations。排列五完整空间和高级模型计算成本较高，建议先用 5 至 10 个目标期冒烟，再扩大窗口。

前推 JSON 与 Markdown 会对 `current_statistics`、`ensemble_voting` 分别给出统计可行性结论。通过条件固定为：至少 500 期、精确单侧 `p<0.01`、相对随机提升至少 25%、99% Wilson 下界高于随机基准，并且三个非重叠时间块都不低于随机。三位彩直选与组选必须各自通过；任一条件失败都显示“不通过”。

福彩3D和排列三默认在完整 `000-999` 空间评分。和值、跨度、豹子和上期原号不会被默认硬排除，历史结构信号只作为排序依据；研究显式过滤策略时仍可通过 `DigitCandidateConfig` 指定范围和形态。

### 2026-07-17 官方数据 500 期结论

完整16模型、20次随机基线的严格前推结果均未通过整体闸门。福彩3D集成直选为 `15/500`，随机期望 `5`，但三个时间块为 `8/6/1`，最近150期为 `0` 次，稳定性失败；组选无显著优势。排列三的统计与集成直选、组选均未显著高于随机。详见 `docs/official_500_validation.md`。

福彩3D集成算法已以 `2026187` 为数据截止期登记 `fc3d_ensemble_forward_v1`。只有后续开奖前快照可以进入新样本，满500期前不作可行性声明。固定参数、初始指纹和候选见 `docs/fc3d_forward_protocol.md`。

### 概率 v2 开发评估

    make digit-probability-walk-forward DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_WF_PERIODS=500
    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_RANKING_MODE=probability OUTPUT_DIR=reports/probability_v2

概率模式精确枚举 `000-999`，校准段前2/3选参、后1/3独立守门；任一时间块的 Log Loss 未优于均匀分布就把学习权重设为0。直选按概率纯Top10，组选按排列概率求和。

官方数据500期开发评估中，福彩3D与排列三的学习剖面都在独立守门段变差，因此均回退到均匀分布，未建立新的前向实验。详见 `docs/probability_v2_development.md`。

### 在线概率 v3 开发评估

    make digit-probability-online DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
    make digit-probability-online DIGIT_LOTTERY=pl3 DIGIT_CSV=data/pl3/official_history.csv

在线模式使用前500期预训练权重，后500期严格执行“先预测、后开奖、再更新”。本次固定参数结果中，福彩3D和排列三 Log Loss 均略差于均匀分布，训练最终把约96.5%权重分配给均匀基线，未建立预测优势。详见 `docs/probability_v3_online_development.md`。

### 在线概率 v3 日报

    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv DIGIT_RANKING_MODE=online_probability OUTPUT_DIR=reports/probability_v3_online_daily DIGIT_JSON=1

首次运行会从第101期开始重放到最新已开奖期，状态写入 `OUTPUT_DIR/state/<彩种>_online_probability_v3_state.json`；下次只消费新增开奖期，再用更新后的权重预测下一期。若历史中间期号或号码被修正，状态指纹失配后自动全量重建。该模式默认不接管 `DIGIT_RANKING_MODE=ensemble`，因为在线 v3 500期开发评估尚未通过随机闸门。

## 关键配置

| 变量 | 默认值 | 说明 |
|---|---:|---|
| DIGIT_LOTTERY | fc3d | fc3d、pl3、pl5 |
| DIGIT_FETCH_PERIODS | 1000 | 显式抓取最近期数，仅支持 fc3d、pl3 |
| DIGIT_CANDIDATE_COUNT | 10 | 候选注数 |
| DIGIT_FREEZE_PICK | 0 | 设为1时冻结同源期推荐快照，禁止覆盖 |
| DIGIT_RANKING_MODE | ensemble | ensemble、composite、probability、online_probability 或 learned_ranker_v4 |
| DIGIT_ENABLE_MONTE_CARLO | 1 | 是否启用蒙特卡洛票 |
| DIGIT_ENABLE_ML | 1 | 是否启用逻辑回归票 |
| DIGIT_CONSTRAINT_MODE | soft | off、soft、hard |
| DIGIT_WF_BASELINE_RUNS | 20 | 随机基线重复次数 |
| DIGIT_WF_NESTED_TUNING | 0 | 是否启用内层未见期调参 |
| DIGIT_PROBABILITY_VALIDATION_PERIODS | 180 | 概率 v2 冻结前校准期数 |
| DIGIT_PROBABILITY_MIN_VALIDATION_PERIODS | 90 | 启用学习概率的最少验证期数 |
| DIGIT_ONLINE_PERIODS | 500 | 在线概率 v3 评估期数 |
| DIGIT_ONLINE_MIN_TRAIN_SIZE | 100 | 在线日报开始反馈前的历史期数 |
| DIGIT_ONLINE_FIXED_SHARE | 0.01 | 每期开奖后向初始权重收缩比例 |

## 目录结构

    src/lotteries/       三种玩法的强类型规则
    src/analysis/        数字彩统计、候选、报告和前推
    scripts/             日报和前推 CLI
    tests/               单元与回归测试
    examples/            无外部数据的最小示例
    reports/             历史评估产物
    config/              环境变量模板

## 测试与质量

    make fmt
    make ci

make ci 会执行 Black/isort 检查、flake8、mypy、Pytest 覆盖率和语法构建。覆盖率门槛为 80%。

## 常见问题

1. 找不到 CSV：确认 DIGIT_CSV 指向存在的本地文件，并检查列名是否符合“数据格式”。
2. 排列五运行慢：降低前推期数、蒙特卡洛次数或临时关闭高级模型。
3. 快照自动重建：历史删减、已处理期修正、规则/窗口/先验变化都会触发全量重建。
4. 候选分是否是概率：复合分、集成分和模型权重不是概率；只有 probability 或 online_probability 模式的 `predictedProbability` 是和为1分布中的概率，守门失败或在线模型无增量时会接近均匀概率。
5. 回测能否证明未来有效：不能。只有严格前推和开奖前留痕能减少数据泄漏，仍无法消除随机波动。
6. 什么情况下算法可投入候选实验：只有前推报告中的对应策略显示“通过”才具备统计实验资格；“不通过”时应保持观察，不能提高投注金额。

## learned_ranker_v4 完整化说明

- 首版仅支持 `fc3d`、`pl3`，`pl5` 会明确拒绝；旧 `ensemble/probability/online_probability` 默认行为不变。
- 特征使用 `10/20/30/50/100/300/all` 独立窗口权重；position、pair、sum、span、shape 分别保留 `30-300` 差、`50-all` 差和 `30/300` log-ratio，聚合趋势只作为兼容摘要。
- 正式搜索 manifest 声明特征权重边界、窗口集合/权重 profile、alpha、half-life、omission cap、temperature、组选聚合、归一化方式及固定同成本推荐配置；单次运行使用 seed 驱动的有界采样与惰性特征准备，不声称穷举整个笛卡尔积。
- 冻结闸门拆分为公共概率/稳定性、直选命中、组选命中；只有公共闸门与对应分项闸门同时通过，才进入 `mainRecommendation`，失败部分仅显示在研究分区。
- v4 快照包含实验、参数、当前 canonical 数据、冻结前缀、源码、激活状态和候选指纹；新增开奖只改变当前历史指纹，不会撤销冻结前缀证据，修改冻结前缀则会关闭激活。
- `sum_prob` 才输出组选 probability；`max_perm/mean_top_perm` 仅输出 score 与 aggregation，不解释为真实概率。
- 本地已获取并校验 fc3d/pl3 各 1000 期官方历史；最终源码 smoke 选参后的完整 250 期冻结测试中，两种彩票 LogLoss 均劣于均匀分布且直选/组选 p 值均未过闸门，因此当前全部保持研究模式。默认正式搜索预算仍可继续研究，但不得把搜索次数包装成效果证据。
