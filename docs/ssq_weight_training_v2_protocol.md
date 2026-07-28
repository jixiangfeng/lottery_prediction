# 双色球静态约束权重训练 v2 协议

## 1. 定位与隔离

`ssq_weight_training_v2` 是叠加在官方数据管线与固定 `ssq_ensemble_v1` 之上的一次性研究挑战。它不修改 v1 的在线指数权重、硬闸门、报告字段或决策语义，不写入任何生产状态，也不影响其他玩法。

输入固定为本地 `data/ssq/official_history.csv`。本任务禁止联网；数据仍由既有官方证据链负责抓取、校验与对账。v2 复用 v1 的红球 EWMA30/EWMA120、蓝球 EWMA60、红球 pairwise EWMA120、Top6/Top1 与研究 Top20 生成逻辑。每期必须先读取严格前序专家状态，再评分，最后才用当期开奖更新专家。

## 2. 不可覆盖常量

| 项目 | 固定值 |
|---|---:|
| 预热 | 120 期 |
| Frozen Test | 最新 500 期 |
| Validation | Frozen 之前 500 期 |
| Search | 预热后、Validation 前全部更早期次 |
| Validation/Frozen 块长 | 100 期，共 5 个完整块 |
| 红球专家顺序 | `[uniform, ewma30, ewma120]` |
| 蓝球专家顺序 | `[uniform, ewma60]` |
| 权重约束 | 非负、和为 1 |
| 网格步长 | 0.10 |
| 红球候选数 | 66 |
| 蓝球候选数 | 11 |
| 联合候选数 | 726 |
| pairwise 修饰权重 | v1 固定 0.20，不搜索 |
| TopK | v1 固定 Top20，仅描述 |
| 重试 | 0 |

CLI 只允许 `--csv` 与 `--output`。窗口、特征、TopK、网格步长、目标函数、分段长度、闸门和 pairwise 权重均无覆盖入口。

## 3. Search 唯一选权

所有 726 个候选只在 Search 上使用严格前序专家输出评估。四项 proper score 分别为：

1. 红球逐球 Bernoulli LogLoss；
2. 红球逐球 Brier；
3. 蓝球 16 类 LogLoss；
4. 蓝球 16 类 Brier。

覆盖指标为红球 Top6 平均命中数与蓝球 Top1 命中率。均匀基线固定沿用 v1 理论值：红球 `36/33`、蓝球 `1/16`；四项 proper score 也使用 v1 的均匀理论值。

候选只有同时满足以下六项条件才合格：四项 proper score 各自不高于均匀基线，红球与蓝球覆盖各自不低于均匀基线。非均匀合格候选按下列固定顺序选择唯一赢家：

1. 四项 proper score 相对均匀值的平均归一化指数最低；
2. 红蓝覆盖相对均匀值的平均归一化指数最高；
3. 红球权重元组、再蓝球权重元组按字典序最小。

若不存在合格的非均匀候选，立即选择 `uniform_abstention`。Validation 与 Frozen 均保持未开封，不允许更换目标、缩小网格、修改窗口或重试。

## 4. 条件式单次开封

只有 Search 冻结了非均匀唯一赢家，才允许用同一权重顺序评估 Validation 一次。Validation 的 5 个固定 100 期完整块及 500 期汇总都必须逐项通过上述四项 proper score 和两项覆盖门槛；任一失败即终止，Frozen 保持未开封。

只有 Validation 全部通过，才允许用完全相同权重评估 Frozen 一次。Frozen 同样要求 5 个完整块与汇总逐项通过六项门槛。Frozen 失败或通过后都禁止再调参、改变目标、挑选其他 Search 候选或重跑挑战。

## 5. 报告与停止规则

JSON 报告必须包含规范化数据 SHA-256、官方来源证据身份 SHA-256、CSV 文件 SHA-256、v2 协议 SHA-256、所复用 v1 协议 SHA-256、精确分段边界、全部 Search 候选摘要、冻结权重、Validation/Frozen 开封标志、汇总与完整块指标、决策及失败处置。

精确 Top20 命中与均匀期望 `20/(C(33,6)*16)` 只作稀疏描述，不参与候选合格性、排序或任何门槛。所有结果固定 `researchOnly=true`、`recommendationEnabled=false`、`productionActivation=false`、`formalCandidates=[]`。

## 6. 运行

```bash
make ssq-weight-train-v2
```

等价命令：

```bash
python -m scripts.ssq_weight_training_v2 \
  --csv data/ssq/official_history.csv \
  --output reports/research/ssq_weight_training_v2.json
```
