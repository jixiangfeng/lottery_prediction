# 双色球官方历史管线与固定集成 v1 预注册协议

## 1. 定位与禁止事项

`ssq_ensemble_v1` 仅用于离线研究。它不是中奖概率模型，不构成投注建议，不接入其他玩法的生产状态、报告与推荐路径。

- 固定 `researchOnly=true`。
- 所有研究票固定 `predictionClaim=false`。
- `formalCandidates=[]`、`recommendationEnabled=false`；v1 不提供自动正式化入口。
- 不做网格搜索、特征搜索、参数搜索、结果后重试或 CLI 模型参数覆盖。
- 本次实现不抓取网络数据，不生成或猜测双色球历史。

## 2. 唯一数据来源与证据链

唯一允许的数据源为中国福利彩票发行管理中心官网接口：

```text
https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice
```

查询参数集合固定为 `name=ssq`、`pageNo`、`pageSize=100`、`issueCount`、`issueStart`、`issueEnd`、`dayStart`、`dayEnd`、`week`、`systemType=PC`。除 `pageNo` 外不得变化或增加参数；协议同时校验 `https`、主机、路径、重定向后的最终 URL 与查询参数。

每页必须满足：

1. `message=查询成功`；
2. `total` 为不超过 100000 的非负整数，且所有分页一致；
3. `result` 为不超过固定 `pageSize` 的列表；
4. 未到最后一页时不得为空或无新增期号；
5. 全历史抓取的去重期数必须恰好等于 `total`。

每条官方记录必须满足：

- `name=双色球`；
- `code` 为 5—12 位数字期号；
- `date` 可严格解析为 `YYYY-MM-DD`，只允许其后出现空格或中文/英文星期括号；
- `red` 为六个逗号分隔的两位数，范围 `01..33`、唯一且严格升序；
- `blue` 为一个两位数，范围 `01..16`。

同一期号的重复记录只有在日期、红球和蓝球完全一致时才合并；任一冲突使整次抓取或对账失败。

### 原始证据

`data/ssq/raw/history.jsonl` 只追加。每行保留：

- 固定 schema 与 `lottery=ssq`；
- 完整 `sourceUrl`；
- UTC/带时区 `fetchedAt`；
- 未改写的官方 `raw` 记录；
- `SHA-256(canonical({sourceUrl, raw}))` 的 `rawHash`。

追加使用进程文件锁、末尾 LF 检查、`flush+fsync`。对账重新验证每行 JSON、来源、时间、原始哈希与开奖语义。只有所有证据通过后，才原子替换 `data/ssq/official_history.csv`。CSV 表头固定为：

```text
issue,date,red,blue,source_url,raw_hash
```

CSV 使用 UTF-8 与 LF 行尾。无原始数据时 `ssq-reconcile` 返回成功但不动作；无规范 CSV 时 `ssq-evaluate` 同样不动作。

## 3. 固定基线专家

所有专家只是预声明的固定基线专家，不宣称具有质量或预测能力。

### 红球

1. 永久保留的均匀专家：每球包含概率 `6/33`。
2. 半衰期 30 期的边际 EWMA：对 33 个红球做 Laplace `α=1` 平滑，并归一到概率和为 6。
3. 半衰期 120 期的边际 EWMA：同上。
4. 半衰期 120 期的 528 对共现 EWMA：对每期 15 对红球做 Laplace 平滑，计算相对均匀对概率的 log-ratio，裁剪到 `[-1,1]`，再以固定权重 `0.20` 作为组合分数修饰器。

### 蓝球

1. 永久保留的均匀专家：每球概率 `1/16`。
2. 半衰期 60 期的 16 类 EWMA，Laplace `α=1`。

红球三名概率专家与蓝球两名概率专家分别使用固定 `eta=0.25` 的指数权重在线聚合。每一期必须按以下顺序执行：

1. 从当前状态生成事前概率与候选；
2. 用当期开奖计算专家与集成 proper score；
3. 更新指数权重；
4. 最后更新 EWMA 和共现状态。

任何当前期或未来期开奖都不得影响本期事前输出。

## 4. 固定研究 Top20

红球组合分数为六个边际 logit 之和，加上有界 pairwise 修饰。使用固定束宽 `256` 的束搜索构造合法 `6-of-33` 升序红球组合，再与全部 16 个蓝球联合排序。

- `TopK=20` 固定；
- 分数相同按红球元组、蓝球字典序升序；
- 号码合法、唯一且输出确定；
- `rankingScore` 仅用于排序，不是开奖概率；
- 每张研究票显式带 `predictionClaim=false`。

报告另在 `auditMetadata` 中绑定当前下一期的确定性预开奖排名证据，且不改变
专家、概率、在线更新、验证门禁或正式状态：

- `orderedRed6Combinations`：直接取既有
  `beam_red_combinations(next_red, next_pairs)` 的前 32 个有序红 6 组合；每项
  保留连续 `rank`、两位数字红球和既有 `redScore`，不新增搜索或调参。
- `orderedRed6CombinationCount=32`。
- `researchBlueTop1`：直接取既有 `blue_top1(next_blue)` 的模型 Top1 蓝球。

这些字段只提供报告哈希绑定的审计证据，供只读统一研究参考确定性派生结构；
不构成新模型输出，也不改变 `protocolSha256`。

## 5. 评估与硬闸门

- 固定预热 120 期；预热期只更新前序状态，不纳入证据。
- 之后按时间顺序切分全部不重叠完整 100 期块。
- 尾部不足 100 期明确计入 `excludedIncompleteTailDraws`，不参与硬闸门。

主要指标：

1. 红球 Top6 平均命中数，对照均匀期望 `36/33`；
2. 蓝球 Top1 命中率，对照 `1/16`；
3. 33 个红球逐球 Bernoulli LogLoss 与 Brier；
4. 蓝球 16 类 LogLoss 与多分类 Brier。

精确 Top20 整票命中仅作描述，均匀单期概率为：

```text
20 / (C(33,6) * 16)
```

由于该事件极度稀疏，不作为主要证据或独立准入依据。

硬闸门要求同时满足：

- 至少存在一个完整评估块；
- 每个完整块的四项 proper score 相对均匀基线的平均比值不大于 1；
- 每个完整块的红蓝覆盖相对均匀基线的平均比值不小于 1；
- 汇总层红/蓝 LogLoss、Brier 分别不劣于均匀基线；
- 汇总层红球 Top6 与蓝球 Top1 分别不低于均匀期望。

失败时输出 `decision=uniform_abstain`，正式候选与推荐保持为空。即使研究硬闸门通过，v1 仍保持研究专用，不自动启用正式推荐。

## 6. 运行入口

```bash
make ssq-fetch       # 显式联网；只追加官方原始证据
make ssq-reconcile   # 全量验证后生成LF规范CSV
make ssq-evaluate    # 固定协议严格前序评估
```

执行环境使用 Python 3.11 的 `python311` conda 环境，或已按 `requirements*.txt` 锁定依赖的等效虚拟环境。
