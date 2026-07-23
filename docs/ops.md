# 运维指南

## 环境

- Python 3.11；推荐使用 `uv run --python 3.11 --with-requirements requirements-dev.txt`。
- 训练、评估和日报只读取本地 CSV，不联网。
- `make digit-fetch`和`make digit-predict-today`会访问固定开奖来源；后者仅在内存中合并，不覆盖CSV或影子状态。
- `digit_predict_today --ai`还会访问固定`api.deepseek.com`；默认关闭，密钥只从被Git忽略的本机配置读取。

## 命令

```bash
make digit-fetch DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make digit-predict-today DIGIT_LOTTERY=fc3d
make digit-probability-v5-development DIGIT_LOTTERY=fc3d DIGIT_V5_SMOKE=1
make digit-behavioral-context DIGIT_LOTTERY=fc3d
make digit-learned-ranker-train DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make digit-learned-ranker-evaluate DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make digit-learned-ranker-daily DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make ci
```

AI文案配置使用`config/ai.local.json`，结构参考`config/ai.example.json`。运行：

```powershell
.\.venv\Scripts\python.exe scripts\digit_predict_today.py --lottery fc3d --ai
```

AI失败不会改变模型结果，命令会保留确定性说明，并在`ai.status`中记录失败原因。

需要查看未准入的研究Top10时，显式增加`--show-research`；该输出不得称为正式推荐：

```powershell
.\.venv\Scripts\python.exe scripts\digit_predict_today.py --lottery fc3d --ai --show-research
```

冒烟可设置 `DIGIT_V4_SMOKE=1`；不得将 smoke 结果用于效果结论。

`probability_v5`使用独立的`DIGIT_V5_SMOKE=1`。该入口至少排除最新500期Frozen，只写`reports/development/`报告，不读取或生成任何v4/v5模型状态。完整开发前必须先执行`make digit-probability-v5-register`；协议和报告相同内容可重复确认，不同内容禁止覆盖。报告中的`validationOpened`、`promotionPassed`、`researchRankingEnabled`和`recommendationEnabled`必须全部为`false`；不得把开发Evaluation命名为Validation，也不得将smoke报告加入模型总账。

`make digit-probability-v5-null-smoke`会在均匀随机历史上完整重放四专家、在线聚合、Calibration和Search+Evaluation联合闸门。smoke次数必须少于5000且永不声明协议已登记或通过随机模拟闸门；正式模式必须传入只读协议/开发报告路径和锁定开发DataFrame、提供`--checkpoint-dir`并恰好运行5000次。协议Frozen边界、报告边界和历史期数必须精确匹配。检查点按完成顺序即时只写一次，重启时校验身份并只计算缺失编号；进程池任何阶段失败都直接失败关闭，不回退线程池。正式任务在全长度性能基准完成前不得启动。

训练参数会记录`validationPassed`和`validationReasons`。参数即使为审计而落盘，只要Validation未同时通过概率质量、目标预算命中和时间稳定性，`digit-learned-ranker-evaluate`就会在读取Frozen之前退出。

## 冻结边界

- 参数文件必须保留切分、数据、源码、参数和产物指纹。
- 参数文件必须为`validationPassed=true`且`smoke=false`。
- Frozen Test 不参与参数或候选预算选择。
- canonical 校验只覆盖训练时 `split.testEnd` 冻结前缀。
- 可以追加冻结边界之后的新开奖；修改冻结前缀必须使评估失效。
- 同一身份的 immutable 产物禁止覆盖。

## 监控

- 长任务应监控 Python 子进程 CPU，而不只看 `uv` 父进程。
- 评估产物只在完整结束后写出；运行中无文件不代表卡死。
- `evaluationValidation` 任一指纹或 frozen-test 证明失败，都必须关闭激活。
- `max_perm/mean_top_perm` 不得显示为百分比概率。

## 故障排查

1. CSV 列错误：先执行标准化入口或核对 `期号,开奖号码` 格式。
2. 源码指纹不匹配：旧参数不能继续用于新源码，应重新训练。
3. canonical 指纹不匹配：检查冻结前缀是否被修改。
4. 运行过慢：提高开发阶段 stride 或减少 trial；正式 Frozen Test 不应抽样。
5. 结果与随机相近：如实保留，不通过闸门，不继续查看 Frozen Test 调参。
6. 影子状态源码指纹不匹配：保留旧文件作为审计证据，生成新路径后显式切换，不得覆盖旧状态：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\digit_full_history_shadow.py --lottery fc3d --csv data/fc3d/official_history.csv --output state/learned_ranker_v4/full_history_shadow_fc3d_current.json
.\.venv\Scripts\python.exe -X utf8 scripts\digit_predict_today.py --lottery fc3d --shadow-state state/learned_ranker_v4/full_history_shadow_fc3d_current.json --no-fetch
```

不提供旧状态迁移入口；源码指纹变化后必须从官方历史完整重训，并把旧状态保留为审计证据。

行为v1～v4已完成全部固定块验证并封存，不再作为常规运维任务运行。统一证据总账使用：

```powershell
.\.venv\Scripts\python.exe scripts\digit_model_scoreboard.py --output-json reports/development/model_scoreboard_20260721.json --output-markdown docs/model_scoreboard.md
```

总账必须保持`selectedModel=null`和`productionMode=uniform_abstain`，除非新的独立前瞻证据同时通过LogLoss、Brier、Top50和时间稳定性。

## 清理说明

历史 v1、概率 v2、在线概率 v3 的可执行入口、状态、测试、文档和报告已删除。不要从旧产物恢复兼容路径；新功能直接在 learned ranker 架构下设计并重新验证。

## 快乐8选5开发挑战器

- 运行环境统一使用`uv`和Makefile，不探测或激活conda。官方抓取必须显式执行`make kl8-fetch`（只追加原始JSONL）或`make kl8-fetch-csv`（只允许首次创建`0444`的`data/kl8/kl8.csv`，差异内容或相同内容但可写的目标均拒绝）；程序只允许`cwl.gov.cn`固定接口及`name=kl8`，重定向主机仍需命中白名单。正`periods`必须恰好返回请求量，0必须恰好返回接口宣告总量。
- 快乐8规范CSV为`issue,date,numbers`。加载器第一遍只读取全部期号/日期确定最新500期Frozen，第二遍仅解析开发区号码；Frozen号码即使损坏也不得被读取或触发解析错误。
- 完整开发先执行`make kl8-pick5-register`；协议发布后自动为只读，再执行`make kl8-pick5-development`。正式路径只接受逐字段等于默认值的唯一规范配置，且Frozen排除期数固定500；通用smoke可缩短分段/重采样，但不得绑定协议或报告，也不得声称已登记。
- 协议、开发报告、null报告和逐试验检查点均只写一次；发布路径使用临时文件、`fchmod 0444`、文件`fsync`、原子硬链接和目录`fsync`，JSON序列化禁止NaN/Inf。恢复时任何配置、种子、数据、边界、字段集合、数值范围或哈希不一致均失败关闭。
- `make kl8-pick5-null-smoke`只允许少量迭代验证链路。正式null至少5000次，必须使用只读协议/报告、锁定开发DataFrame和独立检查点目录；多进程错误不得回退线程。显式执行`make kl8-pick5-null-formal`时默认8 worker，当前10核/16GB机器真实checkpoint基准折算约3.70小时；不得把估算当成已完成证据。
- 业务监控使用`meanHitsPerTicket`、`meanPortfolioTotalHits`、`exactPortfolioTotalHitsPValue`及最佳票`>=3/>=4/=5`比例；不得把首票审计字段或单票超几何显著性外推为五票证据。
- `make kl8-pick5-predict-today`从不自动抓取、从不覆盖状态。当前独立Validation、Frozen和准入均未开放，因此正式候选固定为空；显式研究审计只对应锁定Frozen首期并输出目标边界，不得称为今天推荐；不存在未实现的accepted-report兼容入口。
- 生产监控应记录数据语义哈希、源码指纹、协议/报告哈希、Frozen边界、检查点完成数、试验集哈希和失败栈；不得记录未脱敏外部响应或任何密钥。
