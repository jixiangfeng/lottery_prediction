# 运维指南

## 环境

- 使用名为 python311 的 Conda 环境或等效 Python 3.11 虚拟环境。
- 运行依赖已在 requirements.txt 固定直接版本。
- 日报和回测只从本地 CSV 读取，不需要网络权限。
- 只有显式 `make digit-fetch` 需要网络；来源固定为 `www.cwl.gov.cn` 和 `jc.zhcw.com`，不接受命令行自定义 URL。

## 日常命令

    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv
    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_FREEZE_PICK=1
    make digit-fetch DIGIT_LOTTERY=fc3d DIGIT_FETCH_PERIODS=1000
    make digit-walk-forward DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_WF_PERIODS=30
    make digit-probability-walk-forward DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/data.csv DIGIT_WF_PERIODS=500
    make digit-probability-online DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
    make digit-report DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv DIGIT_RANKING_MODE=online_probability OUTPUT_DIR=reports/probability_v3_online_daily DIGIT_JSON=1
    make ci

## 监控

- 检查 statisticsUpdate.mode 是否符合 full_rebuild、cache_hit、incremental 或 stale_view。
- 检查前推每期 selectedConfigTrainEndIssue 小于目标 issue。
- 监控候选过滤空间大小、模型激活数和真实开奖号分位桶。
- 排列五先小窗口运行，避免一次扩大期数、基线次数和高级模型成本。
- 三位彩完整 500 期、16 模型并行验证在本机约耗时 50 分钟；运行前确认CPU时间窗口，后续应增加逐期缓存与进度输出。
- 保留 reports/picks/digit 中的开奖前快照，不得开奖后改写。
- 冻结实验每期都应在开奖前登记 CSV 哈希、推荐指纹和候选；本地不可变标记不能替代可信时间戳。
- 概率日报建议使用独立 `OUTPUT_DIR=reports/probability_v2`；校准失败时检查 `probabilityCalibration.fallbackReason`，不得绕过守门强制启用权重。
- 在线概率 v3 检查每期 `weightsBefore` 与上一期 `weightsAfter` 一致，并确认 `trainEndIssue` 小于目标期；评估段结束前不得修改固定温度、学习率和收缩比例。
- 在线日报检查 `onlineProbability.stateUpdate.mode`：首次应为 `full_rebuild`，仅新增开奖应为 `incremental`，无新增应为 `cache_hit`；`historyFingerprint` 失配时允许自动 `full_rebuild`。
- 在线日报首次全量重放可能较慢；后续只重算新增目标期，状态文件和开奖前推荐快照均需纳入备份。

## 故障排查

1. CSV 列错误：先核对 README 的两种输入格式。
2. 快照损坏：使用 --rebuild-stats 强制重建。
3. 运行过慢：降低 Monte Carlo 次数、前推期数或 baseline-runs。
4. 结果与随机相近：如实保留结果，不以候选分替代命中证据。
5. 并发写入异常：检查目标目录权限和文件系统是否支持 flock、fsync、os.replace。
6. 官方抓取失败：确认域名可访问后重试；字段校验失败时保留旧 CSV，不要绕过校验。

## learned_ranker_v4 运维

- 使用 `make digit-learned-ranker-v4 ... DIGIT_V4_SMOKE=1` 做短流程冒烟；正式报告需去掉 smoke 并保存完整搜索空间。
- 参数文件必须同时保留 `csvSha256`、`sourceFingerprint`、`paramsFingerprint`、`artifactFingerprint`、切分边界和 `testSegmentUsedForSelection=false`。
- 评估 JSON 是日报闸门的唯一依据；只有 `reportFingerprint`、玩法、源码、参数产物和冻结 test 证明全部匹配时才可晋级，缺失、损坏或 `gate.passed=false` 时日报必须保持研究模式。
- 同一源期、实验、参数的冻结快照重复运行应字节一致；不同实验或参数使用独立路径，已有同身份产物内容不同则拒绝覆盖。
- v4 训练、评估和日报本身不联网；CSV 由调用者提供，也可先用显式固定白名单抓取命令生成。运行前确认 Python 3.11 等效环境及磁盘空间。
# learned_ranker_v4 运维补充

- 推荐使用 `conda activate python311`；若主机无 conda，可在仓库内创建 `.venv` 并安装 `requirements.txt`、`requirements-dev.txt`。
- 正式运行前保存参数、冻结评估和日报目录；同一期同实验同参数快照不可覆盖，冲突表示输入或代码已变化，应新建 experimentId。
- 监控 `evaluationValidation` 的 rule/params/artifact/source/canonical/fingerprint/frozen-test 匹配项；canonical 校验只针对训练时 `split.testEnd` 冻结前缀，允许其后追加新开奖但拒绝前缀修订；任一证据失败都必须关闭 direct/group activation。
- 开奖更新后先运行 daily：程序先复盘旧 immutable 快照，再生成当前 sourceIssue 的新快照，避免事后覆盖。
- v4 live summary 文件名包含玩法、experimentId 和参数指纹前缀；禁止人工合并不同参数或 v1-v3 汇总。
- `max_perm/mean_top_perm` 日志和报告只能标记 score/aggregation，不得转换为百分比概率。
- 真实启用前必须在冻结样本中优于同成本随机基线；无真实 CSV 时只允许合成数据 smoke，并在报告中标记无效果结论。
