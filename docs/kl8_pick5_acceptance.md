# 快乐8选5验收矩阵

> 当前证据状态：`exploratory_reused_development`。本表的“完成”仅表示实现/契约完成，不表示模型有效或可推荐。

| 需求 | 状态 | 验收证据 |
|---|---|---|
| 快乐8玩法注册与20个唯一`1..80`号码校验 | 完成 | `src/lotteries/kl8.py`、数据契约测试 |
| 期号/日期校验与时间正序语义哈希 | 完成 | `canonical_kl8_sha256`及逆序同哈希测试 |
| Frozen 500仅读元数据、不解析号码 | 完成 | 损坏Frozen号码仍成功加载1550期开发数据测试 |
| 固定六专家及每专家80维`float64`、和为20 | 完成 | 直接构造状态验证六输出有限、形状`(6,80)`且逐行和为20 |
| PairwiseAdjustedExpert非退化关联修正 | 完成 | 事前EWMA80 Top20上下文、边际外积归一lift、排除self；保持边际不变并交换关联身份时pair专家响应改变且不同于EWMA80 |
| 严格先预测、后读开奖、再更新 | 完成 | 当前/未来开奖扰动泄漏回归测试 |
| Hedge裁剪损失与更新前后权重 | 完成 | 每期`expertWeightsBefore/After`和扰动测试 |
| Calibration仅固定温度网格、完整配置250期 | 完成 | Evaluation开奖变化不影响温度/首期事前概率；Calibration变化可改变温度但Search逐字段不变 |
| Top20内生成五个唯一组合、确定性并列规则 | 完成 | 组合唯一性/确定性测试；受控等概率下启用集中惩罚降低总重叠；未枚举`C(80,5)` |
| 80标签LogLoss/Brier与Uniform Delta | 完成 | Search/Evaluation独立指标与联合闸门 |
| 五票匹配成本精确PMF和五块稳定性 | 完成 | 每期按80个号码在五票中的重数`0..5`分组，动态规划精确抽取20个号码并跨期卷积；tiny总体暴力枚举逐项匹配 |
| 固定种子连续块bootstrap，完整配置≥2000次 | 完成 | `bootstrap_seed=20260722`、默认2000次；smoke显式缩短 |
| Search与Evaluation均需边际+五票业务闸门 | 完成 | 两项bootstrap `p<=alpha`；五票总命中精确`p<=alpha`、每票均值≥1.25、组合总均值≥6.25及五块每票稳定性；Search=false/Evaluation=true仍总失败 |
| 逐期五票与分布报告 | 完成 | `combinationHits`固定长度5，派生`portfolioTotalHits/portfolioBestHits`；报告票级、组合总命中、最佳票分布及最佳票`>=3/>=4/=5`比例 |
| 空正式候选与显式研究审计 | 完成 | `userVisibleCandidates=[]`；研究候选是最终状态对下一未知期的预测，不复用最后Evaluation组合 |
| 通用入口不可伪造协议登记 | 完成 | 公共签名无`protocol_identity`；通用报告身份为空且登记标志为false |
| 只读协议路径与确定性重算 | 完成 | 协议绑定加载器实际Frozen首末期；首期或末期变化在执行前拒绝 |
| 唯一规范正式配置 | 完成 | `__post_init__`拒绝非有限/越界值并固定Top池20、输出5、五块、Frozen 500、null≥5000；协议构建/加载、登记开发、报告加载、正式null和非smoke CLI均逐字段比较默认配置，smoke不得进入登记路径 |
| 自哈希报告与篡改后重哈希拒绝 | 完成 | 完整重算差异测试 |
| 原子不可覆盖写入 | 完成 | 临时inode先`fchmod 0444`并文件fsync，再硬链接和目录fsync；同内容可写目标也拒绝 |
| 源码指纹覆盖两核心模块和四CLI | 完成 | 共6个文件：probability/null核心及fetch/development/null/predict脚本 |
| 官方抓取白名单、HTTPS、精确数量与分页保护 | 完成 | 最终URL必须为`https`白名单主机；正`periods`恰好返回请求量，0恰好返回接口宣告量；宣告变化或未达目标无进展失败关闭 |
| JSONL进程间安全只追加 | 完成 | `fcntl.flock(LOCK_EX)`、既有非空文件末字节换行校验、`flush+fsync`后解锁；损坏尾部保持原样并拒绝追加 |
| null全流程、逐试验/有序集合哈希 | 完成 | 随机无放回20/80、`trialSha256`、有序集合哈希 |
| null检查点精确键、数值与恢复篡改拒绝 | 完成 | 每个trial持久化Search/Evaluation规范门控输入：完整顶层门控字段、两项bootstrap非正均值p值及五块`deltaLogLoss/deltaBrier/meanHitsPerTicket`；恢复时以生产`_segment_gate`分别重导两段`passed`及联合结果。嵌套键集合严格相等，`index/seed`必须为JSON整数，所有统计量和专家权重必须为非布尔JSON数值并满足有限值/合理范围；翻转布尔或篡改门控输入后重哈希均拒绝 |
| null恢复新完成顺序 | 完成 | 报告仅输出`newCompletionOrder`，只记录本次新计算编号；不再提供旧`completionOrder`字段 |
| 正式null判定与权重摘要 | 完成 | 联合FPR及Delta LogLoss/Brier、每票均值、组合总均值、最佳票`>=3`、`>=4`、`=5`七项经验p值均`<=alpha`；六专家各7项分布统计 |
| 正式null字面5000公共/CLI边界 | 完成 | `FORMAL_MIN_ITERATIONS=5000`且配置构造拒绝`required_null_iterations<5000`；4999在加载历史/模拟前拒绝，5000接受 |
| 多进程错误失败关闭、无线程回退 | 完成 | 进程Future异常契约测试 |
| 今日预测不联网、不覆盖状态、空候选 | 完成 | `scripts/kl8_pick5_predict_today.py`固定安全边界；删除未实现的accepted-report空入口，显式审计绑定Frozen首期并声明不是今日推荐 |
| 1550开发+500 issue-only Frozen CLI smoke | 完成 | 输入2050行；引擎smoke使用50期，加载器验证1550期开发数据且不解析500期空Frozen号码 |
| 正式至少5000次null | 阻塞 | 未执行；不得用smoke替代 |
| 全新独立500期Validation | 阻塞 | 尚无未来独立数据，`validationOpened=false` |
| Frozen评估 | 阻塞 | 未开启且号码未读取，`frozenRead=false` |
| 晋级与正式推荐 | 阻塞 | `promotionPassed=false`、`recommendationEnabled=false` |

## 本轮实测

- KL8聚焦契约`63 passed`，新增组合incidence重叠等价、快慢选择器逐组一致和向量化bootstrap逐元素一致测试；其余规范配置、五票PMF、门控重导及抓取契约继续通过。
- 完整`make ci`：`289 passed`，总覆盖率`86.33%`；Black、isort、flake8、mypy与compileall全部在主环境直接通过。
- 最新源码下已用1550期synthetic开发数据和500行故意损坏号码的Frozen区完成只读协议登记、开发报告生成及从路径完整重算：协议`c6bff9bd829ee968503b769ea05ce7e046d63e07deafdfc21a18dd3e55a3ccbb`，报告`8bf82853ac21b51c7cbed9a363694af14263f6f3a4d29108162387cdae6a8d49`，两者模式均为`0444`；`frozenRead=false`、`promotionPassed=false`、正式候选为空。
- 优化前完整单trial为`35.37秒`；优化后三次无profile实测为`11.37/11.37/11.31秒`，均值`11.35秒`，后续bootstrap/二分提前收敛后单次为`9.10秒`且trial SHA-256保持`bcbe0db379edb0a123c87ab12b8aa92ff0c0626b1f526c37f8b8a1a636a0880`不变。4进程×8次checkpoint实测`23.53秒`（5000次折算`4.09小时`）；8进程×16次实测`42.59秒`（折算`3.70小时`）。
- 最新抓取器从福彩官网白名单实际抓取3期成功，标准CSV为`0444`；JSONL SHA-256为`e251c357a693a85d5007575ac5008d912177fc9f5e61dedc71ff71d04978eafd`，CSV SHA-256为`227f5465e9200511d95095d5412977ada81a60536d039626f786c3cadbaf0e43`。
