# 本次自动执行报告

## 需求摘要

- 改善日常预测的生硬输出，解释为何不展示大量候选。
- 保留研究审计数据，并增加默认关闭的可选AI文案层。

## 关键假设

- AI只做文案，不参与号码生成、排序或准入。
- 正式策略未激活时，用户可见候选必须为空。

## 实现与自测

- 日常JSON新增状态、放弃原因、信号指标、验证进度及候选特征贡献。
- 默认终端隐藏研究Top50；`--json`保留完整审计数据。
- `--ai`使用固定DeepSeek Chat API，凭据从本机忽略配置读取；缺少配置或请求失败时回退确定性说明。
- `--show-research`显式展示研究Top10及前三名特征贡献，不改变准入状态。
- 新增入口测试6项全部通过；全仓Black、isort、flake8、compileall和新增模块mypy通过。
- 跳过Windows不支持的既有`fcntl`快照模块后，其余`140 passed`，覆盖率`80.32%`。

## 风险与后续

- AI文案质量依赖调用方显式配置的模型，但不能改变结构化模型事实。
- 当前模型仍为前瞻影子研究状态，不构成正式推荐。
- Windows全量测试收集仍被`digit_statistics_snapshot.py`无条件导入`fcntl`阻断；该问题不在本次改动范围。

## 2026-07-21 v4行为挑战器加固

### 需求摘要

- 继续完善`learned_ranker_v4 + behavioral_context_v1`，暂不新增v5模型。
- 统一行为回测与日常Top50口径，补齐Frozen隔离、稳定性和状态完整性校验。

### 实现与自测

- A/B Top50逐期排除上期原号、按配置限制豹子并补足50个；概率损失仍使用未过滤的1000候选事前分布。
- 行为CLI默认排除最后500期Frozen，支持`--all-development-blocks`扫描开发区全部完整500期块。
- 晋级闸门新增零放弃期和全部完整固定块稳定性要求；报告新增数据、源码和目标期指纹。
- 影子状态加载新增模型、彩种、源码和内容指纹校验，旧文件不得静默用于新代码。
- 针对性回归`16 passed`；排除既有Windows `fcntl`模块后全仓`149 passed`，覆盖率`80.47%`；Black、isort、flake8、compileall及本次3个核心模块mypy通过。
- Windows全量`make ci`仍被既有`digit_statistics_snapshot.py`的4个`fcntl` mypy错误阻断；Linux默认入口首次下载Python 3.11运行时长期无进展后已中止，未宣称全量CI通过。

### 开发区500期结果

- fc3d：A组`32/500=6.4%`，B组`28/500=5.6%`；B组LogLoss/Brier均变差，放弃`470/500`期，闸门失败。
- pl3：A组`31/500=6.2%`，B组`32/500=6.4%`；B组LogLoss/Brier均变差，放弃`460/500`期，Top50单侧`p=0.0945`，闸门失败。
- 两次运行均排除了已消费的最后500期Frozen，只是复用开发区的500期快速诊断；它们未覆盖全部完整开发块，本身也不具备晋级资格。未创建新影子状态，未替换日常模型。

## 2026-07-21 behavioral_context_v2定义完善

- 用最近一次完整号码间隔、同组选其他排列间隔替换累计频率压力，固定10期半衰。
- 将上期数字重合拆成同位风险和同数字异位风险；形态改为近期超理论占比和连开超理论期望风险。
- 全部行为特征按每期1000候选标准化，L2从10倍改为2倍；C组全部风险权重强制非正。
- 固定A核心、B自由行为、C单调风险，C为唯一主挑战组；报告新增Top50新增/丢失期、精确配对p值和形态总变差。
- 最终500期快速诊断：fc3d A/B/C=`32/33/33`，C相对A为`13`增、`12`失，配对`p=0.5000`；pl3=`31/39/42`，C为`22`增、`11`失，配对`p=0.0401`，但LogLoss/Brier均变差。
- fc3d C组形态总变差`14.95%`，两个彩种仍有`490/460`个放弃期；全部闸门失败，未替换日常模型。该500期已用于开发诊断，不是独立验证。
- 完整开发区13块/6500期结果：fc3d A/B/C=`314/323/323`，C相对A为`163`增、`154`失，配对`p=0.3266`；pl3=`350/323/333`，C为`171`增、`188`失，配对`p=0.8289`。
- 两个彩种C组LogLoss/Brier均未改善，多个500期块低于随机，放弃率超过90%；pl3的`42/500`局部现象未跨时间复现，v2行为模型关闭。
- C组逐期边界归因显示6项风险在两个彩种的13个固定块中均为负贡献；fc3d伤害最大为同数字异位`-0.0464`、组选近期`-0.0278`、完整号码近期`-0.0275`，pl3为同数字异位`-0.0397`、同位重合`-0.0369`、形态近期拥挤`-0.0252`。
- 该证据不支持只删除一项或反转某个权重；当前可接受修正是保持日常行为权重为0，并把逐特征及合计边界贡献写入后续报告和准入闸门。
- v2相关回归`25 passed`、5个核心模块mypy通过；排除既有Windows `fcntl`快照模块后全仓`151 passed`，覆盖率`80.60%`，Black、isort、flake8和compileall通过。

## 2026-07-21 behavioral_context_v3结构优化

- 修复行为标准化在截断后再次缩放导致稀疏列重新超过8倍幅度的问题；v3保持每期中心化，最终绝对值硬限制为8，稀疏列不再强制恢复单位方差。
- 在线梯度新增可选行为分组裁剪：核心维度继续使用1.0范数预算，行为维度独立使用0.25，行为异常梯度不再压缩核心更新；纯核心路径保持原有全局裁剪语义。
- 默认行为配置从六项缩减为完整号码近期、上期同位重合、同数字异位重合三项；组选近期和两个形态项仍可显式研究，但不再默认启用。
- 删除未实际执行的D组外部上下文占位，报告只保留A/B/C。v3仍是复用开发历史的研究挑战器，未证明预测提升、不得替换日常模型。
- 最近500期开发区快速诊断（不读取Frozen）：fc3d A/B/C命中`32/34/33`，C相对A新增13、丢失12、配对`p=0.5000`；pl3为`31/38/38`，C新增20、丢失13、配对`p=0.1481`。两个彩种B/C的LogLoss与Brier都比A差，C行为边界贡献分别为`-0.2175/-0.3499`，且存在大量放弃期，全部准入闸门失败。
- 该500期只是结构修复后的回归诊断，复用了已开发历史；局部Top50增加不能作为独立优势证据，v3继续关闭并等待全历史压力测试。
- 全部开发区13块/6500期压力测试已完成且未读取Frozen。fc3d A/B/C命中`314/333/333`，C相对A新增213、丢失194、配对`p=0.1861`；C的LogLoss/Brier分别比A恶化`0.004010/0.000008327`，13块行为边界贡献全部为负，均值`-0.4095`。pl3为`350/331/331`，C新增209、丢失228、配对`p=0.8306`；LogLoss/Brier恶化`0.002928/0.000006008`，13块边界贡献全部为负，均值`-0.4322`。
- 两个彩种均存在低于5%随机基线的固定块、放弃率超过94%，全部准入闸门失败。按预注册的两阶段淘汰规则，v3行为模型停止，不再为该模型等待未来500期；日常核心模型和其独立前瞻序列不受影响。

## 2026-07-21 behavioral_context_v4极简包

- 按用户指定口径只保留完整号码近期风险和上期0～3个同位置重合比例；同数字异位重合不再进入默认模型。
- 豹子过滤作为硬候选策略：A维持当前Top50最多1个豹子，B/C必须为0个豹子，避免把过滤规则伪装成可学习权重。
- 继续使用最终幅度8、行为独立梯度裁剪0.25、2倍L2和固定A/B/C；v4仍不读取Frozen、不替换日常模型，只允许全部固定块压力测试与淘汰。
- 全部13块/6500期结果：fc3d A/C均为`314`，新增/丢失=`137/137`、配对`p=0.5241`；pl3 A/C=`350/362`，新增/丢失=`167/155`、配对`p=0.2700`。合并仅净增12/13000、配对`p=0.3262`。
- 两个彩种C的LogLoss/Brier均变差，13块行为边界贡献全部为负，放弃率`99.85%/98.00%`。极简包未通过闸门，不创建影子状态、不接入日常预测，也不继续在同一历史拆分过滤与权重寻找局部赢家。

## 2026-07-21 模型总账与预测能力收口

- 新增`digit_model_scoreboard`聚合器、CLI和Makefile入口，统一列出12条固定成本直选Top50证据，并对随机基线p值执行Holm多重比较校正。
- 行为v1和fc3d事后删除`sum_distribution`候选因覆盖不足或非共享而单列；组选投影和独立组选Top10因成本不同单列，不进入直选排名。
- 当前所有可比模型均为`closed`，`selectedModel=null`、`productionMode=uniform_abstain`、`userVisibleCandidates=[]`；行为v1～v4封存，不再作为常规Makefile任务。
- 核心模型只保留参数冻结的被动前瞻，检查点固定为50/100/200期；50期只允许提前淘汰，200期仍无联合优势则终止。

## 2026-07-21 历史多变体执行效率

- 新增独立`digit_online_gradient_variants`流式执行器，A/B/C共享每个目标期的1000候选特征构建，不缓存全部6500期矩阵，也不改变核心影子状态源码指纹。
- 200期四学习器真实历史对照中，旧三次独立运行`26.442s`，共享运行`15.449s`，加速`1.712x`；默认20学习器对照为`63.601s→49.581s`、加速`1.283x`，A/B/C报告逐字段完全相同。
- 行为CLI每500期输出processed/total、完成块、当前期号、elapsed和ETA；按默认20学习器与旧6500期`5278s`推算约降至`4115s`，节省约19.4分钟。

## 2026-07-22 Search口径与影子状态修复

### 实现与自测

- Search目标缓存升级为schema v2，保存上期原号并自动失效旧缓存；正式训练接入16个确定性特征配置采样。
- 静态learned-ranker在`λ=0`时清空直选、组选、位置池和研究分区，不再把均匀概率转成文本顺序排名。
- fc3d/pl3影子状态仅在`observedPeriods=0`条件下完成兼容迁移；候选权重、滚动损失和前瞻计数保持不变，状态与谱系指向最终源码指纹。
- 定向回归`80 passed`；Black、isort、flake8、mypy（3个核心源码文件）和compileall通过。默认fc3d/pl3 `digit_predict_today --no-fetch --json`均成功，fc3d均匀放弃且研究Top50为空，pl3保留50个研究候选。

### 已知环境提示

- Windows子进程测试仍有既有GBK/UTF-8读取warning，但不影响退出码；本次定向测试无失败。

## 2026-07-22 probability_v5隔离开发挑战器

### 需求摘要

- 按修订后的概率算法方案实现开发模式，不接正式推荐或Frozen，不做v4状态兼容。

### 实现

- 新增Uniform、EWMA位置边际、EWMA位置对、旧梯度结构对照四个预注册专家，并用自适应指数权重逐期混合。
- Calibration只选择temperature，不再叠加动态`lambda`；当前实现不包含既有证据失败的FTRL。
- LogLoss/Brier基于原始1000类概率；逐期同时保存raw Top50和复用日常策略后的Top50，Search和开发Evaluation均报告策略后严格门槛。
- CLI强制至少排除500期Frozen，只写开发报告；验证、随机模拟、研究排名和正式推荐固定关闭，不读取或迁移v4状态。
- 重写`概率算法优化设计方案.md`，明确500/250/500开发切分、未来独立500期Validation和5000次全流程随机模拟前置条件。

### 自测

- v5定向回归覆盖概率归一化、prior-only时序、Calibration隔离、双Top50口径、Frozen隔离和永久关闭准入；v5单文件`6 passed`，连同数据、日常策略和在线梯度相关回归共`29 passed`。
- Black、isort、flake8、定向mypy和py_compile通过；官方fc3d CSV的50期smoke明确排除最后500期，未在仓库内生成报告或状态。
- 未运行正式开发Search、真实一次性Validation、5000次随机模拟或Frozen Test。

### 开发闭环继续完善

- 新增只写一次开发协议，锁定开发数据SHA、截止期、源码、配置、Frozen边界、500/250/500切分和随机模拟计划；完整开发CLI缺少或不匹配协议时直接拒绝。
- 开发报告改为不可覆盖；同内容重复写入幂等，不同内容抛出错误。
- 新增确定性全流程均匀随机模拟，逐次重放四专家、指数聚合、temperature选择和联合统计闸门；正式模式必须绑定协议与开发报告且恰好5000次。
- 2次/50期串行与双进程模拟逐试验完全一致；官方fc3d CSV临时smoke排除最后500期，`nullSimulationPassed=false`且未写入仓库报告或状态。
- 随机模拟新增逐试验只写一次检查点；相同身份重启不重复计算，参考报告身份变化时拒绝复用。
- v5及数据/日常策略/在线梯度相关回归当前`36 passed`；正式5000次、独立Validation和Frozen仍未运行。

## 2026-07-22 probability_v5 review clean-break 修复

### 需求摘要

- 修复Search/Evaluation联合闸门、协议身份、报告与检查点自校验、并发恢复、校准/权重审计、Frozen p-value和behavioral_context类型问题。
- 仅执行开发区与smoke验证；明确不运行正式5000次、不读取Frozen、不提交或推送。

### TDD与实现

- 先新增`tests/test_digit_probability_v5_review.py`并运行，首次真实RED为`10 failed, 1 passed`；失败覆盖旧SHA接口、缺少联合闸门/双指纹/自哈希、非原子发布、`executor.map`和Frozen p-value缺失。
- 核心改为只读协议路径绑定；公共已登记入口内部加载协议并用锁定开发DataFrame完整重算，通用smoke入口永远不声明协议已登记。
- 新增raw/calibrated分布指纹、Calibration逐期审计、最后预测/最终更新后权重和专家权重分布摘要。
- 不可变写入统一使用临时文件、文件`fsync`、原子硬链接、目录`fsync`；null试验和检查点集合分别写入`trialSha256`与`trialSetSha256`。
- 并行null使用`as_completed`即时检查点，恢复测试证明只补算缺失编号；正式次数固定为5000。
- 恢复旧模型总账到`HEAD`语义；稀疏Frozen没有非均匀排名时保留空p-value，并由闸门原因明确解释，不读取或重跑Frozen。

### 最终验证

- 审查回归：`11 passed`；扩展受影响回归：`36 passed`。
- 完整`make ci`：`230 passed`，总覆盖率`86.85%`；Black、isort、flake8、全量mypy 38模块和compileall全部通过。
- 官方fc3d真实smoke：开发使用Frozen边界前50期，`frozenRead=false`、`developmentSignalsPassed=false`；2次null smoke完成，`nullSimulationPassed=false`，报告与试验集合哈希均为64位。
- 官方pl3真实smoke：开发使用Frozen边界前50期，`frozenRead=false`、`developmentSignalsPassed=false`；2次null smoke完成，`nullSimulationPassed=false`，报告与试验集合哈希均为64位。
- smoke产物仅写入`/tmp/probability-v5-smoke-20260722/`；未运行正式5000次、独立Validation或Frozen评估，未提交或推送。

### 最终独立复核补充

- 复现了内存`Verified*._create/token`可伪造来源的残余缺陷；已删除该对象链。正式null只接受只读协议/报告路径，并在入口内部使用锁定数据、源码和配置完整重算、逐字段核对。
- null汇总补齐四个专家最终权重的均值、标准差、25%/50%/75%分位数、最小值和最大值。
- 最终完整`make ci`：`231 passed`，Black、isort、flake8、mypy、coverage和compileall全部通过。
- 最终双彩种真实CLI smoke及独立回读：fc3d/pl3开发报告哈希、null报告哈希、trial集合哈希全部有效；`frozenRead=false`，晋级和推荐继续关闭。产物位于`/tmp/probability-v5-final.Bsq8Rv/`。
- 单次完整1400期链路耗时`26.60s`，串行5000次估算`36.95h`；4进程8次完整链路耗时`199.95s`，观测吞吐估算5000次仍约`34.71h`，未获得有效并行加速。因此没有启动正式5000次，也未伪造其结果。

## 2026-07-22 probability_v5 审查问题最终修复

### 实现结果

- 删除可由调用者伪造的`VerifiedProbabilityV5Protocol`与`VerifiedProbabilityV5DevelopmentReport`对象链；通用开发/smoke入口固定不声明协议已登记。
- 新增路径型已登记开发入口与正式null入口：内部加载只读协议/报告文件，并使用锁定开发DataFrame、当前源码与配置完整确定性重算。
- null逐试验闸门改为Search与Evaluation联合通过；正式身份精确绑定协议Frozen边界、报告边界和协议`developmentData.periods`。
- raw概率保持`float64`，指纹哈希与Calibration/指标实际消费同一数组；检查点与null报告的`referenceReportSha256`精确等于开发报告声明的`reportSha256`。
- 多进程构造、提交或结果失败均失败关闭，不回退线程池；完成顺序测试证明快任务在慢Future完成前已写入检查点。
- `digit_model_scoreboard.py`与其测试恢复`HEAD`行为；Frozen无非均匀排名时p-value保留`None`并由闸门原因解释。

### TDD与验证

- 新增契约测试覆盖内存伪造拒绝、只读文件、独立`hashlib`指纹、Search=false/Evaluation=true联合闸门、正式4999/5001拒绝、Frozen/历史绑定、报告哈希、完成顺序和最终更新后权重相等。
- 聚焦契约与v5回归：21项通过；最终`make ci`：226项通过，总覆盖率86.33%，Black、isort、flake8、mypy与compileall全部通过。
- 仅用可恢复的`trash`移走并重建四个指定产物；fc3d协议`ba74633a02ea3935ccdcfcd1a7e8a80019c0ea31249753f1a5ddfd15cb24b668`、报告`3121e2dca9be3197e14cbe2b84ebe481511e7f10cff2a80ac86495fa93d276f7`；pl3协议`8066e340252db70b241879c4919bb4d44cce8b6d58162ece3e086207424cd672`、报告`3a26c08971e301ff5254101c7e63d014082328d64f0eb03db0bb7b17daaa1631`。两种彩票均以1400期锁定开发数据回读并完整重算验证，只读权限有效。
- 最终fc3d/pl3双CLI smoke位于`/tmp/probability-v5-final2.9BnO2V/`；独立回读验证报告哈希有效，开发报告均`developmentProtocolRegistered=false`，null均为`smoke_only`，晋级与推荐均关闭。
- 未运行正式5000次null、独立Validation或任何Frozen评估；未提交或推送。

## 2026-07-22 快乐8选5开发挑战器

### 需求摘要

- 完整实现`docs/kl8_pick5_design.md`，新增快乐8规则、数据证据边界、六专家预序概率挑战器、Top5生成、统计闸门、正式协议/null边界和安全CLI。
- 全程只使用`uv`/Makefile，不搜索或激活conda；保留现有`probability_v5`未提交改动，不提交、不推送。

### 实现与自测

- 新增快乐820/80数据校验、时间正序语义哈希和Frozen两遍加载；2050行CLI smoke中的最新500行号码为空，仍成功验证1550行开发数据，证明Frozen号码未解析。
- 固定六专家、Hedge更新前后权重、Calibration固定温度网格、Top20内五组合、80标签LogLoss/Brier、超几何精确尾部、固定五块和确定性块bootstrap均已落地。
- 协议/报告采用临时文件、文件fsync、硬链接、只读权限和目录fsync；只读加载器会用当前源码、配置、数据和Frozen边界完整重算，篡改后重哈希仍拒绝。
- null支持确定性20/80无放回随机历史、Search+Evaluation联合闸门、逐试验哈希、有序集合哈希、`as_completed`即时检查点、恢复身份校验和进程失败关闭。
- 最终完整`make ci`为`237 passed`、覆盖率`86.74%`，Black、isort、flake8、mypy和compileall全部通过。
- `/tmp/kl8-pick5-smoke.g8V7QR/`只读smoke报告哈希：开发`8f6472ea5f9748535d079f7786f7703ea1828211e714e00f7cf689ccd9fb0e47`，null`cd146016458ca0463d9b2f03439602e7968f160603a859093a581724e6bd7487`，有序试验集`e09ac9010e3249acb0d6e80c8e9eacafc5043bd1f70152495daa31580d1c5c70`。

### 风险与后续

- 正式至少5000次null未执行；独立Validation 500期尚不存在；Frozen未开启且号码未读取。
- 因此`evidenceStatus=exploratory_reused_development`，`promotionPassed=false`、`recommendationEnabled=false`、`userVisibleCandidates=[]`保持锁死。

## 2026-07-22 快乐8独立审查阻断修复

### 实现结果

- 将退化的pair整行均值替换为事前EWMA80 Top20上下文上的平滑共现lift：按边际外积归一、裁剪、排除self取均值，再以小指数因子修正EWMA80并归一到总和20；直接构造状态证明其不同于EWMA80且响应关联身份。
- 新增`FORMAL_MIN_ITERATIONS=5000`，正式manifest、内部模拟、公共入口和CLI统一要求`max(5000, required_null_iterations)`；配置降为1时，4999仍在历史加载和模拟前拒绝。
- null逐试验、检查点、观测分布与经验p值补齐`evaluationHitAtLeast4Rate`和`evaluationHitExactly5Rate`；正式`nullSimulationPassed`要求联合FPR与六项Evaluation观测p值全部不高于`alpha`，`promotionPassed`继续固定为false。
- 官方抓取最终响应强制`https`白名单，增加参数范围与分页无进展保护；JSONL使用`fcntl.flock`独占锁，损坏尾部拒绝追加，并在`flush+fsync`后解锁。
- 新增集中惩罚、Calibration/Evaluation因果边界、HTTPS降级、无进展、损坏JSONL尾部和高奖级统计检查点回归；保留今日预测的Frozen首期审计标签与空正式候选行为。

### 自测与边界

- `make fmt`只格式化3个KL8相关源码/测试文件，没有格式化无关v5文件。
- KL8聚焦测试：`30 passed`。
- 完整`make ci`：`252 passed`，覆盖率`86.92%`；Black、isort、flake8、mypy和compileall全部通过。
- macOS受限沙箱禁止查询`SC_SEM_NSEMS_MAX`；仅在`/tmp`使用运行时shim把该权限拒绝映射为安全下限，既有v5多进程测试随后通过，仓库源码/测试未为此改动。
- 未执行正式5000次null、未读取Frozen、未重生成临时协议/报告、未提交或推送；`promotionPassed=false`与`recommendationEnabled=false`保持不变。

## 2026-07-22 快乐8最终独立复核阻断修复

### 实现结果

- 正式路径现在只接受逐字段等于`Kl8Pick5Config()`的唯一规范配置；配置构造全面校验有限值、范围、正数/非负约束，并固定Top池20、输出5票、五块、Frozen 500与null至少5000。smoke缩短配置只能进入未登记开发/null smoke。
- 命中证据改为完整五票匹配成本：逐期记录`combinationHits`、`portfolioTotalHits`、`portfolioBestHits`；Search/Evaluation使用每票均值、组合总均值6.25、五块每票稳定性和按号码重数动态规划、跨期卷积的精确右尾，首票仅作审计。
- null逐试验、观测统计和经验p值同步改为五票口径；报告只输出`newCompletionOrder`。检查点严格校验精确字段、联合门控、专家权重、有限Delta、每票/组合均值和比例范围。
- 自哈希和不可变JSON写入统一`allow_nan=False`；官方抓取要求恰好满足请求或接口宣告期数，规范CSV以`0444`临时inode、文件`fsync`、硬链接和目录`fsync`发布，相同内容但可写目标也拒绝。
- 源码指纹不再缓存，每次正式校验都会重新读取六个绑定文件。

### TDD与边界

- 新增/更新配置拒绝、正式入口前置失败、tiny暴力PMF对照、五票记录/分布/null重放、NaN/Inf/越界篡改重哈希、抓取缺行和CSV只读耐久性测试。
- KL8聚焦测试`60 passed`；主环境完整`make ci`为`286 passed`、覆盖率`86.38%`，Black、isort、flake8、mypy与compileall全部直接通过。
- 最新源码下已完成synthetic只读协议/开发报告登记及从路径完整重算：协议`ab1ab4719b59d9ed80c9189661becd6eb31f315646a558c81d70a9a3b125cdcf`、报告`b1be639a4393b8aeb6540fb2d428ea314e3e211e9685db4e2fe7263f93307e04`；五票逐期/组合包契约有效，500行损坏Frozen号码未解析。
- 最新完整单次null trial（含完整门控输入持久化）耗时`35.37秒`，串行5000次估算约`49.13小时`。未运行正式5000次null、未读取Frozen、未提交或推送；`promotionPassed=false`、`recommendationEnabled=false`和正式空候选保持不变。

## 2026-07-22 快乐8 null检查点门控派生 Critical 修复

### 实现结果

- `Kl8NullTrial`新增`searchGateInputs`与`evaluationGateInputs`，逐试验持久化生产`_segment_gate`读取的完整规范输入：六项顶层统计、两项block bootstrap的`pValueMeanNonPositive`，以及恰好五块`deltaLogLoss/deltaBrier/meanHitsPerTicket`。
- 新增统一规范提取/校验器：加载检查点时要求顶层和所有嵌套键集合精确相等；所有门控统计量必须为非布尔JSON原生`int/float`、有限且位于合理范围；`index/seed`必须为精确JSON整数，专家权重同样拒绝字符串和布尔值。
- `_run_null_trial`保存两段规范门控输入与布尔结果，并用生产`_segment_gate`核对；`_load_trial`分别重导Search/Evaluation结果，要求存储布尔与派生值一致，再以派生结果验证`jointPassed`，未复制门控公式。
- 检查点摘要Delta与命中均值必须逐项等于对应规范门控输入，避免同一trial内部出现两套不一致统计。

### TDD与验证

- 新增真实trial检查点回归：原始Search/Evaluation/Joint均为false时，将三个布尔全部翻为true并重算`trialSha256`，恢复因门控派生不一致而拒绝。
- 新增门控输入篡改回归：将精确组合总命中p值改为内部失败值、仍声明三个门控为true并重哈希，恢复拒绝；另覆盖顶层/两层bootstrap/五块额外键及嵌套字符串数值。
- JSON精确类型回归覆盖布尔`index`、字符串`seed`、字符串统计量和字符串专家权重。
- `make fmt`通过；KL8聚焦测试（含strict contract）`60 passed`；最终`make ci`为`286 passed`、覆盖率`86.38%`，Black、isort、flake8、mypy和compileall全部通过。
- 首次直接完整CI仅因macOS受限沙箱拒绝查询`SC_SEM_NSEMS_MAX`导致既有v5多进程测试失败；按既有策略仅在`/tmp`注入运行时安全下限shim后完整通过，未修改任何v5文件。
- 未运行正式5000次null、未读取Frozen、未重生成临时协议、未提交或推送；既有无关v5差异保持不变。

## 2026-07-23 快乐8 null 性能优化

### 实现

- 根据完整trial `cProfile`，确认五注组合生成占`29.95/40.73秒`，而bootstrap约`2.55秒`、精确五注PMF约`0.79秒`。
- 为固定Top20组合池预计算`15504×20`只读incidence矩阵，以5列求和替代每轮`15504×5×5`广播比较；按原顺序增量减去每个已选组合的平方重叠惩罚。
- 连续块bootstrap一次生成与旧逐次调用相同顺序的RNG起点矩阵并向量化索引；概率和20二分仅在中点已等于上下界、后续迭代不再改变浮点结果时提前退出。
- 新增慢速选择器逐组对照、incidence重叠精确集合交集、bootstrap逐元素bitwise一致测试；固定seed完整trial SHA-256保持`bcbe0db379edb0a123c87ab12b8aa92ff0c0626b1f526c37f8b8a1a636a0880`不变。
- Makefile新增显式`kl8-pick5-null-formal`，使用独立正式output/checkpoint；当前10核/16GB机器默认`KL8_NULL_WORKERS=8`，不会自动启动。

### 验证与性能

- 三次完整无profile trial为`11.37/11.37/11.31秒`，后续bootstrap/二分优化后单次`9.10秒`；相对原`35.37秒`约`3.89×`。
- 4 worker×8次真实checkpoint为`23.53秒`，5000次折算`4.09小时`；8 worker×16次为`42.59秒`，折算`3.70小时`。
- KL8聚焦测试`63 passed`；最终`make ci`为`289 passed`、覆盖率`86.33%`，Black、isort、flake8、mypy与compileall全部通过。
- 最新synthetic只读协议`c6bff9bd829ee968503b769ea05ce7e046d63e07deafdfc21a18dd3e55a3ccbb`、报告`8bf82853ac21b51c7cbed9a363694af14263f6f3a4d29108162387cdae6a8d49`已从路径完整重算；未运行正式5000次、未读取Frozen、未提交或推送。

## 2026-07-23 快乐8全量历史与v2特征发现

- 福彩官网默认查询仅暴露最近1000期；抓取器新增2020年至今的年度日期窗口分页、跨窗语义去重/冲突、年度越界、零记录和无进展失败关闭。全量获得2014期：`2020001/2020-10-28`至`2026193/2026-07-22`。
- canonical CSV共2014期、逐年期号零缺口、无重复、每期20个唯一1—80号码，模式`0444`；CSV SHA-256为`cb53ec0e9855bf72a9f7e69aeb23ccce24e199d0e411e04f221dadd32d63efb9`，raw JSONL为`7c844602f0298568ae485fe4fe6420e77f095a1d794d3c11531d9fd693d4a318`。
- 总历史少于v1固定2050期需求：最新500期Frozen为`2025045..2026193`且号码未解析，开发区1514期为`2020001..2025044`，不能伪造缺少的36期。
- 新增隔离v2：300期初始训练、714期Search、500期开发Evaluation；LightGBM每50期按此前数据扩展重训，比较频率、遗漏、lag、趋势、EWMA、上期上下文及pair lift。
- 五个嵌套组总体LogLoss/Brier均不优于均匀且五块不稳定；十三个独立消融也无人通过。最接近零的`frequency320`仍为负proper-score增益且Top20少`0.0742`；`frequency80`虽Top20多`0.0938`，但两项proper score为负且仅2/5块为正，按规则淘汰。
- 最终`selectedFeatureSet=uniform`、`frozenRead=false`、`promotionPassed=false`、`recommendationEnabled=false`。Search无合法胜者，因此不做温度校准，Evaluation保持均匀基线。
- 结果产物：`kl8_feature_discovery_v2_official_20260723.json` SHA-256 `d6a0cc6e7c7b5d00ff423300837d05e1cda67bf2d9f6d87661584cb7281a23e8`；独立消融报告 SHA-256 `85df910a129722c1dcb72368a1c1f19166b71f6b3ae4eb5d1dd1f9253a5d4e08`。
- v2专项`7 passed`；最终完整CI `297 passed`、覆盖率`86.28%`，Black/isort/flake8/mypy/compileall全部通过。未运行正式5000次null，未提交或推送。

## 2026-07-23 快乐8选4安全入口

- 新增`kl8_pick4_prediction.py`与CLI：每票4个唯一1—80号码，精确超几何随机基线平均命中`1.0`；命中0/1/2/3/4概率分别为`30.8321%/43.2732%/21.2635%/4.3248%/0.3063%`。
- 默认入口固定`formalRecommendation=null`、`userVisibleCandidates=[]`；只有显式`--test`才根据日期与开发区SHA-256生成等概率、可复现测试组合，标记`uniform_random_test_only`。
- 复用两遍加载器并通过非法Frozen号码测试；不读取Frozen号码、不打开Validation、不把Pick5门槛套给Pick4。
- Makefile新增`kl8-pick4-predict-today`和`kl8-pick4-test-today`；专项测试`6 passed`，与既有安全入口联合测试`7 passed`。真实2014期CSV默认/测试/重复CLI返回码均为0，同日测试组合逐字段一致；完整CI `303 passed`、覆盖率`86.25%`，Black/isort/flake8/mypy/compileall全部通过。

## 2026-07-23 快乐8Pick4固定排名优化（审查修正版）

- 按一次性受控挑战实现`kl8_pick4_rank_challenger.py`：只用`frequency80/320`、两者差、`omissionLog`、`ewma80`、`inPrevious`；300期初始训练后对1214期严格前向评估，每50期扩展重训。
- 独立审查发现v1仅设置`eval_at=4`并不等于训练截断Top4。v1只读报告保留为被替代审计记录；v2显式锁定`lambdarank_truncation_level=4`、`label_gain=[0,1]`、每期80行/20正例query契约。
- 排名分数风险审计固定为`q=normalize_sum20(sigmoid(0.1*zscore(score)))`，再以`lambda=0.1`收缩至0.25；明确标记非校准概率。LogLoss/Brier使用2000次、块长12的循环块bootstrap，Top4/Top20使用精确超几何卷积，四项执行Holm校正联合gate。
- 每期将Top20按排名层轮转为5张互不重叠Pick4票；主Top4是独立排名审计，Top20组合池不能救活Top4失败。
- v2真实结果：主Top4`1244/1214=1.0247`、原始`p=0.1594`；五注`6138`总命中、每票`1.0112`、每期组合`5.0560`、原始`p=0.1255`。
- Delta LogLoss/Brier仅`0.00000209/0.00000080`，块bootstrap `p=0.4313/0.4198`；Holm调整后Top4/组合池均`0.5022`、proper scores均`0.8396`。第2块主Top4低于随机，第4/5块proper scores恶化，联合闸门失败。
- v2只读报告`kl8_pick4_rank_challenger_v2_official_20260723.json` SHA-256为`a9d000ec8b85fceee5cd568accac699a98e154ce59d1a7a0de7f9ca2059e82f5`；源码指纹`974035c3ae7d86bc9a07263f8bbdeb59fe91eb79cf2a1d096971a9ac09ba9fa4`匹配，`frozenRead=false`、证据状态`exploratory_post_failure_reused_development`，正式候选保持空。
- 修正版最终完整CI `310 passed`、覆盖率`86.06%`，Black/isort/flake8/mypy/compileall全部通过；Make入口dry-run和`git diff --check`通过。
