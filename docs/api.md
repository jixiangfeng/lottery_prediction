# 公共 API 一览（2025.10）

## 核心模块

| 模块 | 函数 | 参数 | 返回值 | 说明 |
|------|------|------|--------|------|
| `src.common` | `get_data_run(name, sequence_mode=False, start_issue=None, end_issue=None)` | 彩票代号；是否抓取顺序数据；期号区间 | `None` | 下载快乐 8 历史数据并写入 `data/kl8/data.csv`。 |
| `src.common` | `get_current_number(name)` | 彩票代号 | `str` | 读取快乐 8 最新期号。 |
| `src.common` | `load_history(name)` | 彩票代号 | `pandas.DataFrame` | 从本地 CSV 加载历史开奖数据。 |
| `src.data_fetcher` | `download_history(code, start=None, end=None, use_sequence_order=False, client=None)` | 彩票代号、期号区间、顺序模式、HTTP 客户端 | `DownloadResult` | 带重试和白名单校验的抓取实现。 |
| `src.data_fetcher` | `get_current_issue(code, client=None)` | 彩票代号、可选 HTTP 客户端 | `str` | 获取官网最新期号。 |
| `src.data_fetcher` | `load_history(code)` | 彩票代号 | `pandas.DataFrame` | 与 `common.load_history` 等价，直接暴露底层能力。 |
| `src.analysis.feature_enhancer` | `compute_enhanced_scores(draws, limit, recent_window=40, reference_window=160, decay=0.97, weights=(0.45,0.25,0.30), dirichlet_weight=0.22, pca_components=1, use_pca=True)` | 历史开奖多维特征（动量、共现谱、Dirichlet 后验、图嵌入）融合 | `(List[Tuple[int, float]], FeatureDebugInfo)` | 返回排序列表与调试信息，包含 `graph_embedding_scores`、Dirichlet 均值/方差等字段 |
| `src.analysis.rule_miner` | `build_rule_filter(draws, lottery_code, limit, mode, min_support=None, min_confidence=None, max_itemset_size=None, penalty_weight=None)` | ��ʷ�����顢��Ʊ������ͳ�Ʒ�Χ��ģʽ/����ֵ | `RuleBasedFilter | None` | ����FP-Growth ��Ƶ�� �� -> ��������ƵȨ�أ�֧���̲�ģʽ/����ģʽ |
| `src.analysis.feature_enhancer` | `compute_recency_and_momentum_scores(draws, limit, recent_window=40, reference_window=160)` | 历史开奖二维数组、窗口设置 | `(np.ndarray, np.ndarray)` | 分别返回近期频率得分与动量得分。 |
| `src.analysis.feature_enhancer` | `compute_co_occurrence_scores(draws, limit, decay=0.97)` | 历史开奖二维数组、统计范围、衰减因子 | `np.ndarray` | 通过共现矩阵的主特征向量衡量号码中心性。 |

#### PCA主成分特征用法
```python
from src.analysis.feature_enhancer import compute_enhanced_scores
ranked, debug = compute_enhanced_scores(draws, limit=100, use_pca=True, pca_components=1)
```
- `use_pca`: 是否启用PCA特征（默认True）
- `pca_components`: 主成分数量，通常取1即可
- 返回结果已自动融合PCA特征，无需手动拼接

## Plus 版本并行接口

### `kl8_analysis_plus.py`
- `download_data_if_needed(download: int, cal_nums: int) -> None`：主线程数据下载，避免重复 IO。
- `sub_process(task_args) -> list`：线程池工作函数，返回单组号码组合。

### `kl8_cash_plus.py`
- `check_lottery(file_path, file_name, data_dict, download_flag) -> dict`：单文件收益分析。
- `process_files_parallel(file_list, max_workers=4) -> list`：线程池批量收益分析。

## 线程安全约定
- 共享资源必须使用 `threading.Lock`：`with results_lock: shared.append(item)`。
- 工作线程需捕获异常，单次失败不会终止主流程。

> CLI 脚本仍以命令行方式存在；若需在代码中复用，请通过 `subprocess.run([...])` 调用并写明全部参数。
# 数字彩第二轮 API

- `analyze_digit_history(...)`：保留旧字段，输出位置、位置对、形态、和值、跨度、奇偶、大小、质合、连号、镜像、和值尾、上期距离和同位重号的多窗口平滑概率，以及 `omissionWindows` 多窗口截断遗漏。
- `score_digit_prefix(stats, numbers, config)`：共享三位前缀启发式复合评分，排列三与排列五前三位均调用该函数；它不是规范联合概率。
- `rank_digit_numbers(...)`：不构造全量候选对象，返回目标号码在过滤空间中的复合模型分排名与评分。
- 候选 JSON 使用 `modelWeight` 与过滤空间归一化的 `compositeModelWeight`；旧 `jointProbability` / `probabilityMass` 为 deprecated 兼容字段，不表示实际开奖概率。
- `simulate_digit_candidates(..., pair_strength=0.75, structure_strength=0.35)`：按多窗口边际概率、位置对条件概率和形态接受率联合模拟，通过同一过滤/约束器后返回可复现的候选频率排序。
- `train_digit_ranker(...)` / `score_digit_ranker(...)`：用 `StandardScaler + LogisticRegression` 构建时间严格的候选二分类排序器；返回分数只作排序。
- `build_advanced_model_scores(...)`：统一构建蒙特卡洛/ML 外部票及 `DigitAdvancedModelDiagnostics` 运行证据。
- `run_digit_walk_forward_backtest(..., baseline_runs=20, nested_tuning=False, inner_validation_periods=10, advanced_models=False, compare_windows=False)`：输出多随机分布、严格嵌套调参、可选高级投票和 30/50/100/300/全历史独立窗口比较。
- `DigitCandidateConfig(ranking_mode="ensemble")`：使用 14 个统计子模型加蒙特卡洛、ML 共 16 个固定槽位的过滤空间排名分位做集成排序；inactive 外部槽位以中性 `0.5` 占位并保留固定分母，只改变绝对分值尺度，不提供相对排序信号。候选输出 `ensembleScore`、`modelRankPercentiles` 与 `topDecileVotes`，active 表示存在非中性结果或模型候选信号。
- `DigitCandidateConfig(constraint_mode="soft", constraint_probability_floor=0.02, constraint_penalty_weight=0.05)`：提供奇偶/大小/质合结构的 `off|soft|hard` 约束；候选输出 `constraintPenalty`。
- `generate_digit_candidates(...)`：兼容旧 `candidates` 直选字段；在 `ensemble` 模式下额外返回实际有分数的子模型 `modelCandidates` TopK。
- `generate_digit_betting_candidates(...)`：返回直选/组选；组三和组六分别在各自无序空间重排模型分位，输出 `rankingModel=shape_specific_ensemble`，排列五组选为空。
- `save_digit_pick_snapshot(...)` / `process_digit_pick_evaluations(...)`：保存开奖前推荐，并在后续数据中找到源期之后第一期开奖进行自动复盘和累计汇总。
- `derive_live_ensemble_weights(evaluations, base_weights, min_samples=5)`：只基于开奖前逐模型留痕做加一平滑与±20% 封顶调权。

### 数字彩理论概率与增量快照 API

- `get_digit_theoretical_probabilities(rule)`：返回按规则签名缓存的精确数学枚举表，包含 `shape`、`sum`、`span`、`parity`、`bigSmall`；每次返回防御复制，`baselineType=exact_mathematical_enumeration` 且 `isPrediction=false`。
- `analyze_digit_history_with_snapshot(df, rule, snapshot_path, *, frequency_windows, bayesian_prior_strength, all_history_window=False, rebuild=False)`：返回 `(DigitStatisticsResult, DigitStatisticsUpdateMetadata)`。首次为 `full_rebuild`，无新增为 `cache_hit`，纯追加为 `incremental`。若等待锁期间快照已被其他进程推进，且本次输入是当前快照的严格短前缀，则只在内存遍历旧输入并返回 `stale_view`，不写回快照；真实历史修正或删减仍沿用原全量重建逻辑。
- `DigitStatisticsUpdateMetadata.to_dict()`：输出 `mode`、`addedIssues`、`processedRows`、`rebuildReason`、`requestedRebuildReason`、`snapshotPath`、`persisted`、`snapshotWritten`。`stale_view` 固定使用 `rebuildReason=stale_view_not_persisted`，并在 `requestedRebuildReason` 保留窗口、先验、schema/engine 或 `explicit_rebuild` 等原判断；`processedRows` 是实际遍历数，禁止伪装为 cache hit。`full_rebuild`/`incremental` 为 `persisted=true,snapshotWritten=true`，`cache_hit` 为 `true,false`，`stale_view` 为 `false,false`。
- 快照 JSON 包含 schema/engine 版本、规则签名、窗口/先验配置、已处理前缀摘要、全历史聚合、10/30/50/100/300 等有界窗口状态、近期队列、遗漏状态与最新期号/号码；写入使用同目录临时文件加 `os.replace` 原子替换。
- 自动重建原因包括 `corrupt_json`、`schema_version_mismatch`、`engine_version_mismatch`、`rule_signature_mismatch`、`window_config_mismatch`、`prior_strength_mismatch`、`history_truncated`、`historical_prefix_changed`、`non_append_issue`；显式 `rebuild=True` 为 `explicit_rebuild`。

`generate_digit_report_from_csv(...)` 新增参数：

- `stats_snapshot_path=None`：默认 `output_dir/state/{code}_statistics_snapshot.json`。
- `rebuild_stats=False`：强制重建统计快照。
- `incremental_stats=True`：关闭时直接全量分析，仅用于诊断。
- `enable_hindsight_backtest=False`：默认不把当前候选回放全部历史；启用后恢复旧诊断输出。

CLI 对应 `--stats-snapshot-path`、`--rebuild-stats`、`--no-incremental-stats`、`--hindsight-backtest`。

日报 JSON `schemaVersion=2` 保持不变，旧字段只增不删，并新增 `theoreticalProbabilities`、`statisticsUpdate`、`hindsightBacktest`；`backtest` 字段继续保留，默认关闭 hindsight 时返回零检查摘要以维持结构兼容。前推 JSON `schemaVersion=4`，继续输出 `modelPerformance`、`strategies`、`strategyScoreBucketDistributions`、`advancedModels` 和 `windowComparison`，且不读取日报统计快照。

所有评分、排名和回测接口仅用于历史研究，不能保证中奖。
