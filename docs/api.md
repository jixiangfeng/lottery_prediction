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
