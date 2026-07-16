# 运维与监控指南（多线程版本）

## 环境要求
- 建议在名为 `python311` 的 Conda 环境中运行，保持依赖一致。
- 网络需可访问 `https://datachart.500.com` 与 `https://data.917500.cn`。
- `make setup` 会自动创建 `data/kl8`、`results`、`logs` 等目录。
- 建议在批量任务前执行：
  ```bash
  make download-data
  make train-graph
  python src/analysis/kl8_analysis.py ...  # 先单次验证，再批量投入
  ```

## 数字彩闭环运行建议

```bash
make digit-walk-forward \
  DIGIT_LOTTERY=fc3d \
  DIGIT_CSV=data/fc3d/data.csv \
  DIGIT_WF_PERIODS=30 \
  DIGIT_WF_BASELINE_RUNS=20 \
  DIGIT_WF_ADVANCED_MODELS=1 \
  DIGIT_WF_COMPARE_WINDOWS=1 \
  DIGIT_WF_NESTED_TUNING=1 \
  DIGIT_WF_INNER_VALIDATION_PERIODS=10
```

- 日常快速验证可将 `DIGIT_WF_BASELINE_RUNS` 降为 5；正式比较建议至少 20。
- Makefile 默认开启蒙特卡洛/ML 票与独立窗口比较；快速冒烟可设 `DIGIT_WF_ADVANCED_MODELS=0 DIGIT_WF_COMPARE_WINDOWS=0`。
- 高级模型成本可用 `DIGIT_WF_MC_SIMULATIONS`、`DIGIT_WF_ML_TRAINING_PERIODS`、`DIGIT_WF_ML_NEGATIVE_SAMPLES` 调整；报告中必须保留是否启用的状态。
- 排列五嵌套前推成本最高，先用 5～10 个外层目标期冒烟，再扩大窗口。
- 使用 `DIGIT_WF_REPORT_PREFIX=second_round` 写入独立文件，避免覆盖既有评估报告。
- 监控每期 `selectedConfigTrainEndIssue < issue`，若不满足应立即停止使用报告。
- 监控 `strategyScoreBucketDistributions` 和 `windowComparison`；若真实开奖号长期未进入高分位，不得以候选分数代替命中证据。
- 结果不超过随机时必须如实保留；本工具不能保证提高中奖概率。
- `make digit-report` 默认 `DIGIT_RANKING_MODE=ensemble`；切回旧排序时显式设置 `DIGIT_RANKING_MODE=composite`。
- 结构约束默认 `DIGIT_CONSTRAINT_MODE=soft`；切换 `hard` 前必须使用相同阈值完成严格前推，并监控过滤空间是否过小。
- 每期快照的 `modelCandidates` 只记录当期实际有分数的模型；ML 未训练或蒙特卡洛关闭时不得生成伪模型复盘样本。
- 自动调权只使用 `reports/evaluations/<彩种>_live_summary.json` 中有快照证据的逐模型样本；单模型样本不足 5 期保持基础权重。
- 数字彩日报会写入 `reports/picks/digit`，并把已开奖快照复盘到 `reports/evaluations/<彩种>_<期号>.*`；不得在开奖后删除或改写历史快照再重新统计。

## 关键监控指标
| 项目 | 检查方式 | 目标 |
|------|----------|------|
| 下载次数 | 主线程日志 | 每次运行仅下载一次 |
| 线程数 | `threading.active_count()` | ≈ `max_workers` + 1 |
| 内存使用 | `psutil.virtual_memory()` | < 80% |
| I/O 负载 | 监控磁盘写入速率 | 避免长时间 100% |
| Copula 诊断 | 日志 `cond≈...`、`effective_draws=...` | 条件数不过高、样本量 ≥ `min_draws` |

示例命令：
```bash
python -c "import threading; print('active threads:', threading.active_count())"
python -c "import psutil; print('memory usage:', psutil.virtual_memory().percent, '%')"
```

## 故障排查
| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 下载失败 | 网络受限或域名未放行 | 配置代理或启用离线模式（`--download 0`） |
| 线程卡住 | 锁竞争 / I/O 拥堵 | 降低 `--max_workers`，清理结果目录 |
| Copula 跳过 | 样本量不足 | 提高 `--limit_line` 或 `analysis.copula.min_draws` |
| 互信息惩罚过高 | 权重偏大 | 调整 `analysis.graph_embedding.weight` |
| 结果混淆 | 输出目录冲突 | 使用 `--path` 区分任务 |

调试示例：
```bash
python src/analysis/kl8_analysis.py --max_workers 1 --copula_mode force --debug 1
python src/analysis/kl8_running.py --max_workers 1 --cal_nums_list "5" --total_create_list "10"
```

## 常驻任务建议
```bash
# 每日 05:00 下载数据
0 5 * * * cd /path/to/kl8-lottery-analyzer && make download-data

# 每日 06:00 执行全算法分析（示例参数）
0 6 * * * cd /path/to/kl8-lottery-analyzer && \
python src/analysis/kl8_analysis.py \
  --cal_nums 20 --total_create 240 --limit_line 240 \
  --advanced_mode 2 --feature_mode hybrid \
  --rule_filter soft --rule_support 0.08 --rule_confidence 0.7 \
  --copula_mode force --copula_samples 64 --copula_shrinkage 0.1 \
  >> logs/daily_analysis.log 2>&1

# 每周日 00:00 批量收益分析
0 0 * * 0 cd /path/to/kl8-lottery-analyzer && \
python src/analysis/kl8_cash_plus.py --path weekly_results --max_workers 6 \
  >> logs/weekly_cash.log 2>&1
```

## 告警建议
- 下载失败连续 ≥3 次。
- 单次任务执行超过 10 分钟未结束。
- 内存使用率 ≥ 85% 持续 5 分钟。
- Copula 拟合条件数异常升高（>1e6）或样本量低于阈值。
- 结果输出目录增长过快，需要定期清理。
