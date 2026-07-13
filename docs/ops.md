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
