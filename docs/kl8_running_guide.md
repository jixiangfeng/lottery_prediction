# KL8 Running 批量运行指南

## 概述
`src/analysis/kl8_running.py` 可批量调度多个分析与现金流任务。脚本已优化为线程池模式：主线程负责一次性下载数据，工作线程共享缓存，避免重复 IO。

在批量任务前，建议先使用 README 或《使用指南》中的“一键启用全算法”命令验证单次运行效果，再将参数迁移到 `kl8_running.py`。

## 运行模式
| `--running_mode` | 说明 |
|------------------|------|
| `0`              | 同时运行分析与现金流模式 |
| `1`              | 仅运行分析模式 |
| `2`              | 仅运行现金流模式 |

## 快速示例
```bash
# 小规模安全启动（建议首次执行）
python src/analysis/kl8_running.py \
  --cal_nums_list "5" \
  --total_create_list "10" \
  --nums_range "2024268,2024268" \
  --running_mode 1 \
  --max_workers 2 \
  --download 1
```

```bash
# 应用全算法配置（在单次命令验证通过后迁移）
python src/analysis/kl8_running.py \
  --cal_nums_list "20" \
  --total_create_list "240" \
  --nums_range "2024300,2024305" \
  --running_mode 1 \
  --max_workers 6 \
  --copula_mode force \
  --copula_samples 64 \
  --copula_shrinkage 0.1 \
  --feature_mode hybrid \
  --rule_filter soft \
  --rule_support 0.08 \
  --rule_confidence 0.7
```

## 参数说明
| 参数 | 默认 | 说明 |
|------|------|------|
| `--cal_nums_list` | `"4,5,7,10"` | 多个选号数量，逗号分隔 |
| `--total_create_list` | `"50,100,1000"` | 组合生成数量列表 |
| `--nums_range` | `"2023140,2023241"` | 起止期号（包含） |
| `--max_workers` | 4 | 线程池大小，建议接近 CPU 核心数 |
| `--download` | 1 | 是否下载最新数据（0=使用本地缓存） |
| 其他分析参数 | — | 透传给 `kl8_analysis.py` / `kl8_cash.py` |

## 并发调优
| 设备配置 | 推荐 `max_workers` | 内存建议 |
|----------|-------------------|----------|
| 4 核 / 8 GB | 2–3 | ≥8 GB |
| 8 核 / 16 GB | 4–6 | ≥12 GB |
| 12 核 / 32 GB | 6–10 | ≥16 GB |

建议逐步放大参数：先固定 `max_workers=1` 验证逻辑，再逐步提升线程数、历史窗口与生成数量。

## 监控与告警建议
- **内存/CPU**：保持内存使用率 < 80%，CPU 长期 100% 需减小并发。
- **下载频率**：每日一次即可，如需多次执行可设置 `--download 0`。
- **输出目录**：大量结果文件会增加 I/O 压力，建议按日期或任务名拆分目录，并定期清理。

## 常见问题
1. **数据下载失败**：确认网络可访问官方源域名，或改用离线数据。
2. **线程卡住**：降低 `--max_workers`，检查磁盘是否被大量旧结果占用。
3. **Copula 被跳过**：确保 `limit_line` ≥ `analysis.copula.min_draws`（默认 180）。
4. **结果混淆**：使用 `--path` 为不同任务指定独立输出目录。

## 调试技巧
```bash
# 单线程调试
python src/analysis/kl8_running.py --max_workers 1 --cal_nums_list "5" --total_create_list "5"

# 分阶段验证
python src/analysis/kl8_running.py --running_mode 1   # 仅分析
python src/analysis/kl8_running.py --running_mode 2   # 仅现金流
```

## 版本历史
- **v1.4.0**：适配 Copula / 互信息 / 图嵌入参数透传，新增全算法示例。
- **v1.3.0**：线程池架构与统一下载。
- **v1.1.0**：初始多线程版本。
