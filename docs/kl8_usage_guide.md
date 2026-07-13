# KL8 快乐 8 使用指南（2025.10 更新）

## 最新亮点
- `--advanced_mode` 支持三档算法（0=传统，1=遗传+贝叶斯，2=全栈增强）。
- `feature_enhancer` 集成 Dirichlet 平滑、PCA 主成分、图嵌入等特征通道，可通过 `--feature_mode` 切换。
- Plus 版本脚本引入线程池，并行生成组合且只下载一次数据。
- 新增 Copula 多样性采样与互信息惩罚，可在 CLI 中启用。

## 环境准备
```bash
conda activate python311
make setup
make download-data
make train-graph   # 建议执行，生成图嵌入缓存
```

## 一键启用全算法
```bash
python src/analysis/kl8_analysis.py \
  --cal_nums 20 \
  --total_create 240 \
  --limit_line 240 \
  --advanced_mode 2 \
  --feature_mode hybrid \
  --rule_filter soft \
  --rule_support 0.08 \
  --rule_confidence 0.7 \
  --copula_mode force \
  --copula_samples 64 \
  --copula_shrinkage 0.1
```
- 包含遗传算法、贝叶斯排序、特征增强、FP-Growth 规则、Copula 采样与互信息惩罚。
- 可通过 `--copula_min_draws`、`--copula_multiplier` 等参数继续调优。

## 快速上手脚本

### 1. 原始模式（Mode 0）
```bash
python src/analysis/kl8_analysis.py --cal_nums 5 --total_create 50 --limit_line 80 --advanced_mode 0
```

### 2. 中级模式（Mode 1）
```bash
python src/analysis/kl8_analysis.py --cal_nums 10 --total_create 100 --limit_line 200 --advanced_mode 1
```

### 3. Mode 2 + 特征增强
```bash
python src/analysis/kl8_analysis.py \
  --cal_nums 12 \
  --total_create 180 \
  --limit_line 220 \
  --advanced_mode 2 \
  --feature_mode hybrid
```

### 4. 多线程批量生成（Plus 版本）
```bash
python src/analysis/kl8_analysis_plus.py \
  --cal_nums 18 \
  --total_create 600 \
  --limit_line 250 \
  --max_workers 6 \
  --advanced_mode 2 \
  --feature_mode momentum \
  --copula_mode auto
```

### 5. 批量收益分析
```bash
python src/analysis/kl8_cash_plus.py --path outputs/daily --max_workers 4
```

### 6. 参数遍历调度
```bash
python src/analysis/kl8_running.py \
  --cal_nums_list "5,7,10" \
  --total_create_list "100,300" \
  --nums_range "2023200,2023210" \
  --running_mode 0
```

## 核心参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--cal_nums` | 10 | 每组号码数量（推荐 5~20） |
| `--total_create` | 50 | 生成组合数量，可结合 `--multiple` 使用 |
| `--limit_line` | 50 | 分析使用的历史期数 |
| `--advanced_mode` | 0 | 0=基础，1=遗传+贝叶斯，2=全栈增强 |
| `--feature_mode` | `hybrid` | `hybrid` / `momentum` / `cooccurrence` |
| `--rule_filter` | `none` | `none` / `soft` / `hard` |
| `--copula_mode` | `auto` | `auto` / `off` / `force` |
| `--copula_samples` | 0 | ≤0 使用配置默认，>0 手工指定采样数量 |
| `--copula_shrinkage` | -1 | <0 使用默认，≥0 覆盖收缩强度 |
| `--max_workers` | 4 | Plus 版本线程数，建议近似 CPU 核心数 |

## 特征增强模式对照

| 模式 | 特点 | 场景 |
|------|------|------|
| `hybrid` | 动量 + 共现谱 + Dirichlet + PCA + 图嵌入 | 综合推荐 |
| `momentum` | 强化近期趋势 | 有明显热点时 |
| `cooccurrence` | 注重号码关联 | 追求组合协同性 |

## 新增特性说明
- **Dirichlet 平滑**：通过滑动窗口对冷号进行先验平滑，避免概率为零。
- **PCA 主成分**：自动提取全局趋势，丰富组合多样性。
- **图嵌入特征**：使用 Node2Vec 表征号码结构关系，需先执行 `make train-graph`。
- **Copula + 互信息**：建模 80 维相关性并适度惩罚高关联组合，提高多样性。

## 故障排查
- **下载失败**：确认网络可访问 `https://datachart.500.com` 和 `https://data.917500.cn`，必要时使用代理。
- **特征增强报错**：确保安装 `numpy`、`scikit-learn`，未安装时脚本会自动回退并在日志中提示。
- **Copula 跳过**：当 `limit_line` 小于 `analysis.copula.min_draws`（默认 180）时会自动禁用，提升历史样本量即可。
- **线程模式卡住**：降低 `--max_workers`，并检查磁盘是否存在锁文件或大量历史输出未清理。

## 建议
- 常规生产场景推荐 `--advanced_mode 2 --feature_mode hybrid --copula_mode auto --rule_filter soft`。
- 多线程运行时建议为不同任务设置 `--path`，避免输出文件覆盖。
- 如果仅需回测，可设定 `--download 0` 并直接使用本地数据，降低网络请求频率。
