# 开发假设与约束

1. **仅支持快乐 8**  
   当前代码与配置均围绕快乐 8（`kl8`）玩法设计，其他彩票类型暂不维护。
2. **Python 3.11 环境**  
   默认在名为 `python311` 的 Conda 环境或等效的 Python 3.11 虚拟环境中执行，确保依赖一致。
3. **离线演示数据有限**  
   仓库附带的 `data/kl8/data.csv` 仅包含少量示例，用于 CI 与快速演示；正式分析需自行下载最新数据。
4. **外部网络访问受限**  
   `src/data_fetcher.py` 只允许访问白名单域名，当前主数据源为中国福彩网 `www.cwl.gov.cn`；旧 500/917500 数据源仅作为兼容配置保留。

5. **H5 用户端数据来源**  
   Vue3 H5 首版作为用户端前端工程，不直连 Python 进程；通过 `scripts/sync_h5_data.py` 将最新 `reports/data/kl8_daily_<期号>.json` 同步为 `h5/public/report-data/latest.json`，前端运行时通过 HTTP 动态读取该 JSON。
6. **分析脚本以 CLI 形式运行**  
   `src/analysis` 下脚本保持命令行入口，导入时如需自定义参数需显式传入或使用子进程调用。
7. **Dirichlet 与规则参数配置**  
   先验强度、方差惩罚与频繁项集阈值默认使用 `config.analysis` 中的值，可通过 CLI 覆盖，但需评估样本量与缓存命中率。
8. **Copula 采样样本量前提**  
   `src/analysis/copula_sampler.py` 估计 80 维相关结构，推荐 `limit_line ≥ analysis.copula.min_draws`（默认 180）；不足时会跳过并记录日志。
9. **图嵌入缓存一致性**  
   期望在分析前执行 `make train-graph`（或直接运行 `scripts/train_graph_embeddings.py`）生成最新嵌入；若缓存缺失，特征通道将自动退化为零向量。
10. **互信息惩罚权重可调**  
   互信息矩阵基于当前窗口估计，如需缩短窗口或切换玩法，应同步调整 `analysis.graph_embedding.weight`，避免惩罚过大或过小。
