# 开发假设与约束

1. **玩法范围**
   快乐 8 链路保持不变；数字彩链路维护福彩3D、排列三、排列五的本地 CSV 统计、候选生成、日报与严格逐期前推回测。
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
11. **数字彩前推口径**
   每个目标期只使用期号更早的历史；默认比较 `current_statistics` 与可复现的 `uniform_random`，两者使用相同过滤条件、候选数和形态配额。
12. **数字彩统计默认值**
   位置频率默认使用 20/50/100/300 期窗口与保守权重，概率使用弱贝叶斯先验平滑；遗漏分使用对数压缩并在 30 期封顶，避免极端遗漏无限放大。
13. **形态防守配额**
   三位彩默认排除豹子，前 5 注至多 1 注组三且整体以组六为主；排列五“三一一/三二”仅占少量防守配额，不作为主流形态。
# 2026-07-16 数字彩第二轮优化假设

- 候选评分是多窗口、重叠特征加权形成的启发式复合对数分，不是规范联合概率或实际开奖概率；默认 `score_floor=2.0` 表示候选不得低于最高复合模型分 2 个单位，多样性只能在该质量池内选取。
- `frequency_weight`、`random_weight`、旧 `candidates` 字段和 `generate_digit_candidates` API 为兼容保留；第二轮主评分由 `marginal_weight`、`pair_weight`、`shape_weight`、`sum_weight`、`span_weight` 与小权重 `omission_weight` 控制。
- 排列五完整评分由共享的前三位复合评分加后两位边际、相关位置对和五位结构附加分组成；不要求最终候选文本与排列三相同。
- 三位彩组选的过滤空间归一化模型质量按同一无序 key 的有序排列模型权重求和，默认排除豹子并严格按 80% 组六、20% 组三分配；排列五不生成组选。
- 多随机基线默认运行 20 次；单次 `uniform_random` 摘要继续保留用于旧消费方，新增均值、5%/95% 分位与当前策略百分位。
- 嵌套调参只在外层训练集尾部验证固定的 `marginal_only`、`joint_balanced`、`joint_heavy`，不读取外层目标期；历史优于随机也不能推导未来中奖概率。
