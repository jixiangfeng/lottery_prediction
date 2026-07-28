# 双色球分散组合 v2 仅未来 HMAC 前瞻链

## 定位与边界

本链只从人工批准后的登记边界向未来收集 500 期研究观测，不改动
`ssq_ensemble_v1` 模型、优化器、历史回溯结果、正式门禁、canonical 数据、cron、
其他彩种或任何自动晋级逻辑。所有输出固定为：

- `researchOnly=true`
- `predictionClaim=false`
- `formalRecommendationStatus=uniform_abstain`
- `formalActivation=false`
- `autoPromotion=false`

默认状态目录固定为 `state/ssq_diversified_portfolio_v2`。HMAC 密钥必须通过
`--hmac-key-file` 或环境变量 `SSQ_DIVERSIFIED_PORTFOLIO_V2_HMAC_KEY_FILE`
显式提供，并位于状态目录外；程序不生成、不打印、不复制密钥。

## 固定流程

1. `register`：精确重算并验证当前 `ssq_ensemble_v1` 报告，登记 canonical 最新期、
   数据哈希、报告哈希、协议哈希、构建器哈希与 500 期冻结配置，不生成快照。
2. `snapshot`：仅在 canonical 仍停留于登记边界时生成唯一 `versions/0000`；目标为
   最新开奖之后第一个周二、周四或周日，并且必须在目标日 21:30 前生成。
3. `update`：先验证完整 SHA-256/HMAC 链，再要求 canonical 恰好新增一条且
   `issue/date` 精确匹配锁定目标；目标日 21:30 前拒绝结算，禁止补追、重放、跳期。
4. `status`：验证完整链，汇总 `completed/500` 与 A/B/C 每期均值；不写独立可覆盖
   报告，也不输出签名或密钥内容。

截至当前仓库 canonical 最新记录 `2026085 / 2026-07-26`，按固定周二、周四、
周日排期计算的下一目标是 `2026086 / 2026-07-28`。这只是协议计算示例；本任务
不创建真实状态、密钥或首快照。

## 每期锁定内容

- A：集中式 5×红7、共享模型 Top1 蓝球的影子控制，展开恰好 35 注唯一票。
- B：覆盖优先分散式 5×红7、5 个互异蓝球的主研究组合，展开恰好 35 注唯一票。
- C：仅由固定协议、目标期号和控制编号生成的 32 组同成本控制；每组完整保存
  5×红7、逐组蓝球和 35 注票，可在未来精确复算。

`versions/NNNN/` 永久只含 `snapshot.json`、`observation.json`、`status.json`。
`0000/observation.json` 明确为 `genesis_no_result`，不含实际开奖结果；以后每个版本
结算紧邻前一快照，同时锁定下一目标。第 500 个观测完成后仍写完整版本，但快照为
`completed_snapshot`，不再包含未来组合。

## 完整性与原子性

每个 JSON artifact 使用稳定排序 JSON 计算 SHA-256，并使用状态目录外密钥计算
HMAC-SHA256。`status.json` 绑定本版快照、观测、前一版状态和不可变 manifest；
`current.json` 是原子替换的签名指针。版本先在临时目录完整落盘并 `fsync`，再以
目录重命名发布；若 current 指针推进失败，新版本回滚，不留下半提交版本。

## 运行

父流程先自行创建权限为 `0600`、至少 32 字节的外部密钥，然后显式设置：

```bash
export SSQ_DIVERSIFIED_PORTFOLIO_V2_HMAC_KEY_FILE=/安全路径/ssq-v2.key
make ssq-diversified-portfolio-v2-prospective-register
make ssq-diversified-portfolio-v2-prospective-snapshot
```

每次官方数据经既有人工流程更新 canonical 后，只允许执行一次：

```bash
make ssq-diversified-portfolio-v2-prospective-update
make ssq-diversified-portfolio-v2-prospective-status
```

任何 HMAC、前缀、目标期、目标日期、成本、组合结构、累计指标或 current 指针不一致
都会失败关闭，必须人工审计；不得删除版本后重建或自动跨期追赶。
