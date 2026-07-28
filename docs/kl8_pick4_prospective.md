# 快乐8 Pick4 正式前瞻证据链 v2

## 状态与边界

- `kl8_pick4_prospective_v1` 已被独立审查否决，状态为 `review_rejected`；其既有 state 目录仅保留为被否决谱系，不得删除、修改或读取为正式链。
- 正式 schema 为 `kl8_pick4_prospective_v2`，默认目录为 `state/kl8_pick4_prospective_v2`。
- v2 不改变旧 Frozen 已消费事实：正式登记逐字段验证 `2025045..2026193` 恰好 500 期，并绑定 raw JSONL 全字节、canonical 全历史语义、消费 500 期语义和旧审查报告全字节哈希。
- smoke 仅能由测试直接调用 core 且只能写系统临时目录；其 `profile=smoke`、`formalEligible=false`、`historicalFrozenConsumed=false`，不能进入正式 gate。

## 密钥与签名

正式命令必须通过 `--hmac-key-file` 使用 state 目录外密钥，默认 `~/.hermes/secrets/kl8_pick4_prospective_v2.key`。仅 `register --generate-hmac-key` 可在路径不存在时生成 32 字节随机密钥，权限固定为 `0600`；程序不输出密钥。

每个 JSON artifact 同时包含 `artifactSha256` 与 `artifactHmacSha256`。HMAC 输入为规范 payload、换行和 payload SHA-256。模型文件由 state 同时绑定 `modelSha256` 与 `modelHmacSha256`。`register/update/status` 均验证两类绑定，因此文件所有者只重算自哈希无法伪造链。

## 事务目录

根目录登记先在同父目录 staging 中完成并 `fsync`，最后原子 rename。版本布局为：

```text
state/kl8_pick4_prospective_v2/
├── protocol.json
├── lineage.json
└── versions/
    └── 0000/
        ├── model.txt
        ├── snapshot.json
        ├── anchor_payload.json
        ├── state.json
        └── commit.json
```

从 `0001` 起增加 `observation.json`。更新在 `NNNN.tmp` 中写完模型、状态、快照、观测、锚点和 commit manifest，最后原子发布整个版本目录。status 只承认 commit 完整版本，会清理未提交 `.tmp`；已提交版本拒绝覆盖。

## 精确前瞻目标

snapshot 固定记录 `targetIssue`、`targetDate` 和系统 Asia/Shanghai `createdAt`。同年目标期号为序号加一，跨年为新年份 `001`；当前 KL8 按日要求目标日期为上一开奖日期加一。update 只能消费精确相等的 issue/date，不能跳期或回放。

正式 CLI 不允许注入 `createdAt`。创建时间必须早于目标日 `21:30+08:00`；晚于该时点即 fail closed。未来停开需要开奖前发布不可覆盖 reschedule artifact，v2 当前尚未实现该扩展，因此遇到停开直接拒绝，不能事后跳期。

## 外部锚点

每个版本输出 `externalAnchorPayload`/`anchor_payload.json`，字段包括 version、targetIssue、targetDate、createdAt、snapshot artifact SHA/HMAC。主代理必须在开奖前将该 payload 发布到 Telegram 或等效外部时间戳平台。

update 必须用 `--anchor-receipt-file` 提供外部回执 JSON。回执必须逐字段绑定锚点，包含唯一 `provider`、`messageId`、`anchoredAt`，且 `anchoredAt` 早于开奖时点；缺失、伪造、重复或过晚回执均拒绝消费开奖。代码不查询 Telegram，回执真实性由操作者和外部平台审计。

## 全链验证与 gate

- protocol 绑定唯一 canonical formal 配置、Python/LightGBM/NumPy/pandas/SciPy 版本以及 `requirements.txt`、`requirements-dev.txt`、`pyproject.toml` 和全部模型源码字节哈希；缺失或变化即失败。
- state 递推绑定 previous state、previous model、本期 observation 和当前 snapshot；commit 绑定本版本全部非 commit 文件。
- status/update 遍历 `0000..latest`，验证每个 state/snapshot/model/observation/commit/HMAC/指针，从 canonical history 逐期重导 observation，并用对应 prefix、模型和配置重算每个 snapshot。
- 概率必须逐项严格 `0<p<1` 且和为 20；主 Top4 和五票必须与模型分数 Top4、Top20 round-robin 完全一致。
- gate 只接受 `observed==500`、`profile=formal`、`formalEligible=true`、500 条有效外部锚点和 5 个稳定块；全部 p 值必须有限且位于 `[0,1]`。501、负 p、NaN/Inf 或 smoke 均拒绝。

## 命令

```bash
make kl8-pick4-prospective-register \
  KL8_PICK4_PROSPECTIVE_CONSUME=1 \
  KL8_PICK4_PROSPECTIVE_GENERATE_KEY=1

make kl8-pick4-prospective-update \
  KL8_PICK4_PROSPECTIVE_ANCHOR_RECEIPT=/path/to/telegram_receipt.json

make kl8-pick4-prospective-status
```

不要由自动测试或本次修复执行真实 register；登记后立即将命令返回的 `externalAnchorPayload` 发布到外部平台。

## 信任限制

HMAC 解决“只改本地 artifact 并重算自哈希”的攻击，但无法消除本地主机所有者同时控制文件和密钥的信任边界。本地代码单独不能证明预测在开奖前存在；正式证据仍依赖 Telegram 等外部平台的可信时间戳、消息标识和不可事后改写能力。
