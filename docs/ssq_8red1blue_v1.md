# 双色球独立8红+1蓝研究影子 v1

## 研究边界

- 本模块是独立研究影子，不修改既有 A/B/C、模型概率、验证门禁、正式门禁、数据或旧报告语义。
- 永久输出 `researchOnly=true`、`predictionClaim=false`、`equalChanceNoEdge=true`、`formalRecommendationStatus=uniform_abstain`。
- 28 注中的每一注合法号码机会均等；组合不会产生预测优势，也不作官方奖级或奖金声明。

## 固定构造

1. 使用当前 ensemble 红球边际概率，按概率降序、球号升序取得 Top12。
2. 枚举全部 `C(12,8)=495` 个红8候选。
3. 候选按 `sum(log(max(1e-12, p_red)))` 降序，再按红球升序元组字典序排序。
4. 蓝球固定为模型 Top1。
5. 按候选顺序选择首个其 `C(8,6)=28` 张完整红6+蓝票与既有 B35 完整票零重叠的候选；不存在则失败关闭。
6. 审计固定要求 8 个唯一红球、1 个蓝球、28 注唯一票、与 B 重叠 0、B+D8 名义与唯一票数均为 63。

## 全历史诊断

```bash
make ssq-8red1blue-v1-history
```

- 固定预热 120 期，其后全部历史严格执行“先预测/构造/评分，后更新”。
- 每期对比 32 个仅由期号与控制编号生成的确定性同成本随机 8+1 对照。
- 输出红8与实际红6交集分布、任一票红球至少 3/4/5/6、蓝球、精确红6+蓝/无蓝、与 B 重叠及合并票数、逐期记录、数据/协议/报告哈希。
- 配对检验均为未校正、后验、描述性统计，不构成正式门禁。

## 独立前瞻链

```bash
make ssq-8red1blue-v1-prospective-register SSQ_8RED1BLUE_V1_HMAC_KEY_FILE=/外部/0600/key
make ssq-8red1blue-v1-prospective-snapshot SSQ_8RED1BLUE_V1_HMAC_KEY_FILE=/外部/0600/key
make ssq-8red1blue-v1-prospective-status SSQ_8RED1BLUE_V1_HMAC_KEY_FILE=/外部/0600/key
```

- 默认状态目录为 `state/ssq_8red1blue_v1`，固定 horizon 500，密钥变量为独立的 `SSQ_8RED1BLUE_V1_HMAC_KEY_FILE`。
- `register`、`snapshot` 均只允许一次；`update` 要求 canonical 相对链尾恰好新增一个且期号/日期与锁定目标完全一致。
- 不追赶、不重放、不自动更新多期、不自动晋级；每个 artifact 同时绑定规范 JSON SHA-256 与 HMAC-SHA256，版本目录只追加，`current.json` 原子替换。
- 项目不创建真实密钥或真实状态。
