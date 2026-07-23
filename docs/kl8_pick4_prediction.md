# 快乐8选4安全预测入口

## 定义

- 每注从`1..80`选择4个唯一号码。
- 每期官方开奖20个唯一号码。
- 选4与选5使用不同的同成本基线，不复用Pick5每票平均命中`1.25`门槛。

公平随机下一注Pick4的理论命中分布：

| 命中数 | 概率 |
|---:|---:|
| 0 | 30.8321% |
| 1 | 43.2732% |
| 2 | 21.2635% |
| 3 | 4.3248% |
| 4 | 0.3063% |

理论平均命中为`4×20/80=1.0`；至少中1个为`69.1679%`，至少中2个为`25.8947%`，至少中3个为`4.6311%`。

以上仅为号码命中分布，不包含奖金、票价或收益率假设。

## 入口

默认安全入口：

```bash
make kl8-pick4-predict-today
```

在没有通过独立统计准入时固定返回：

```text
formalRecommendation=null
userVisibleCandidates=[]
promotionPassed=false
recommendationEnabled=false
```

显式娱乐测试入口：

```bash
make kl8-pick4-test-today
```

可用`KL8_PICK4_TEST_TICKETS`调整测试注数。测试组合由目标日期和Frozen之前的开发数据SHA-256生成固定seed，同输入可复现；每注4个唯一`1..80`号码，组合之间互不重复。

## 安全边界

- 使用快乐8两遍CSV加载器；最新500期Frozen只读取期号和日期边界，号码字段不解析。
- 测试组合标记为`uniform_random_test_only`，不会写入`formalRecommendation`或`userVisibleCandidates`。
- 当前历史特征发现结论为`uniform/no-signal`，所以没有把频率、遗漏、EWMA或pair特征包装为Pick4正式预测。
- 不自动抓取、不覆盖状态、不打开Validation、不产生收益保证。

## 固定排名挑战器结果（2026-07-23）

执行入口：

```bash
make kl8-pick4-rank-challenger
```

该挑战器修正版固定使用`LambdaRank`且显式锁定`lambdarank_truncation_level=4`与`label_gain=[0,1]`，特征仅为`frequency80/320`、两者差、`omissionLog`、`ewma80`和`inPrevious`；前300期初始训练，随后1214期严格前向评估，每50期扩展重训。每期Top20按排名层轮转为5张互不重叠Pick4票。排名分数只做固定风险映射：`q=normalize_sum20(sigmoid(0.1*zscore(score)))`，再以`lambda=0.1`收缩到均匀；该值不是校准概率证据。

审查修正版v2真实结果：主Top4平均`1.0247`（随机`1.0`，精确`p=0.1594`）；五注每票平均`1.0112`，组合总平均`5.0560`（随机`5.0`，精确`p=0.1255`）。Delta LogLoss/Brier只有`0.00000209/0.00000080`，块bootstrap p值为`0.4313/0.4198`；四项Holm校正后最小p值`0.5022`。第2块主Top4低于随机，第4/5块proper scores恶化，联合闸门失败。

旧v1报告没有显式锁定LambdaRank Top4截断，保留为被审查替代的无效审计记录，不用于结论。当前有效报告为`reports/development/kl8_pick4_rank_challenger_v2_official_20260723.json`，证据状态固定为`exploratory_post_failure_reused_development`，不能晋级。
