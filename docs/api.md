# API与命令边界

项目只支持：

| 代码 | 玩法 | 核心入口 |
|---|---|---|
| `ssq` | 双色球 | `scripts/ssq_*.py`、`src/analysis/ssq_*.py` |
| `dlt` | 超级大乐透 | `scripts/dlt_*.py`、`src/analysis/dlt_*.py` |

规则注册表：

```python
from src.lotteries import get_lottery_rule, list_lottery_rules
```

历史数据必须先写原始证据，再由对账步骤生成规范CSV。研究报告和前瞻状态不得作为外部写接口；更新必须通过对应CLI完成。
