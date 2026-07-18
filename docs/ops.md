# 运维指南

## 环境

- Python 3.11；推荐使用 `uv run --python 3.11 --with-requirements requirements-dev.txt`。
- 训练、评估和日报只读取本地 CSV，不联网。
- 只有 `make digit-fetch` 会访问固定白名单来源。

## 命令

```bash
make digit-fetch DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make digit-learned-ranker-train DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make digit-learned-ranker-evaluate DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make digit-learned-ranker-daily DIGIT_LOTTERY=fc3d DIGIT_CSV=data/fc3d/official_history.csv
make ci
```

冒烟可设置 `DIGIT_V4_SMOKE=1`；不得将 smoke 结果用于效果结论。

## 冻结边界

- 参数文件必须保留切分、数据、源码、参数和产物指纹。
- Frozen Test 不参与参数或候选预算选择。
- canonical 校验只覆盖训练时 `split.testEnd` 冻结前缀。
- 可以追加冻结边界之后的新开奖；修改冻结前缀必须使评估失效。
- 同一身份的 immutable 产物禁止覆盖。

## 监控

- 长任务应监控 Python 子进程 CPU，而不只看 `uv` 父进程。
- 评估产物只在完整结束后写出；运行中无文件不代表卡死。
- `evaluationValidation` 任一指纹或 frozen-test 证明失败，都必须关闭激活。
- `max_perm/mean_top_perm` 不得显示为百分比概率。

## 故障排查

1. CSV 列错误：先执行标准化入口或核对 `期号,开奖号码` 格式。
2. 源码指纹不匹配：旧参数不能继续用于新源码，应重新训练。
3. canonical 指纹不匹配：检查冻结前缀是否被修改。
4. 运行过慢：提高开发阶段 stride 或减少 trial；正式 Frozen Test 不应抽样。
5. 结果与随机相近：如实保留，不通过闸门，不继续查看 Frozen Test 调参。

## 清理说明

历史 v1、概率 v2、在线概率 v3 的可执行入口、状态、测试、文档和报告已删除。不要从旧产物恢复兼容路径；新功能直接在 learned ranker 架构下设计并重新验证。
