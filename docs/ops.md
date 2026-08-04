# 运维说明

## 质量闸门

```bash
make ci
```

包含 Black、isort、flake8、mypy、pytest、覆盖率不低于80%和 compileall。

## 数据同步

```bash
make ssq-fetch && make ssq-reconcile
```

任何官方分页、玩法身份、号码范围、重复期号或哈希冲突都必须失败关闭。

## 前瞻链

仅双色球 D8 使用外置 HMAC 密钥。只允许结算恰好一个已锁定目标；缺少开奖前快照时禁止回填。密钥权限必须不高于 `0600`。
