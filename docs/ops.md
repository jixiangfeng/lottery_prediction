# 运维说明

## 质量闸门

```bash
make ci
```

包含Black、isort、flake8、mypy、pytest、覆盖率不低于80%和compileall。

## 数据同步

```bash
make ssq-fetch && make ssq-reconcile
make dlt-fetch && make dlt-reconcile
make kl8-fetch
```

任何官方分页、玩法身份、号码范围、重复期号或哈希冲突都必须失败关闭。

## 前瞻链

- 双色球：B、D8、E各用独立外置HMAC密钥；
- 快乐8：选4联合A/B链使用独立外置HMAC密钥；
- 超级大乐透：Validation已拒绝，未创建未来链。

只允许结算恰好一个已锁定目标；缺少开奖前快照时禁止回填。密钥权限必须不高于`0600`。
