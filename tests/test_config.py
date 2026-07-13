# -*- coding: utf-8 -*-
import pytest

from src import config


def test_only_supports_kl8():
    assert list(config.LOTTERY_CONFIGS.keys()) == ["kl8"]
    kl8 = config.get_lottery_config("kl8")
    assert kl8.code == "kl8"
    assert kl8.red.sequence_len == 20


def test_get_lottery_config_invalid():
    with pytest.raises(ValueError):
        config.get_lottery_config("ssq")


def test_ensure_runtime_directories(tmp_path):
    original_paths = config.PATHS.copy()
    for key in original_paths:
        config.PATHS[key] = tmp_path / key
    config.ensure_runtime_directories()
    for path in config.PATHS.values():
        assert path.exists()
        assert path.is_dir()
    config.PATHS.update(original_paths)


def test_name_path_matches_paths(tmp_path):
    original_paths = config.PATHS.copy()
    original_name_path = config.name_path.copy()
    config.PATHS["data"] = tmp_path / "data"
    config.ensure_runtime_directories()
    config.name_path = {
        code: {
            "name": cfg.name,
            "path": f"{(config.PATHS['data'] / code).as_posix()}/",
        }
        for code, cfg in config.LOTTERY_CONFIGS.items()
    }

    assert config.name_path["kl8"]["name"] == "快乐8"
    assert config.name_path["kl8"]["path"].endswith("/kl8/")

    config.PATHS.update(original_paths)
    config.name_path = original_name_path
