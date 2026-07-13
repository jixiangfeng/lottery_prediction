# -*- coding: utf-8 -*-
"""
测试 src.analysis.shared_download 模块

覆盖统一下载助手的行为和边界情况
"""

import pytest
from unittest.mock import patch

from src.analysis.shared_download import ensure_data_available


class TestEnsureDataAvailable:
    """测试统一下载助手功能"""

    @patch("src.common.get_data_run")
    def test_ensure_data_available_download_enabled(self, mock_get_data_run):
        """测试启用下载时的行为"""
        mock_get_data_run.return_value = "mock_data"

        result = ensure_data_available("kl8", download_flag=1)

        # 验证调用了数据获取函数
        mock_get_data_run.assert_called_once_with(name="kl8", cq=0)
        # ensure_data_available 函数返回 None
        assert result is None

    @patch("src.common.get_data_run")
    def test_ensure_data_available_download_disabled(self, mock_get_data_run):
        """测试禁用下载时的行为"""
        mock_get_data_run.return_value = "mock_data"

        result = ensure_data_available("kl8", download_flag=0)

        # download_flag=0 时不应调用下载
        mock_get_data_run.assert_not_called()
        assert result is None

    @patch("src.common.get_data_run")
    def test_ensure_data_available_default_behavior(self, mock_get_data_run):
        """测试默认行为（未指定 download_flag）"""
        mock_get_data_run.return_value = "mock_data"

        result = ensure_data_available("kl8")

        # 默认应该启用下载
        mock_get_data_run.assert_called_once_with(name="kl8", cq=0)
        assert result is None

    @patch("src.common.get_data_run")
    @patch("builtins.print")
    def test_ensure_data_available_prints_messages(self, mock_print, mock_get_data_run):
        """测试是否正确打印开始和结束消息"""
        mock_get_data_run.return_value = "mock_data"

        ensure_data_available("kl8", download_flag=1)

        # 验证打印了开始和结束消息
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("开始下载数据" in call for call in print_calls)
        assert any("数据下载完成" in call for call in print_calls)

    @patch("src.common.get_data_run")
    def test_ensure_data_available_different_names(self, mock_get_data_run):
        """测试不同彩种名称的处理"""
        mock_get_data_run.return_value = "mock_data"

        # 测试不同名称
        ensure_data_available("test_lottery", download_flag=1)
        mock_get_data_run.assert_called_with(name="test_lottery", cq=0)

        mock_get_data_run.reset_mock()
        ensure_data_available("another_name", download_flag=1)
        mock_get_data_run.assert_called_with(name="another_name", cq=0)

    @patch("src.common.get_data_run")
    def test_ensure_data_available_exception_handling(self, mock_get_data_run):
        """测试异常处理"""
        mock_get_data_run.side_effect = Exception("下载失败")

        # 异常应该被向上传播（或根据实际实现进行断言）
        with pytest.raises(Exception, match="下载失败"):
            ensure_data_available("kl8", download_flag=1)

    def test_ensure_data_available_non_one_flags(self):
        """测试非1的 download_flag 值"""
        # 这些情况下都不应该调用下载
        with patch("src.common.get_data_run") as mock_get_data_run:
            ensure_data_available("kl8", download_flag=0)
            ensure_data_available("kl8", download_flag=2)
            ensure_data_available("kl8", download_flag=-1)

            # 所有非1的值都不应调用下载
            mock_get_data_run.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
