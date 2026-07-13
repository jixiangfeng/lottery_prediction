#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kl8_running.py 的单元测试
测试ThreadPoolExecutor升级后的并发运行功能
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# 添加项目根目录到sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 模拟subprocess.run的结果
class MockCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestKl8Running(unittest.TestCase):
    """测试kl8_running模块的并发功能"""

    def setUp(self):
        """测试前准备"""
        # 模拟命令行参数
        self.mock_args = MagicMock()
        self.mock_args.cal_nums_list = "4,5"
        self.mock_args.total_create_list = "50,100"
        self.mock_args.nums_range = "2023140,2023142"
        self.mock_args.repeat = 1
        self.mock_args.running_mode = 0
        self.mock_args.max_workers = 2
        self.mock_args.random_mode = 0
        self.mock_args.download = 1

    @patch("src.analysis.kl8_running.ensure_data_available")
    @patch("src.analysis.kl8_running.subprocess.run")
    def test_main_function_success(self, mock_subprocess, mock_ensure_data):
        """测试_main函数成功执行"""
        from src.analysis.kl8_running import _main

        # 模拟成功的subprocess调用
        mock_subprocess.return_value = MockCompletedProcess(returncode=0)

        # 调用函数
        _main(50, 4, 2023140, "test_process.py", self.mock_args)

        # 验证subprocess.run被正确调用
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]  # 获取第一个位置参数（命令列表）

        # 验证关键参数
        self.assertIn("test_process.py", args)
        self.assertIn("--total_create", args)
        self.assertIn("50", args)
        self.assertIn("--cal_nums", args)
        self.assertIn("4", args)
        self.assertIn("--current_nums", args)
        self.assertIn("2023140", args)
        self.assertIn("--download", args)
        self.assertIn("0", args)  # download应该设为0

    @patch("src.analysis.kl8_running.ensure_data_available")
    @patch("src.analysis.kl8_running.subprocess.run")
    def test_main_function_failure(self, mock_subprocess, mock_ensure_data):
        """测试_main函数处理失败情况"""
        from src.analysis.kl8_running import _main

        # 模拟失败的subprocess调用
        mock_subprocess.side_effect = Exception("subprocess failed")

        # 验证异常被正确抛出
        with self.assertRaises(Exception):
            _main(50, 4, 2023140, "test_process.py", self.mock_args)

    @patch("src.analysis.kl8_running.ensure_data_available")
    def test_download_data_if_needed(self, mock_ensure_data):
        """测试统一下载数据功能"""
        from src.analysis.kl8_running import download_data_if_needed

        # 调用函数
        download_data_if_needed(self.mock_args)

        # 验证ensure_data_available被正确调用
        mock_ensure_data.assert_called_once_with(name="kl8", download_flag=1)

    @patch("src.analysis.kl8_running.ensure_data_available")
    @patch("src.analysis.kl8_running.subprocess.run")
    @patch("builtins.print")
    def test_threadpool_executor_usage(
        self, mock_print, mock_subprocess, mock_ensure_data
    ):
        """测试ThreadPoolExecutor的使用"""
        from src.analysis.kl8_running import main

        # 模拟成功的subprocess调用
        mock_subprocess.return_value = MockCompletedProcess(returncode=0)

        # 模拟命令行参数
        test_args = [
            "--cal_nums_list",
            "4",
            "--total_create_list",
            "50",
            "--nums_range",
            "2023140,2023141",
            "--running_mode",
            "1",  # 只运行分析阶段
            "--max_workers",
            "2",
        ]

        # 模拟sys.argv
        with patch("sys.argv", ["kl8_running.py"] + test_args):
            main()

        # 验证ensure_data_available被调用
        mock_ensure_data.assert_called()

        # 验证subprocess.run被调用了正确的次数（2个nums_range * 1个cal_nums * 1个total_create = 2次）
        expected_calls = 2
        self.assertEqual(mock_subprocess.call_count, expected_calls)

        # 验证进度信息被打印
        print_calls = [
            call_obj.args[0] for call_obj in mock_print.call_args_list if call_obj.args
        ]
        task_info_printed = any("开始分析阶段" in str(msg) for msg in print_calls)
        completion_info_printed = any("分析阶段完成" in str(msg) for msg in print_calls)

        self.assertTrue(task_info_printed, "应该打印任务启动信息")
        self.assertTrue(completion_info_printed, "应该打印完成信息")

    def test_script_path_resolution(self):
        """测试脚本路径解析"""
        from src.analysis.kl8_running import create_parser

        # 测试parser创建功能
        parser = create_parser()
        self.assertIsNotNone(parser)

        # 测试默认参数
        args = parser.parse_args([])
        self.assertEqual(args.max_workers, 4)

    def test_parameter_parsing(self):
        """测试参数解析功能"""
        from src.analysis.kl8_running import create_parser

        parser = create_parser()

        # 测试默认参数
        args = parser.parse_args([])
        self.assertEqual(args.cal_nums_list, "4,5,7,10")
        self.assertEqual(args.total_create_list, "50,100,1000")
        self.assertEqual(args.nums_range, "2023140,2023241")
        self.assertEqual(args.max_workers, 4)
        self.assertEqual(args.download, 1)

        # 测试自定义参数
        args = parser.parse_args(
            ["--cal_nums_list", "1,2,3", "--max_workers", "8", "--download", "0"]
        )
        self.assertEqual(args.cal_nums_list, "1,2,3")
        self.assertEqual(args.max_workers, 8)
        self.assertEqual(args.download, 0)


if __name__ == "__main__":
    unittest.main()
