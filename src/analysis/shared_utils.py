# -*- coding:utf-8 -*-
"""
通用分析工具函数（供 kl8_analysis 与 kl8_analysis_plus 复用）
- 目录/路径工具
- 数学与序列小工具
"""
from __future__ import annotations
import os
from typing import List, Tuple
import random
from multiprocessing import Process
import threading



def compute_output_dir(random_mode: int, path_label: str) -> str:
    """根据 random_mode 与路径标签生成输出目录路径。
    random_mode: 0 -> results, 1 -> random
    path_label: 追加在目录名之后的自定义标识
    """
    base = "./results" if int(random_mode) == 0 else "./random"
    if not path_label:
        return base + "/"
    return f"{base}_{path_label}/"


def ensure_dir(path: str) -> None:
    """确保目录存在（等幂）。"""
    os.makedirs(path, exist_ok=True)


def check_odd_even(lst: List[int]) -> Tuple[int, int]:
    """统计列表中的奇偶个数。"""
    odd = 0
    even = 0
    for item in lst:
        if item % 2 == 0:
            even += 1
        else:
            odd += 1
    return odd, even


def find_consecutive_number(numbers: List[int]) -> List[Tuple[int, ...]]:
    """找出连续号码的组合段。
    输入需为升序列表。
    """
    if not numbers:
        return []
    consecutive_group: List[Tuple[int, ...]] = []
    group: List[int] = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] == 1:
            group.append(numbers[i])
        else:
            if len(group) > 1:
                consecutive_group.append(tuple(group))
            group = [numbers[i]]
    if len(group) > 1:
        consecutive_group.append(tuple(group))
    return consecutive_group


def _build_unique_filename(file_dir: str, prefix: str, current_time_str: str, cal_nums: int, period_num: str) -> str:
    """生成不重复的结果文件名，格式：prefix_time_calnums_period.csv"""
    assert file_dir.endswith("/"), "file_dir should end with /"
    file_name = f"{file_dir}{prefix}_{current_time_str}_{cal_nums}_{period_num}.csv"
    # 如果存在则添加随机扰动重试
    while os.path.exists(file_name):
        rnd = random.randint(0, 999999)
        file_name = f"{file_dir}{prefix}_{int(current_time_str) + rnd}_{cal_nums}_{period_num}.csv"
    return file_name


def write_results_core(
    rows: List[List[int]],
    file_dir: str,
    file_prefix: str,
    cal_nums: int,
    total_create: int,
    multiple: int,
    multiple_ratio: str,
    period_num: str,
    current_time_str: str,
) -> None:
    """同步写入结果文件（供线程/进程后端调用）。"""
    ensure_dir(file_dir)
    file_name = _build_unique_filename(file_dir, file_prefix, current_time_str, cal_nums, str(period_num))
    with open(file_name, "w") as f:
        # 写表头
        for i in range(cal_nums - 1):
            f.write(f"b{i+1},")
        f.write(f"b{cal_nums}\n")

        # 写内容
        cnt = 0
        item_index = 0
        if multiple > 1:
            div_nums = multiple_ratio.split(",")
            div_a = int(div_nums[0])
            div_b = int(div_nums[1])
        for item in rows:
            if multiple > 1:
                item_index += 1
                if item_index % div_a != div_b:
                    continue
                cnt += 1
            for idx in range(len(item) - 1):
                f.write(f"{item[idx]},")
            f.write(f"{item[-1]}\n")
            if multiple > 1 and cnt >= total_create:
                break


def write_results_async(
    rows: List[List[int]],
    file_dir: str,
    file_prefix: str,
    cal_nums: int,
    total_create: int,
    multiple: int,
    multiple_ratio: str,
    period_num: str,
    current_time_str: str,
    backend: str = "thread",
) -> None:
    """
    异步写入结果文件。

    注意：为兼容 Windows 上的 multiprocessing（spawn 启动方式），目标函数必须
    是顶层可 picklable 的可调用，不能使用 lambda/闭包。

    backend 支持：
    - 'thread'：使用 threading.Thread，适用于 I/O 为主（默认）。
    - 'process'：使用 multiprocessing.Process，避免 GIL 影响。
    """
    target_args = (
        rows,
        file_dir,
        file_prefix,
        cal_nums,
        total_create,
        multiple,
        multiple_ratio,
        period_num,
        current_time_str,
    )
    if backend == "process":
        p = Process(target=write_results_core, args=target_args)
        p.start()
    else:
        t = threading.Thread(target=write_results_core, args=target_args)
        t.start()
