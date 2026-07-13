# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path
# import subprocess
import threading
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# 兼容脚本直跑：相对导入失败时，回退到把项目根加入 sys.path 并做绝对导入
try:
    from ..config import *  # type: ignore
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import *  # type: ignore
from itertools import combinations
# from loguru import logger
# from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from .shared_cash import CASH_SELECT_LIST, CASH_PRICE_LIST  # type: ignore
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.shared_cash import CASH_SELECT_LIST, CASH_PRICE_LIST  # type: ignore


parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="lottery name")
parser.add_argument('--download', default=1, type=int, help="download data")
parser.add_argument('--cash_file_name', default="-1", type=str, help='cash_file_name')
parser.add_argument('--current_nums', default=-1, type=int, help='current nums')
parser.add_argument('--path', default="", type=str, help='path')
parser.add_argument('--simple_mode', default=0, type=int, help='simple mode')
parser.add_argument('--random_mode', default=0, type=int, help='random mode')
parser.add_argument('--cal_nums', default=10, type=int, help='cal_nums')
#--------------------------------------------------------------------------------------------------#
parser.add_argument('--limit_line', default=0, type=int, help='useless')
parser.add_argument('--total_create', default=50, type=int, help='useless')
parser.add_argument('--multiple', default=1, type=int, help='useless')
parser.add_argument('--multiple_ratio', default="1,0", type=str, help='useless')
parser.add_argument('--repeat', default=1, type=int, help='useless')
parser.add_argument('--calculate_rate', default=0, type=int, help='useless')
parser.add_argument('--calculate_rate_list', default="5", type=str, help='useless')
parser.add_argument('--max_workers', default=4, type=int, help='useless')
args = parser.parse_args()

if args.random_mode == 0:
    if args.path == "":
            file_path = "./results/" 
    else:
        file_path = "./results_" + args.path + "/"
elif args.random_mode == 1:
    if args.path == "":
        file_path = "./random/"
    else:
        file_path = "./random_" + args.path + "/"
endstring = ["csv"]
name = args.name
nums_index = 0
cal_nums = int(args.cal_nums)
content = []

# 数据下载函数，单独处理
def download_data_if_needed():
    """单线程下载数据"""
    if args.download == 1:
        if args.simple_mode == 0:
            print("开始下载数据...")
        try:
            from ..common import get_data_run  # type: ignore
        except Exception:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from src.common import get_data_run  # type: ignore
        get_data_run(name=name, cq=0)
        if args.simple_mode == 0:
            print("数据下载完成")

# 数据加载将在主程序块中处理
# if args.current_nums >= 0:
#     index = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0] - (args.current_nums + 1)
#     if index >= 0:
#         ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[index][1:]
cash_select_list = CASH_SELECT_LIST
cash_price_list = CASH_PRICE_LIST

def sub_check_lottery(item, cash_select, cash_price, cash_list):
    for index in range(len(cash_select)):
        ori_split = list(combinations(ori_numpy, cash_select[index]))
        cash_split = list(combinations(item, cash_select[index]))
        cash_set = set(ori_split) & set(cash_split)
        if cash_select[index] != 0:
            cash_list[index] += len(cash_set)
            if cash_price[index] != 0 and len(cash_set) != 0:
                return cash_list
        elif cash_select[index] == 0 and len(cash_set) == 0:
            cash_list[index] += 1
            return cash_list

def check_lottery(file_path, filename, args, ori_data_param):
    # 使用本地变量替代全局变量，避免线程冲突
    ori_data = ori_data_param
    local_content = []
    local_all_cash = 0
    local_all_lucky = 0
    cal_nums = int(args.cal_nums)  # 初始化本地cal_nums变量
    cash_file_name = file_path + filename
    filename_split = filename.split('_') 
    if len(filename_split) == 4:
        period_str = filename_split[-1].split('.')[0]
        if period_str != "next" and period_str.isdigit() and int(period_str) > 0:
            args.current_nums = int(period_str)
    
    # 从文件名中获取索引，避免使用全局变量
    file_index = len([f for f in os.listdir(file_path) if f <= filename and f.endswith('.csv')])
    
    # 设置默认的ori_numpy，然后根据current_nums调整
    ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][1:]
    if args.current_nums >= ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[-1][0] and args.current_nums <= ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0]:
        index = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0] - args.current_nums
        if index >= 0:
            ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[index][1:]
    cash_data = pd.read_csv(cash_file_name)
    cash_numpy = cash_data.to_numpy()
    if cal_nums >= 0:
        cal_nums = cash_numpy.shape[1]
    else:
        cal_nums = abs(cal_nums)
    cash_select = cash_select_list[cal_nums]
    cash_price = cash_price_list[10 - (cal_nums)]
    cash_list = [0] * len(cash_select)

    # for j in tqdm(range(len(cash_numpy)), desc='subCashThread {}'.format(args.path), leave=False):
    for item in cash_numpy:
        # item = cash_numpy[j]
        for index in range(len(cash_select)):
            ori_split = list(combinations(ori_numpy, cash_select[index]))
            cash_split = list(combinations(item, cash_select[index]))
            cash_set = set(ori_split) & set(cash_split)
            if cash_select[index] != 0:
                cash_list[index] += len(cash_set)
                if cash_price[index] != 0 and len(cash_set) != 0:
                    break
            elif cash_select[index] == 0 and len(cash_set) == 0:
                cash_list[index] += 1
                break
    # with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
    #     future_to_url = {executor.submit(sub_check_lottery, item, cash_select, cash_price, cash_list): item for item in cash_numpy}
    #     for future in as_completed(future_to_url):
    #         data = future.result()
    #         if data != None:
    #             cash_list = data
    total_cash = 0
    for i in range(len(cash_select)):
        total_cash += cash_list[i] * cash_price[i]
    if args.simple_mode == 0 or (args.simple_mode == 2 and total_cash / (len(cash_numpy) * 2) * 100 >= 100):
        local_content.append("{}, 第{}张，本期共投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(args.path, file_index, len(cash_numpy) * 2, total_cash, total_cash / (len(cash_numpy) * 2) * 100))
    local_all_cash += len(cash_numpy) * 2
    local_all_lucky += total_cash
    return local_all_cash, local_all_lucky, local_content, args

## 判断文件是否存在
def check_file(_file_name):
    if os.path.exists(_file_name):
        return True
    else:
        return False

## 多线程调用写入文件
def write_file(_content,_file_name="./kl8_runnint_results.txt"):
    # t = threading.Thread(target=write_file_core, args=(_content, _file_name))
    t = Process(target=write_file_core, args=(_content, _file_name))
    t.start()

## 写入文件
def write_file_core(_content,_file_name="./kl8_runnint_results.txt"):
    if check_file(_file_name):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(_file_name, write_mode) as f:
        for item in _content:
            f.write(item + "\n")

if __name__ == "__main__":
    # 先下载数据（单线程）
    download_data_if_needed()
    
    # 然后加载数据
    ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
    ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][1:]
    
    nums_index = 0
    if args.path == "" or args.cash_file_name != "-1":
        if args.random_mode == 0:
            if args.path == "":
                    file_path = "./results/" 
            else:
                file_path = "./results_" + args.path + "/"
        elif args.random_mode == 1:
            if args.path == "":
                file_path = "./random/"
            else:
                file_path = "./random_" + args.path + "/" 
        if args.cash_file_name != "-1":
            cash_file_name = file_path + args.cash_file_name + ".csv"
        else:
            ## 寻找目录下最新的文件
            import os
            file_list = [_ for _ in os.listdir(file_path) if _.split('.')[1] in endstring]
            file_list.sort(key=lambda fn: os.path.getmtime(file_path + fn))
            cash_file_name = file_path + file_list[-1]   
            filename_split = file_list[-1].split('_')
            if len(filename_split) == 4:
                period_str = filename_split[-1].split('.')[0]
                if period_str != "next" and period_str.isdigit() and int(period_str) > 0:
                    args.current_nums = int(period_str)
        # 处理单个文件
        filename = os.path.basename(cash_file_name)
        file_dir = os.path.dirname(cash_file_name) + "/"
        all_cash, all_lucky, file_content, _ = check_lottery(file_dir, filename, args, ori_data)
        content.extend(file_content)
    else:
        if args.random_mode == 0:
            if args.path == "":
                    file_path = "./results/" 
            else:
                file_path = "./results_" + args.path + "/"
        elif args.random_mode == 1:
            if args.path == "":
                file_path = "./random/"
            else:
                file_path = "./random_" + args.path + "/" 
        all_cash, all_lucky = 0, 0
        import os
        file_list = [_ for _ in os.listdir(file_path) if _.split('.')[1] in endstring]
        file_list.sort(key=lambda fn: os.path.getmtime(file_path + fn))
        
        # 使用线程锁保护共享变量
        results_lock = threading.Lock()
        
        if args.simple_mode == 0:
            print(f"开始处理 {len(file_list)} 个文件...")
        
        # 使用线程池来避免多进程的全局变量共享问题
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            future_to_file = {executor.submit(check_lottery, file_path, filename, args, ori_data): filename for filename in file_list}
            for future in tqdm(as_completed(future_to_file), total=len(file_list), desc='CashThread {}'.format(args.path), leave=False):
                data = future.result()
                if data != None:
                    thread_all_cash, thread_all_lucky, thread_content, thread_args = data
                    with results_lock:
                        all_cash += thread_all_cash
                        all_lucky += thread_all_lucky
                        content.extend(thread_content)

        # with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
        #     future_to_url = {executor.submit(check_lottery, file_path, file_list[filename_index], args): file_list[filename_index] for filename_index in tqdm(range(len(file_list)), desc='CashThread {}'.format(args.path), leave=False)}
        #     for future in as_completed(future_to_url):
        #         data = future.result()
                # if data != None:
                #     all_cash, all_lucky, content, args = data
        # 计算总返奖率，避免除零错误
        if all_cash > 0:
            return_rate = all_lucky / all_cash * 100
            content.append("{}, 总投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(args.path, all_cash, all_lucky, return_rate))
        else:
            content.append("{}, 总投入0元，总奖金为{}元，无法计算返奖率。".format(args.path, all_lucky))
    write_file(content)