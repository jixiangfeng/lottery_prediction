# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import random
import argparse
import datetime
import os
import sys
from pathlib import Path
# import time
# import threading
# import subprocess
try:
    from tqdm import tqdm
except ImportError:
    # Fallback: create a dummy tqdm for environments where it's not available
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", leave=True):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            return iter(self.iterable)
        def update(self, n=1):
            pass
        def close(self):
            pass
        def set_description(self, desc):
            pass
        @staticmethod
        def write(s):
            print(s)

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
from collections import defaultdict
# 兼容脚本直跑：相对导入失败时，回退到把项目根加入 sys.path 并做绝对导入
try:
    from ..config import *  # type: ignore
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import *  # type: ignore
try:
    from .feature_enhancer import compute_enhanced_scores  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.feature_enhancer import compute_enhanced_scores  # type: ignore
try:
    from .shared_utils import ensure_dir, check_odd_even, find_consecutive_number, compute_output_dir  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.shared_utils import ensure_dir, check_odd_even, find_consecutive_number, compute_output_dir  # type: ignore
try:
    from .shared_utils import ensure_dir, check_odd_even, find_consecutive_number  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.shared_utils import ensure_dir, check_odd_even, find_consecutive_number  # type: ignore
try:
    from .shared_download import ensure_data_available  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.shared_download import ensure_data_available  # type: ignore
try:
    from .rule_miner import build_rule_filter  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.rule_miner import build_rule_filter  # type: ignore
from itertools import combinations
from loguru import logger
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="lottery name")
parser.add_argument('--download', default=1, type=int, help="download data")
parser.add_argument('--limit_line', default=50, type=int, help='limit line')
parser.add_argument('--total_create', default=50, type=int, help='total create')
parser.add_argument('--err_nums', default=1000, type=int, help='err nums')
parser.add_argument('--cal_nums', default=10, type=int, help='cal nums')
parser.add_argument('--analysis_history', default=1, type=int, help='analysis history')
parser.add_argument('--current_nums', default=-1, type=int, help='current nums')
parser.add_argument('--check_in_main', default=0, type=int, help='check in main')
parser.add_argument('--calculate_rate', default=0, type=int, help='calculate rate')
parser.add_argument('--calculate_rate_list', default="5", type=str, help='calculate rate list')
parser.add_argument('--multiple', default=1, type=int, help='multiple')
parser.add_argument('--multiple_ratio', default="1,0", type=str, help='multiple_ratio')
parser.add_argument('--repeat', default=1, type=int, help='repeat') 
parser.add_argument('--path', default="", type=str, help='path')
parser.add_argument('--simple_mode', default=0, type=int, help='simple mode') 
parser.add_argument('--random_mode', default=0, type=int, help='random mode')
parser.add_argument('--max_workers', default=4, type=int, help='max_workers')
parser.add_argument('--advanced_mode', default=0, type=int, help='advanced algorithm mode: 0=original, 1=genetic+bayesian, 2=full_advanced')
parser.add_argument(
    '--feature_mode',
    default="hybrid",
    type=str,
    help='feature ranking mode: hybrid / momentum / cooccurrence'
)
parser.add_argument(
    '--rule_filter',
    default="none",
    type=str,
    choices=["none", "soft", "hard"],
    help='association rule filter mode: none / soft / hard'
)
parser.add_argument(
    '--rule_support',
    default=-1.0,
    type=float,
    help='override minimum support for frequent itemsets (<=0 使用默认值)'
)
parser.add_argument(
    '--rule_confidence',
    default=-1.0,
    type=float,
    help='override minimum confidence for association rules (<=0 使用默认值)'
)
parser.add_argument(
    '--rule_max_size',
    default=0,
    type=int,
    help='override maximum itemset size used for mining (<=0 使用默认值)'
)
parser.add_argument(
    '--rule_penalty',
    default=-1.0,
    type=float,
    help='override penalty weight in soft mode (<=0 使用默认值)'
)
#-------------------------------------------------------------------------------------------------------------#
args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
name = args.name
if args.cal_nums < 0:
    args.cal_nums = abs(args.cal_nums) + 1

# 数据下载由 shared_download 统一处理

# 数据加载将在主程序块中处理

file_path = compute_output_dir(args.random_mode, args.path)

# limit_line = len(ori_numpy)
limit_line = args.limit_line
rule_filter = None
ori_avg_rate = [0.05, 0.05, 0.05, 0.05, 0.01, 0.05]
ori_shiftings_list = [ori_avg_rate] * 10
rate_file = "./kl8_rate.csv"
if os.path.exists(rate_file):
    rate_data = pd.read_csv(rate_file)
    ori_shiftings_list = rate_data.to_numpy()
ori_shiftings = ori_shiftings_list[args.cal_nums - 1]
if len(ori_shiftings) != len(ori_avg_rate):
    ori_shiftings = ori_avg_rate
shifting = ori_shiftings.copy()
total_create = args.total_create * args.multiple
err_nums = args.err_nums
shiftings = []
err = -1
group_size = 50
prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
analysis_history = [3, 5, 7, 9]
err_num_rate = 5
shifting_rate = 0.1

try:
    from .analysis_metrics import cal_not_repeat_rate as _metrics_cal_not_repeat_rate  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.analysis_metrics import cal_not_repeat_rate as _metrics_cal_not_repeat_rate  # type: ignore

def cal_not_repeat_rate(limit=limit_line, result_list=None, j_shiftint=1):
    draws = ori_numpy
    return _metrics_cal_not_repeat_rate(draws, limit, j_shiftint=j_shiftint, result_list=result_list)

try:
    from .analysis_metrics import cal_repeat_rate as _metrics_cal_repeat_rate  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.analysis_metrics import cal_repeat_rate as _metrics_cal_repeat_rate  # type: ignore

def cal_repeat_rate(limit=limit_line, result_list=None, j_shiftint=1):
    draws = ori_numpy
    return _metrics_cal_repeat_rate(draws, limit, args.cal_nums, j_shiftint=j_shiftint, result_list=result_list)

try:
    from .analysis_metrics import cal_hot_cold as _metrics_cal_hot_cold  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.analysis_metrics import cal_hot_cold as _metrics_cal_hot_cold  # type: ignore

def cal_hot_cold(begin=0, end=limit_line):
    return _metrics_cal_hot_cold(ori_numpy, begin, end)

## 计算指定号码组在每期出现的概率
def cal_ball_rate(limit=limit_line, result_list=None, i_shiftint=1):
    hot_rate_times = 0
    cold_rate_times = 0
    times = 0
    if result_list is None:
        result_list = ori_numpy
        i_shiftint = 1
    length = len(result_list[0])
    
    for i in range(limit):
        hot_balls, cold_balls = cal_hot_cold(i + i_shiftint, i + limit_line)
        for j in range(1, length):
            times += 1
            if result_list[i][j] in hot_balls:
                hot_rate_times += 1
            if result_list[i][j] in cold_balls:
                cold_rate_times += 1
    hot_ball_rate = hot_rate_times / times
    cold_ball_rate = cold_rate_times / times
    # logger.info("{:.2f}%".format(hot_ball_rate * 100))
    # logger.info("{:.2f}%".format(cold_ball_rate * 100))
    return hot_ball_rate, cold_ball_rate

try:
    from .analysis_metrics import cal_ball_parity as _metrics_cal_ball_parity  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.analysis_metrics import cal_ball_parity as _metrics_cal_ball_parity  # type: ignore

def cal_ball_parity(limit=limit_line, result_list=None):
    draws = ori_numpy if result_list is None else result_list
    return _metrics_cal_ball_parity(draws, limit)

try:
    from .analysis_metrics import cal_ball_group as _metrics_cal_ball_group  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.analysis_metrics import cal_ball_group as _metrics_cal_ball_group  # type: ignore

def cal_ball_group(limit=limit_line, result_list=None):
    draws = ori_numpy if result_list is None else result_list
    return _metrics_cal_ball_group(draws, limit)

## 找出连续号码的组合（已迁移到 shared_utils.find_consecutive_number）

try:
    from .analysis_metrics import analysis_consecutive_number as _metrics_analysis_consecutive_number  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.analysis_metrics import analysis_consecutive_number as _metrics_analysis_consecutive_number  # type: ignore

def analysis_consecutive_number(limit=limit_line, result_list=None):
    draws = ori_numpy if result_list is None else result_list
    return _metrics_analysis_consecutive_number(draws, limit)

## 分析质数比
def analysis_prime_number(limit=limit_line, result_list=None):
    # prime_group = defaultdict(int)
    total_draws = 0
    if result_list is None:
        result_list = ori_numpy
    for i in range(limit):
        prime_num = 0
        total_draws += 1
        for item in result_list[i]:
            if item in prime_list:
                prime_num += 1
    prime_rate = prime_num / total_draws
    logger.info(prime_rate)
    return prime_rate

## 分析和值概率
def sum_analysis(limit=limit_line, result_list=None):
    sum_group = defaultdict(int)
    sum_rate_group = defaultdict(float)
    total_numbers = 0
    if result_list is None:
        result_list = ori_numpy
    length = len(result_list[0])
    for i in range(limit):
        result_list_split = combinations(result_list[i][1:length], args.cal_nums)
        for item in result_list_split:
            current_sum = sum(item)
            group_index = (current_sum - 1) // group_size
            group_key = f"{group_index * group_size + 1}-{(group_index + 1) * group_size}"
            sum_group[group_key] += 1
            total_numbers += 1
    sum_rate_group = {key: count / total_numbers for key, count in sum_group.items()}
    # logger.info(sum_rate_group)
    return sum_rate_group

## 使用贝叶斯定理分析（修正版）
def bayesian_analysis():
    import numpy as np
    from scipy import stats
    
    number_counts = defaultdict(int)
    total_draws = 0
    total_numbers = 0

    # 先验概率：每个号码被抽中的概率（考虑每期选20个数）
    prior_prob = 20/80  # 修正：每期选20个号码

    for row in ori_numpy[:limit_line]:
        total_draws += 1
        for num in row[1:21]:
            total_numbers += 1
            number_counts[int(num)] += 1

    # 计算后验概率（修正贝叶斯公式）
    posterior_probs = {}
    evidence = sum(number_counts.values()) / len(number_counts)  # 修正边际概率
    
    for num in range(1, 81):
        # 修正似然概率计算
        likelihood = number_counts[num] / total_draws if total_draws > 0 else 0
        # 使用Beta-Binomial共轭先验
        alpha = 1 + number_counts[num]  # 后验参数
        beta = 1 + total_draws - number_counts[num]
        posterior_prob = alpha / (alpha + beta)  # Beta分布的期望
        posterior_probs[num] = posterior_prob

    # 按后验概率排序
    sorted_probs = sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs

## 高级号码生成策略（新增）
def advanced_number_generation_plus(use_genetic=True, use_ml=True):
    """并行版本的高级号码生成策略"""
    try:
        # 简化版的高级生成，适合多进程环境
        candidate_solutions = []
        feature_score_lookup = {}
        feature_debug = None
        enhanced_ranked = []
        try:
            recent_window = max(20, min(limit_line, args.limit_line))
            reference_window = max(recent_window * 2, 60)
            enhanced_ranked, feature_debug = compute_enhanced_scores(
                ori_numpy,
                limit=limit_line,
                recent_window=recent_window,
                reference_window=reference_window,
                decay=0.97,
            )
            if enhanced_ranked:
                feature_score_lookup = {num: score for num, score in enhanced_ranked}
        except Exception as exc:
            logger.warning(f"特征增强初始化失败: {exc}")
            enhanced_ranked = []
        
        # 1. 贝叶斯优化生成
        bayesian_probs = bayesian_analysis()
        top_numbers = [num for num, _ in bayesian_probs[:args.cal_nums*2]]
        if feature_score_lookup:
            mode = (args.feature_mode or "hybrid").lower()
            if feature_debug and mode == "momentum":
                ranked_source = sorted(
                    feature_debug.momentum_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            elif feature_debug and mode == "cooccurrence":
                ranked_source = sorted(
                    feature_debug.co_occurrence_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            else:
                ranked_source = enhanced_ranked
            ranked_numbers = [num for num, _ in ranked_source]
            merged = []
            for num in ranked_numbers + top_numbers:
                if num not in merged:
                    merged.append(num)
            if merged:
                top_numbers = merged[:max(args.cal_nums * 2, len(top_numbers))]
                if args.check_in_main:
                    logger.info("特征增强排序后的候选列表：{}", top_numbers[:args.cal_nums])
        
        # 基于贝叶斯概率的智能选择
        bayesian_solution = []
        remaining_numbers = top_numbers.copy()
        
        while len(bayesian_solution) < args.cal_nums and remaining_numbers:
            # 考虑约束的贪心选择
            best_num = None
            best_score = float('-inf')
            
            for num in remaining_numbers:
                test_solution = sorted(bayesian_solution + [num])
                
                # 简单评分
                score = 0
                
                # 奇偶平衡
                odd_count = sum(1 for n in test_solution if n % 2 == 1)
                target_odd = int(his_odd * len(test_solution))
                score -= abs(odd_count - target_odd)
                
                # 分组平衡
                group_counts = [0] * 8
                for n in test_solution:
                    group_counts[(n-1)//10] += 1
                for i, count in enumerate(group_counts):
                    expected = his_group_rate[i] * len(test_solution)
                    score -= abs(count - expected)
                
                if feature_score_lookup:
                    score += feature_score_lookup.get(num, 0.0)
                
                if score > best_score:
                    best_score = score
                    best_num = num
            
            if best_num:
                bayesian_solution.append(best_num)
                remaining_numbers.remove(best_num)
            else:
                break
        
        # 补齐不足的号码
        while len(bayesian_solution) < args.cal_nums:
            available = [n for n in range(1, 81) if n not in bayesian_solution]
            if available:
                if feature_score_lookup:
                    available.sort(key=lambda n: feature_score_lookup.get(n, 0.0), reverse=True)
                    bayesian_solution.append(available[0])
                else:
                    bayesian_solution.append(random.choice(available))
            else:
                break
                
        return sorted(bayesian_solution)
        
    except Exception as e:
        logger.warning(f"高级生成失败: {e}")
        # 降级到基本随机生成
        return sorted(random.sample(range(1, 81), args.cal_nums))

## 使用K均值聚类算法
def kmeans_clustering(ori_numpy, n_clusters=3):
    # 使用K均值聚类算法
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(ori_numpy)
    
    # 获取聚类标签和中心点
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return labels, centers

## 绘制聚类图
def plot_clusters(ori_numpy, labels, centers):
    plt.scatter(ori_numpy[:, 0], ori_numpy[:, 1], c=labels, cmap='rainbow')
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='black')
    plt.show()

## 验证各概率是否正常
def check_rate(result_list):
    ## 验证总数
    if len(result_list[0][1:]) != args.cal_nums:
        # logger.info("总数异常！",len(result_list[0][1:]),args.cal_nums)
        return -1, False
    
    ## 验证重复
    # for i in range(1,args.cal_nums + 1):
    #     for j in range(i + 1, args.cal_nums + 1):
    #         if result_list[0][i] == result_list[0][j]:
    #             # logger.info("重复异常！", result_list[0][i], result_list[0][j])
    #             return -1, False
    if len(result_list[0]) != len(set(result_list[0])):
        return -1, False
    
    for item in results:
        if result_list[0] == item:
            # logger.info("重复异常！", result_list[0], item)
            return -1, False

    ## 验证重复率
    current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list, j_shiftint=0)
    for i in range(1, args.cal_nums + 1):
        if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
            # logger.info("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
            return 0, False
        
    his_index = 0
    for i in range(args.cal_nums, 0, -1):
        if his_repeat_rate[i] > 0 and his_repeat_rate[i] >= 0.1:
            his_index = i + 1
            break
    if current_repeat_rate[his_index] - his_repeat_rate[his_index] > shifting[0]:
        # logger.info("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
        return 0, False    
    
    ## 验证冷热号
    current_hot_balls, current_cold_balls = cal_ball_rate(limit=1, result_list=result_list, i_shiftint=0)
    if abs(his_hot_balls - current_hot_balls) > shifting[1] or abs(his_cold_balls - current_cold_balls) > shifting[1]:
        # logger.info("冷热号异常！", abs(his_hot_balls - current_hot_balls), abs(his_cold_balls - current_cold_balls), shifting)
        return 1, False
    
    ## 验证奇偶比
    current_odd, current_even = cal_ball_parity(limit=1, result_list=result_list)
    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
        # logger.info("奇偶比异常！", abs(his_odd - current_odd), abs(his_even - current_even), shifting)
        return 2, False
    
    ## 验证号码组
    current_group_rate = cal_ball_group(limit=1, result_list=result_list)
    # for i in range(8):
        # if abs(his_group_rate[i] - current_group_rate[i]) > shifting[3]:
        #     # logger.info("号码组异常！", abs(his_group_rate[i] - current_group_rate[i]), shifting)
        #     return 3, False
        # if his_group_rate[i] == 0 and current_group_rate[i] > 0.1 or his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01 :
        #     # logger.info("号码组异常！", abs(his_group_rate[i] - current_group_rate[i]), shifting)
        #     return -1, False
    for i in range(8):
        if args.cal_nums >= 8:
            if (his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01) or (his_group_rate[i] <= 0.01 and current_group_rate[i] > 0.1):
                # logger.info("号码组异常！", i, abs(his_group_rate[i] - current_group_rate[i]), shifting)
                return 3, False
        else:
            if (current_group_rate[i] > 0 and his_group_rate[i] < 0.01):
                # logger.info("号码组异常！", i, abs(his_group_rate[i] - current_group_rate[i]), shifting)
                return 3, False
    
    ## 验证连续号码
    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=result_list)
    correct_flag = False
    for i in range(2, args.cal_nums + 1):
        if (current_consecutive_rate[i] >= 0.1 and his_consecutive_rate[i] <= 0.01):
            # logger.info("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
            return 4, False
        if (his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0 ):
            correct_flag = True
    if correct_flag == False:
        return 4, False
    # for i in range(2, args.cal_nums + 1):
    #     if abs(his_consecutive_rate[i] - current_consecutive_rate[i]) > shifting[4]:
    #         # logger.info("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
    #         return 4, False
    #     if his_consecutive_rate[i] == 0 and current_consecutive_rate[i] > 0.1 or his_consecutive_rate[i] > 0.1 and current_consecutive_rate[i] < 0.01 :
    #         # logger.info("号码组异常！", abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
    #         return -1, False
    
    ## 验证和值
    current_sum = sum(result_list[0][1:])
    group_index = (current_sum - 1) // group_size
    group_key = f"{group_index * group_size + 1}-{(group_index + 1) * group_size}"
    current_sum_rate = his_sum_rate.get(group_key, 0)
    if current_sum_rate < 0.1:
        # logger.info("和值异常！", current_sum_rate, shifting)
        return -1, False
    
    ## 验证非重复元素等差概率:
    current_march_rate = cal_not_repeat_rate(limit=1, result_list=result_list, j_shiftint=0)
    if abs(current_march_rate - his_not_repeat_rate) > shifting[5]:
        # logger.info("非重复元素等差概率异常！", abs(current_march_rate - his_not_repeat_rate), shifting)
        return 5, False    
    
    return 99, True


def evaluate_rule_penalty(numbers):
    """执行关联规则过滤，返回是否通过、惩罚值与触发规则。"""

    if rule_filter is None:
        return True, 0.0, []
    evaluation = rule_filter.evaluate(numbers)
    return evaluation.accepted, evaluation.penalty, evaluation.violated_rules


def append_result_with_rules(results_list, shiftings_list, base_shifting, numbers):
    """
    封装组合登记逻辑，自动应用规则惩罚。

    返回 (是否成功, 惩罚值, 触发规则列表)。
    """

    accepted, penalty, violations = evaluate_rule_penalty(numbers)
    if not accepted:
        return False, penalty, violations
    penalised = list(base_shifting)
    if penalty > 0:
        penalised = [value + penalty for value in penalised]
    results_list.append(list(numbers))
    shiftings_list.append(penalised)
    return True, penalty, violations

## 判断文件夹是否存在，不存在就创建
def check_dir(path):
    ensure_dir(path)

try:
    from .shared_utils import write_results_async  # type: ignore
except Exception:
    if "PROJECT_ROOT" not in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.shared_utils import write_results_async  # type: ignore

def write_file(lst, file_name="result"):
    period_num = args.current_nums if args.current_nums != -1 else "next"
    write_results_async(
        rows=lst,
        file_dir=file_path,
        file_prefix=file_name,
        cal_nums=args.cal_nums,
        total_create=args.total_create,
        multiple=args.multiple,
        multiple_ratio=args.multiple_ratio,
        period_num=str(period_num),
        current_time_str=current_time,
        backend="process",
    )
## 判断数组中有几个奇数几个偶数（已迁移到 shared_utils.check_odd_even）

## 计算list中大于0的元素的平均值
def cal_average(lst):
    total = 0
    count = 0
    for item in lst:
        if item > 0:
            total += item
            count += 1
    if count == 0:
        return 0
    return total / count

## 分析当前期与历史概率数据的乖离性
def analysis_rate(rate_mode=0):
    global limit_line
    rate_diff = [] 
    result_list = [ori_numpy[0]]
    current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list, j_shiftint=1)
    current_hot_balls, current_cold_balls = cal_ball_rate(limit=1, result_list=result_list, i_shiftint=1)
    current_odd, current_even = cal_ball_parity(limit=1, result_list=result_list)
    current_group_rate = cal_ball_group(limit=1, result_list=result_list)
    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=result_list)
    current_march_rate = cal_not_repeat_rate(limit=1, result_list=result_list, j_shiftint=1)
    for item in analysis_history:
        if item == -1:
            item = len(ori_numpy) - 1
        ori_numpy_except_last = ori_numpy[1:item+1]
        limit_line = item
        his_repeat_rate = cal_repeat_rate(limit=item, result_list=ori_numpy_except_last, j_shiftint=2)
        his_hot_balls, his_cold_balls = cal_ball_rate(limit=item, result_list=ori_numpy_except_last, i_shiftint=2)
        his_odd, his_even = cal_ball_parity(limit=item, result_list=ori_numpy_except_last)
        his_group_rate = cal_ball_group(limit=item, result_list=ori_numpy_except_last)
        his_consecutive_rate = analysis_consecutive_number(limit=item, result_list=ori_numpy_except_last)
        hit_march_rate = cal_not_repeat_rate(limit=item, result_list=ori_numpy_except_last, j_shiftint=2)
        rate_diff.append([item, 
            cal_average([abs(his_repeat_rate[i] - current_repeat_rate[i]) for i in range(1, args.cal_nums + 1)]), 
            cal_average([abs(his_hot_balls - current_hot_balls), abs(his_cold_balls - current_cold_balls)]),
            cal_average([abs(his_odd - current_odd), abs(his_even - current_even)]),
            cal_average([abs(his_group_rate[i] - current_group_rate[i]) for i in range(8)]),
            cal_average([abs(his_consecutive_rate[i] - current_consecutive_rate[i]) for i in range(2, args.cal_nums + 1)]),
            cal_average([abs(hit_march_rate - current_march_rate)])])
    avg_rate = [0.0] * len(rate_diff[0])
    max_rate = [0.0] * len(rate_diff[0])
    avg_rate[0] = "avg"
    max_rate[0] = "max"
    for i in range(len(rate_diff)):
        for j in range(len(rate_diff[i])):
            if args.simple_mode == 0:
                print(round(rate_diff[i][j], 5), end=" ")
            if j > 0:
                # avg_rate[j] += rate_diff[i][j] * ((len(rate_diff) - i) / 10)
                avg_rate[j] += rate_diff[i][j]
                # if rate_diff[i][j] > max_rate[j]:
                #     max_rate[j] = rate_diff[i][j]
                max_rate[j] = max(max_rate[j], rate_diff[i][j])
                # max_rate[j] = max(max_rate[j], shifting[j - 1])
        if args.simple_mode == 0:
            print()
    for i in range(len(avg_rate)):
        if i > 0:
            avg_rate[i] = round(avg_rate[i] / len(analysis_history), 5)
            if args.simple_mode == 0:
                print(avg_rate[i], end=" ")
        else:
            if args.simple_mode == 0:
                print(avg_rate[i], end=" ")
    if args.simple_mode == 0:
        print()
    for i in range(len(max_rate)):
        if i > 0:
            max_rate[i] = round(max_rate[i], 5)
            if args.simple_mode == 0:
                print(max_rate[i], end=" ")
        else:
            if args.simple_mode == 0:
                print(max_rate[i], end=" ")
    if args.simple_mode == 0:            
        print()
    # avg_rate = rate_diff[0]
    result_rate = len(avg_rate[1:]) * [0.0]
    for i in range(len(avg_rate[1:])):
        result_rate[i] = max(avg_rate[i + 1], ori_shiftings[i])

    if rate_mode == 1:
        result_rate = len(avg_rate[1:]) * [0.0]
        for i in range(len(avg_rate[1:])):
            result_rate[i] = max(avg_rate[i + 1], ori_shiftings[i])
        return result_rate
    if rate_mode == 2:
        result_rate = len(max_rate[1:]) * [0.0]
        for i in range(len(max_rate[1:])):
            result_rate[i] = max(max_rate[i + 1], ori_shiftings[i])
        return result_rate
    elif rate_mode == 0:
        return avg_rate[1:]

## 判断list长度是否超过限制
def check_list_length(lst):
    if len(lst) > args.cal_nums + 1:
        return True
    return False

def init_func(rate_mode=1):
    global shifting, cal_shiftings, limit_line, his_repeat_rate, hot_list, cold_list, hot_rate, cold_rate, his_hot_balls, his_cold_balls, his_odd, his_even, his_group_rate, his_consecutive_rate, his_sum_rate, his_not_repeat_rate
    if args.analysis_history == 1:
        cal_shiftings = analysis_rate(rate_mode=rate_mode).copy()
    else:
        analysis_rate(rate_mode=rate_mode)
    limit_line = args.limit_line
    his_repeat_rate = cal_repeat_rate()
    hot_list, cold_list = cal_hot_cold()
    hot_rate, cold_rate = cal_ball_rate()
    his_hot_balls, his_cold_balls = cal_ball_rate(limit_line)
    his_odd, his_even = cal_ball_parity(limit_line)
    his_group_rate = cal_ball_group()
    his_consecutive_rate = analysis_consecutive_number()
    his_sum_rate = sum_analysis()
    his_not_repeat_rate = cal_not_repeat_rate()

def sub_process(i):
    """简化的多进程处理函数"""
    global results, shiftings, shifting, start_time
    current_result = [0]
    
    # 使用高级算法模式
    if args.advanced_mode > 0:
        try:
            advanced_solution = advanced_number_generation_plus(
                use_genetic=(args.advanced_mode >= 1),
                use_ml=(args.advanced_mode >= 2)
            )
            
            if advanced_solution:
                current_result = [0] + advanced_solution
                # 验证高级解是否符合约束
                err_code, check_result = check_rate([current_result])
                if check_result:
                    with results_lock:
                        success, penalty, violations = append_result_with_rules(
                            results,
                            shiftings,
                            shifting,
                            current_result[1:],
                        )
                    if success:
                        if args.simple_mode == 0 and penalty > 0:
                            logger.debug(
                                "高级模式组合触发软规则惩罚：{} penalty={:.3f}",
                                current_result[1:],
                                penalty,
                            )
                        return results, shiftings, shifting, start_time
                    if args.simple_mode == 0 and violations:
                        logger.debug(
            "高级模式组合被规则过滤淘汰：{} -> {} 条规则",
            current_result[1:],
            len(violations),
        )
        except Exception as e:
            logger.warning(f"高级算法失败，使用原始算法: {e}")
    
    # 原始算法作为后备
    err = [0] * len(cal_shiftings)
    err_code_max = -1
    attempt_count = 0
    max_attempts = 1000 if args.advanced_mode == 0 else 500
    while True:
        attempt_count += 1
        if attempt_count > max_attempts:
            # 强制生成一个基本解
            current_result = [0] + sorted(random.sample(range(1, 81), args.cal_nums))
            break
            
        err_code, check_result = check_rate([current_result])
        if check_result:
            accepted, pen, violations = evaluate_rule_penalty(current_result[1:])
            if accepted:
                break
            if args.simple_mode == 0 and violations:
                logger.debug(
                    "规则过滤淘汰组合：{} -> {} 条规则",
                    current_result[1:],
                    len(violations),
                )
            current_result = [0]
            continue
        # err_results.append(current_result)
        current_result = [0]
        if err_code > -1:
            if err_code < err_code_max:
                continue
            err[err_code] += 1
            if err[err_code] > err_nums // err_num_rate:
                err_code_max = err_code
            if err[err_code] > err_nums:
                shifting[err_code] += 0.01 if shifting[err_code] * shifting_rate > 0.01 else shifting[err_code] * shifting_rate

                err[err_code] = 0
                for j in range(err_code + 1, len(err)):
                    shifting[j] = cal_shiftings[j]
                    err[j] = 0
        ## 按比例插入冷热号
        hot_selection = random.randint(int(round((hot_rate - 0) * args.cal_nums,0)), int(round((hot_rate + 0) * args.cal_nums,0)))
        cold_selection = random.randint(int(round((cold_rate - 0) * args.cal_nums,0)), int(round((cold_rate + 0) * args.cal_nums,0)))
        hot_selection = 1 if hot_selection < 1 else hot_selection
        cold_selection = 1 if cold_selection < 1 else cold_selection
        current_result.extend(random.sample(hot_list, hot_selection))
        current_result.extend(random.sample(cold_list, cold_selection))
        
        repeat_flag = True
        temp_result = current_result.copy()
        repeat_start_time = datetime.datetime.now()
        last_result_length = 0
        while repeat_flag:
            repeat_flag = False
            current_result = temp_result.copy()
            ## 随机插入其他数字
            useful_list_odd = []
            useful_list_even = []
            for item in range(1, 81):
                if item not in current_result \
                    and item not in hot_list \
                    and item not in cold_list:
                    # and item not in prime_list:
                    if item % 2 == 1:
                        useful_list_odd.append(item)
                    else:
                        useful_list_even.append(item)
            current_odd, current_even = check_odd_even(current_result[1:])
            odd_need = random.randint(int(round((his_odd - shifting[2]) * args.cal_nums,0)), int(round((his_odd + shifting[2]) * args.cal_nums,0)))
            if current_odd > odd_need:
                odd_need = current_odd
            even_need = args.cal_nums - odd_need
            current_result.extend(random.sample(useful_list_odd, odd_need - current_odd))
            if check_list_length(current_result):
                repeat_flag = True
                continue
            current_result.extend(random.sample(useful_list_even, args.cal_nums + 1 - len(current_result)))
            current_result.sort()
            if args.check_in_main == 1:
                ## 验证重复率
                current_repeat_rate = cal_repeat_rate(limit=1, result_list=[current_result], j_shiftint=0)
                for i in range(1, args.cal_nums + 1):
                    if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
                        repeat_flag = True
                        err_results.append(current_result)
                        break
                ## 验证奇偶比
                if repeat_flag == False:
                    current_odd, current_even = cal_ball_parity(limit=1, result_list=[current_result])
                    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
                        repeat_flag = True
                        err_results.append(current_result)
                ## 验证号码组
                if repeat_flag == False:
                    current_group_rate = cal_ball_group(limit=1, result_list=[current_result])
                    for i in range(8):
                        if args.cal_nums >= 8:
                            if (his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01) or (his_group_rate[i] <= 0.01 and current_group_rate[i] > 0.1):
                                repeat_flag = True
                                err_results.append(current_result)
                                break
                        else:
                            if (current_group_rate[i] > 0 and his_group_rate[i] < 0.01):
                                repeat_flag = True
                                err_results.append(current_result)
                                break
                ## 验证连续号码
                if repeat_flag == False:
                    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=[current_result])
                    correct_flag = False
                    for i in range(2, args.cal_nums + 1):
                        if (current_consecutive_rate[i] >= 0.1 and his_consecutive_rate[i] <= 0.01):
                            repeat_flag = True
                            err_results.append(current_result)
                            break
                        if (his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0 ):
                            correct_flag = True
                    if correct_flag == False:
                        repeat_flag = True
                        err_results.append(current_result)
                        break
            # 注意：在子进程中不需要定期写文件，这会在主进程中处理
            # if (datetime.datetime.now() - start_time).seconds > 60 and len(results) > last_result_length:
            #     last_result_length = len(results)
            #     start_time = datetime.datetime.now()
            #     sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
            #     sorted_results, sorted_shiftings = zip(*sorted_results)
            #     sorted_results = list(sorted_results)
            #     write_file(sorted_results, "result")
    
    with results_lock:
        success, penalty, violations = append_result_with_rules(
            results,
            shiftings,
            shifting,
            current_result[1:],
        )
    if not success:
        if args.simple_mode == 0 and violations:
            logger.debug(
                "标准模式组合被规则过滤淘汰：{} -> {} 条规则",
                current_result[1:],
                len(violations),
            )
        return None
    if args.simple_mode == 0 and penalty > 0:
        logger.debug(
            "标准模式组合触发软规则惩罚：{} penalty={:.3f}",
            current_result[1:],
            penalty,
        )
    shifting = [round(num, 3) for num in shifting]
    return results, shiftings, shifting, start_time

def generate_random_numbers(num_rows, num_nums_per_row):
    results = []
    for _ in range(num_rows):
        row = sorted(random.sample(range(1, 81), num_nums_per_row))
        results.append(row)
    return results

if __name__ == "__main__":
    # 先下载数据（单线程）
    ensure_data_available(name=name, download_flag=args.download)
    
    # 然后加载数据
    ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
    ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()
    
    if args.current_nums > 0 and args.current_nums >= ori_numpy[-1][0] and args.current_nums <= ori_numpy[0][0]:
        index_diff = ori_numpy[0][0] - args.current_nums + 1
        ori_numpy = ori_numpy[index_diff:]
    
    if args.rule_filter in {"soft", "hard"}:
        global rule_filter
        filter_kwargs = {}
        if args.rule_support > 0:
            filter_kwargs["min_support"] = args.rule_support
        if args.rule_confidence > 0:
            filter_kwargs["min_confidence"] = args.rule_confidence
        if args.rule_max_size > 0:
            filter_kwargs["max_itemset_size"] = args.rule_max_size
        if args.rule_penalty >= 0:
            filter_kwargs["penalty_weight"] = args.rule_penalty
        try:
            rule_filter = build_rule_filter(
                draws=ori_data.to_numpy(),
                lottery_code=name,
                limit=limit_line,
                mode=args.rule_filter,
                **filter_kwargs,
            )
        except Exception as exc:
            logger.warning("关联规则挖掘初始化失败，将禁用规则过滤：{}", exc)
            rule_filter = None
    
    check_dir(file_path)
    last_time = ""
    if args.random_mode == 1:
        for _i in range(args.repeat):
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            while current_time == last_time:
                current_time = str(int(current_time) + 1)
            last_time = current_time
            write_file(generate_random_numbers(args.total_create, args.cal_nums), "random")
        exit(0)
    if args.calculate_rate == 1:
        cal_rate_list = args.calculate_rate_list.split(",")
        if int(cal_rate_list[0]) > 0:
            for rate_item in cal_rate_list:
                rate_data = pd.read_csv(rate_file)
                ori_shiftings_list = rate_data.to_numpy()
                args.cal_nums = int(rate_item)
                if args.current_nums == -1:
                    args.current_nums = int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0]) + 1
                if args.current_nums > 0 and args.current_nums >= ori_numpy[-1][0] and args.current_nums <= ori_numpy[0][0]:
                    index_diff = ori_numpy[0][0] - args.current_nums + 1
                    ori_numpy = ori_numpy[index_diff:]
                init_func(rate_mode=0)
                shifting = cal_shiftings.copy()
                err_results = []
                results = []
                start_time = datetime.datetime.now()
                for i in tqdm(range(1, total_create + 1), desc='AnalysisThread {}-{}'.format(str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1), str(args.cal_nums))):
                    current_result = [0]
                    err = [0] * len(cal_shiftings)
                    err_code_max = -1
                    while True:
                        err_code, check_result = check_rate([current_result])
                        if check_result:
                            break
                        # err_results.append(current_result)
                        current_result = [0]
                        if err_code > -1:
                            if err_code < err_code_max:
                                continue
                            err[err_code] += 1
                            if err[err_code] > err_nums // err_num_rate:
                                err_code_max = err_code
                            if err[err_code] > err_nums:
                                shifting[err_code] += 0.01 if shifting[err_code] * shifting_rate > 0.01 else shifting[err_code] * shifting_rate

                                err[err_code] = 0
                                for j in range(err_code + 1, len(err)):
                                    shifting[j] = cal_shiftings[j]
                                    err[j] = 0
                        ## 按比例插入冷热号
                        hot_selection = random.randint(int(round((hot_rate - 0) * args.cal_nums,0)), int(round((hot_rate + 0) * args.cal_nums,0)))
                        cold_selection = random.randint(int(round((cold_rate - 0) * args.cal_nums,0)), int(round((cold_rate + 0) * args.cal_nums,0)))
                        hot_selection = 1 if hot_selection < 1 else hot_selection
                        cold_selection = 1 if cold_selection < 1 else cold_selection
                        current_result.extend(random.sample(hot_list, hot_selection))
                        current_result.extend(random.sample(cold_list, cold_selection))
                        
                        repeat_flag = True
                        temp_result = current_result.copy()
                        repeat_start_time = datetime.datetime.now()
                        last_result_length = 0
                        while repeat_flag:
                            repeat_flag = False
                            current_result = temp_result.copy()
                            ## 随机插入其他数字
                            useful_list_odd = []
                            useful_list_even = []
                            for item in range(1, 81):
                                if item not in current_result \
                                    and item not in hot_list \
                                    and item not in cold_list:
                                    # and item not in prime_list:
                                    if item % 2 == 1:
                                        useful_list_odd.append(item)
                                    else:
                                        useful_list_even.append(item)
                            current_odd, current_even = check_odd_even(current_result[1:])
                            odd_need = random.randint(int(round((his_odd - shifting[2]) * args.cal_nums,0)), int(round((his_odd + shifting[2]) * args.cal_nums,0)))
                            if current_odd > odd_need:
                                odd_need = current_odd
                            even_need = args.cal_nums - odd_need
                            current_result.extend(random.sample(useful_list_odd, odd_need - current_odd))
                            if check_list_length(current_result):
                                repeat_flag = True
                                continue
                            current_result.extend(random.sample(useful_list_even, args.cal_nums + 1 - len(current_result)))
                            current_result.sort()
                            if args.check_in_main == 1:
                                ## 验证重复率
                                current_repeat_rate = cal_repeat_rate(limit=1, result_list=[current_result], j_shiftint=0)
                                for i in range(1, args.cal_nums + 1):
                                    if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
                                        repeat_flag = True
                                        err_results.append(current_result)
                                        break
                                ## 验证奇偶比
                                if repeat_flag == False:
                                    current_odd, current_even = cal_ball_parity(limit=1, result_list=[current_result])
                                    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
                                        repeat_flag = True
                                        err_results.append(current_result)
                                ## 验证号码组
                                if repeat_flag == False:
                                    current_group_rate = cal_ball_group(limit=1, result_list=[current_result])
                                    for i in range(8):
                                        if args.cal_nums >= 8:
                                            if (his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01) or (his_group_rate[i] <= 0.01 and current_group_rate[i] > 0.1):
                                                repeat_flag = True
                                                err_results.append(current_result)
                                                break
                                        else:
                                            if (current_group_rate[i] > 0 and his_group_rate[i] < 0.01):
                                                repeat_flag = True
                                                err_results.append(current_result)
                                                break
                                ## 验证连续号码
                                if repeat_flag == False:
                                    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=[current_result])
                                    correct_flag = False
                                    for i in range(2, args.cal_nums + 1):
                                        if (current_consecutive_rate[i] >= 0.1 and his_consecutive_rate[i] <= 0.01):
                                            repeat_flag = True
                                            err_results.append(current_result)
                                            break
                                        if (his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0 ):
                                            correct_flag = True
                                    if correct_flag == False:
                                        repeat_flag = True
                                        err_results.append(current_result)
                                        break
                            if (datetime.datetime.now() - start_time).seconds > 60 and len(results) > last_result_length:
                                last_result_length = len(results)
                                start_time = datetime.datetime.now()
                                sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
                                sorted_results, sorted_shiftings = zip(*sorted_results)
                                sorted_results = list(sorted_results)
                                write_file(sorted_results, "result")
                    results.append(current_result[1:])
                    shiftings.append(shifting)
                    shifting = [round(num, 3) for num in shifting]
                avg_rate = [round(sum(col) / len(col), 3) for col in zip(*shiftings)]     
                ori_shiftings_list[int(rate_item) - 1] = avg_rate  
                with open(rate_file, "w") as f:
                    for i in range(len(ori_avg_rate) - 1):
                        f.write("s" + str(i + 1) + ",")
                    f.write("s" + str(len(ori_avg_rate)) + "\n")
                    for item in ori_shiftings_list:
                        for index in range(len(item)-1):
                            f.write("{},".format(item[index]))
                        f.write("{}\n".format(item[-1]))
    else: 
        if args.simple_mode == 0:
            print("开始初始化分析环境...")
        init_func(rate_mode=2)      
        shifting = cal_shiftings.copy()
        if args.simple_mode == 0:
            print("初始化完成，开始多线程数据处理...")
        # for _i in tqdm(range(args.repeat), desc='AnalysisThread {}-{}'.format(str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums, str(args.cal_nums)), leave=False):
        for _i in range(args.repeat):
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            while current_time == last_time:
                current_time = str(int(current_time) + 1)
            last_time = current_time
            err_results = []
            results = []
            start_time = datetime.datetime.now()
            results_lock = threading.Lock()
            if args.simple_mode == 0:
                print(f"启动 {total_create} 个处理进程...")
            # 使用线程池来避免多进程的全局变量共享问题
            with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
                future_to_url = {executor.submit(sub_process, i): i for i in range(1, total_create + 1)}
                for future in tqdm(as_completed(future_to_url), total=total_create, desc='AnalysisThread {}-{}-{}'.format(str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums, str(args.cal_nums), _i), leave=False):
                    data = future.result()
                    if data != None:
                        results, shiftings, shifting, start_time = data
            # with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            #     future_to_url = {executor.submit(sub_process, i, results, shiftings, shifting, start_time): i for i in tqdm(range(1, total_create + 1), desc='AnalysisThread {}-{}-{}'.format(str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums, str(args.cal_nums), _i), leave=False)}
            #     for future in as_completed(future_to_url):
            #         data = future.result()
            #         if data != None:
            #             results, shiftings, shifting, start_time = data
            sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
            sorted_results, sorted_shiftings = zip(*sorted_results)
            sorted_results = list(sorted_results)
            write_file(sorted_results, "result")
