import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
from pathlib import Path
try:
    from .shared_download import ensure_data_available  # type: ignore
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.analysis.shared_download import ensure_data_available  # type: ignore

# 兼容脚本直跑：相对导入失败时，回退到把项目根加入 sys.path 并做绝对导入
try:
    from ..config import *  # type: ignore
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import *  # type: ignore

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal_nums_list', default="4,5,7,10", type=str, help='cal_nums_list')
    parser.add_argument('--total_create_list', default="50,100,1000", type=str, help='total_create_list')
    parser.add_argument('--nums_range', default="2023140,2023241", type=str, help='nums_range')
    parser.add_argument('--repeat', default=1, type=int, help='repeat')
    parser.add_argument('--running_mode', default=0, type=int, help='running_mode')
    parser.add_argument('--max_workers', default=4, type=int, help='max_workers')
    parser.add_argument('--random_mode', default=0, type=int, help='random_mode')
    parser.add_argument('--download', default=1, type=int, help='download data before processing')
    return parser

def download_data_if_needed(args):
    """统一下载数据，避免多线程冲突（兼容旧函数名）。"""
    ensure_data_available(name="kl8", download_flag=args.download)

def _main(_total_create, _cal_nums, _current_nums, _process="./kl8_analysis.py", args=None):
    """执行单个分析任务，不下载数据（数据已在主线程下载完成）"""
    try:
        result = subprocess.run([
            "python", _process, "--download", "0", "--total_create", str(_total_create),
            "--cal_nums", str(_cal_nums), "--current_nums", str(_current_nums), "--limit_line", "5",
            "--path", str(_total_create) + '_' + str(abs(int(_cal_nums))), "--repeat", str(args.repeat),
            "--simple_mode", "1", "--random_mode", str(args.random_mode), "--max_workers", str(args.max_workers)
        ], capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"任务失败 [{_total_create}, {_cal_nums}, {_current_nums}]: {e}")
        print(f"错误输出: {e.stderr}")
        raise

def main():
    """主执行函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 使用绝对路径来找到脚本文件
    script_dir = Path(__file__).parent
    kl8_analysis = str(script_dir / "kl8_analysis_plus.py")
    kl8_cash = str(script_dir / "kl8_cash_plus.py")
    cal_nums_list = [int(element) for element in args.cal_nums_list.split(',')]
    total_create_list = [int(element) for element in args.total_create_list.split(',')]
    begin, end = [int(element) for element in args.nums_range.split(',')]

    # 先统一下载数据，避免多线程竞争
    download_data_if_needed(args)

    if args.running_mode in [0, 1]:
        tasks = []
        for _total_create in total_create_list:
            for _cal_nums in cal_nums_list:
                for _current_nums in range(begin, end + 1):
                    tasks.append((_total_create, _cal_nums, _current_nums, kl8_analysis, args))
        
        # 使用ThreadPoolExecutor替代手动线程管理，确保适当的资源控制
        print(f"开始分析阶段: {len(tasks)} 个任务, max_workers={args.max_workers}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_main, *task) for task in tasks]
            completed = 0
            failed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc='AnalysisThread', leave=True):
                try:
                    future.result()  # 获取结果并处理可能的异常
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"分析任务失败: {e}")
            print(f"分析阶段完成: {completed} 成功, {failed} 失败")

    if args.running_mode in [0, 2]:
        tasks = []
        for _total_create in total_create_list:
            for _cal_nums in cal_nums_list:
                _current_nums = -1
                tasks.append((_total_create, _cal_nums, _current_nums, kl8_cash, args))
        
        # 使用ThreadPoolExecutor替代手动线程管理，确保适当的资源控制
        print(f"开始现金分析阶段: {len(tasks)} 个任务, max_workers={args.max_workers}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_main, *task) for task in tasks]
            completed = 0
            failed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc='CashThread', leave=True):
                try:
                    future.result()  # 获取结果并处理可能的异常
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"现金分析任务失败: {e}")
            print(f"现金分析阶段完成: {completed} 成功, {failed} 失败")

if __name__ == '__main__':
    main()
