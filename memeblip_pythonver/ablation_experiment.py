# -*- coding: utf-8 -*-
import subprocess
import re
import statistics
import sys
import os
import shutil
from Config import Config

# 要测试的消融配置列表，每个元素包含:
#  1) 说明 desc
#  2) 指令行额外参数 args
ABLATION_SCENARIOS = [
    {
        "desc": "All modules (no flags)",
        "args": []
    },
    {
        "desc": "Remove cosine classifier",
        "args": ["--no-cosine-classifier"]
    },
    {
        "desc": "Remove cosine classifier + pre-output",
        "args": ["--no-cosine-classifier", "--no-pre-output-layer"]
    },
    {
        "desc": "Remove cosine classifier + pre-output + adapter",
        "args": ["--no-cosine-classifier", "--no-pre-output-layer", "--no-adapter"]
    },
    {
        "desc": "Remove cos + pre-output + adapter + text & image proj",
        "args": ["--no-cosine-classifier", "--no-pre-output-layer", "--no-adapter", "--no-text-proj", "--no-image-proj"]
    }
]

# main.py 路径，确保与实际文件位置一致
MAIN_PY = "main.py"
cfg = Config()

# 解析验证输出行的正则表达式 - 更宽松的模式
acc_pattern = re.compile(r"Validation Accuracy:\s*([\d.]+)")
auroc_pattern = re.compile(r"Validation AUROC:\s*([\d.]+)")
f1_pattern = re.compile(r"Validation F1 Score:\s*([\d.]+)")

def run_and_collect_metrics(args_list):
    """
    调用 main.py 训练 + 验证，并解析控制台输出中的指标
    """

    cmd = ["python", MAIN_PY, "--train", "--validate"] + args_list
    print(f"\n执行命令: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # 使用Popen获取实时输出，便于调试
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # 行缓冲
        )
        
        # 实时显示并捕获输出
        full_output = ""
        for line in proc.stdout:
            print(line, end='')  # 实时显示
            sys.stdout.flush()   # 确保立即输出
            full_output += line  # 保存完整内容
        
        proc.wait()
        
        if proc.returncode != 0:
            print(f"命令返回非零代码: {proc.returncode}")
    except Exception as e:
        print(f"执行出错: {e}")
        return None, None, None
    
    # 匹配控制台输出提取验证指标
    acc_match = acc_pattern.search(full_output)
    auroc_match = auroc_pattern.search(full_output)
    f1_match = f1_pattern.search(full_output)
    
    # 默认值
    acc_val, auroc_val, f1_val = None, None, None
    
    if acc_match:
        try:
            acc_val = float(acc_match.group(1))
            print(f"找到 Accuracy: {acc_val}")
        except ValueError:
            print(f"无法解析 Accuracy 值: {acc_match.group(1)}")
    else:
        print("未找到 Accuracy 值")
    
    if auroc_match:
        try:
            auroc_val = float(auroc_match.group(1))
            print(f"找到 AUROC: {auroc_val}")
        except ValueError:
            print(f"无法解析 AUROC 值: {auroc_match.group(1)}")
    else:
        print("未找到 AUROC 值")
    
    if f1_match:
        try:
            f1_val = float(f1_match.group(1))
            print(f"找到 F1: {f1_val}")
        except ValueError:
            print(f"无法解析 F1 值: {f1_match.group(1)}")
    else:
        print("未找到 F1 值")
    
    return acc_val, auroc_val, f1_val

def main():
    results = []

    # 创建并清理检查点目录
    checkpoint_dir = cfg.checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 清理可能存在的旧检查点
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"清理检查点时出错: {e}")

    # 针对每种消融配置执行多次训练验证
    for scenario in ABLATION_SCENARIOS:
        print(f"\n\n开始执行场景: {scenario['desc']}")
        print("=" * 80)
        
        acc_list, auroc_list, f1_list = [], [], []
        for run_idx in range(3):
            print(f"\n运行 {run_idx+1}/1")

            # 每次运行前清理检查点目录
            if os.path.exists(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    file_path = os.path.join(checkpoint_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"清理检查点时出错: {e}")
            
            # 也清理可能的默认lightning_logs
            lightning_logs = "D:/NIW/memeblip_pythonver/lightning_logs"
            if os.path.exists(lightning_logs):
                try:
                    shutil.rmtree(lightning_logs)
                except Exception as e:
                    print(f"清理lightning_logs时出错: {e}")
            
            acc, auroc, f1 = run_and_collect_metrics(scenario["args"])
            if acc is not None:
                acc_list.append(acc)
            if auroc is not None:
                auroc_list.append(auroc)
            if f1 is not None:
                f1_list.append(f1)
        
        # 计算均值和标准差
        if acc_list:
            acc_mean = statistics.mean(acc_list)
            acc_stdev = statistics.pstdev(acc_list) if len(acc_list) > 1 else 0
        else:
            acc_mean, acc_stdev = 0, 0
        
        if auroc_list:
            auroc_mean = statistics.mean(auroc_list)
            auroc_stdev = statistics.pstdev(auroc_list) if len(auroc_list) > 1 else 0
        else:
            auroc_mean, auroc_stdev = 0, 0
        
        if f1_list:
            f1_mean = statistics.mean(f1_list)
            f1_stdev = statistics.pstdev(f1_list) if len(f1_list) > 1 else 0
        else:
            f1_mean, f1_stdev = 0, 0
        
        results.append(
            f"Scenario: {scenario['desc']}\n"
            f"ACC = {acc_mean:.2f} ± {acc_stdev:.2f}\n"
            f"AUROC = {auroc_mean:.2f} ± {auroc_stdev:.2f}\n"
            f"F1 = {f1_mean:.2f} ± {f1_stdev:.2f}\n"
        )

    # 将结果写入本地文件
    with open("ablation_results.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(res + "\n")
    
    print("\n所有实验完成，结果已保存到 ablation_results.txt")

if __name__ == "__main__":
    main()