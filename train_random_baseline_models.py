import os
import shutil
import pandas as pd
import subprocess
import json
import re
import time
import ast
import numpy as np

# --- 配置路径 ---
PROJECT_ROOT = "/home/weizixiang/dev/wk/github/NAS_LEGO" # Updated to current environment path
RANDOM_RESULTS_CSV = os.path.join(PROJECT_ROOT, "gridsearch_output/random_baseline_results.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gridsearch_output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RUN_TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "run_train.sh")

# --- 辅助函数 ---

def backup_csv(file_path):
    """备份 CSV 文件，防止数据损坏"""
    if os.path.exists(file_path):
        backup_path = file_path + ".bak"
        shutil.copy(file_path, backup_path)
        print(f"【备份】已将 {file_path} 备份为 {backup_path}")
    else:
        print(f"Error: 找不到文件 {file_path}")
        exit(1)

def ensure_json_config(row):
    """
    确保架构的 JSON 配置文件存在。
    """
    filename = row['filename']
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(filepath):
        return True
    
    print(f"Error: 配置文件 {filename} 不存在，Random Baseline 不支持自动重建 (因为需要随机种子复现)")
    return False

def parse_accuracy_from_log(log_path):
    """从日志文件中提取 Top-1 Accuracy"""
    if not os.path.exists(log_path):
        return None
    
    final_acc = None
    try:
        with open(log_path, "r") as f:
            for line in f:
                # 匹配模式 1: "Final Accuracy of Best Model (Top-1): 72.33%"
                m1 = re.search(r"Final Accuracy of Best Model \(Top-1\):\s+([\d\.]+)", line)
                if m1:
                    final_acc = float(m1.group(1))
                
                # 匹配模式 2: "Final Test Accuracy (Best Val Model): Top1=72.33%"
                m2 = re.search(r"Final Test Accuracy \(Best Val Model\): Top1=([\d\.]+)", line)
                if m2:
                    final_acc = float(m2.group(1))
    except Exception as e:
        print(f"Error reading log {log_path}: {e}")
        
    return final_acc

def update_csv_row(run_id, acc):
    """将结果写回 CSV"""
    try:
        df = pd.read_csv(RANDOM_RESULTS_CSV)
        
        # 确保 ID 是字符串类型以便匹配
        run_id = str(run_id)
        df['id'] = df['id'].astype(str)
        
        mask = df['id'] == run_id
        if mask.any():
            df.loc[mask, "test_acc_top1"] = acc
            df.to_csv(RANDOM_RESULTS_CSV, index=False)
            print(f"【更新】已更新 CSV: ID {run_id} -> Acc {acc}%")
            return True
        else:
            print(f"Warning: ID {run_id} 在 CSV 中未找到，无法更新。")
            return False
    except Exception as e:
        print(f"Error updating CSV: {e}")
        return False

# --- 主程序 ---

def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    # 1. 备份 CSV
    backup_csv(RANDOM_RESULTS_CSV)
    
    # 2. 读取 CSV 并筛选未训练的模型
    df = pd.read_csv(RANDOM_RESULTS_CSV)
    
    # 如果 CSV 中还没有 test_acc_top1 列，先创建并填充 NaN
    if 'test_acc_top1' not in df.columns:
        print("Warning: 'test_acc_top1' column missing in CSV. Creating it...")
        df['test_acc_top1'] = np.nan
        # 立即保存一次，确保列存在
        df.to_csv(RANDOM_RESULTS_CSV, index=False)
    
    # 将 test_acc_top1 转为 numeric，非数字转为 NaN
    df['test_acc_top1'] = pd.to_numeric(df['test_acc_top1'], errors='coerce')
    
    # 筛选 NaN 的行
    missing_df = df[df['test_acc_top1'].isna()]
    
    total_missing = len(missing_df)
    print(f"\n========================================")
    print(f"检测到 Random Baseline 总任务数: {len(df)}")
    print(f"需要训练的模型数: {total_missing}")
    print(f"========================================\n")
    
    if total_missing == 0:
        print("所有模型均已有 Accuracy 数据，无需训练。")
        return

    # 3. 遍历训练
    for idx, (index, row) in enumerate(missing_df.iterrows()):
        run_id = str(row['id'])
        filename = row['filename']
        
        print(f"[{idx+1}/{total_missing}] 处理 ID: {run_id} (File: {filename})")
        
        # 3.1 确保 JSON 存在
        if not ensure_json_config(row):
            print(f"Skipping {run_id} due to missing config file.")
            continue
            
        # 3.2 构造日志路径和命令
        log_name = f"train_random_{run_id}.log"
        log_path = os.path.join(LOG_DIR, log_name)
        
        # 构造完整的文件路径给 run_train.sh
        # run_train.sh 期望 --arch_path 指向 json 文件
        arch_path = os.path.join(OUTPUT_DIR, filename)
        
        # 构造命令: uv run bash run_train.sh --arch_path ... --log_name ...
        cmd = [
            "uv", "run", "bash", RUN_TRAIN_SCRIPT,
            "--arch_path", arch_path,
            "--log_name", log_name
        ]
        
        try:
            print(f"   >>> 开始训练... (Log: {log_name})")
            # 记录开始时间
            start_time = time.time()
            
            # 调用训练脚本
            subprocess.run(cmd, check=True)
            
            duration = time.time() - start_time
            print(f"   >>> 训练结束，耗时: {duration:.1f}s")
            
            # 3.3 解析结果
            acc = parse_accuracy_from_log(log_path)
            
            if acc is not None:
                print(f"   >>> 解析成功: Top-1 Acc = {acc}%")
                # 3.4 更新 CSV
                update_csv_row(run_id, acc)
                
                # 可选：更新 JSON 文件里的 test_acc_top1
                try:
                    with open(arch_path, 'r') as f:
                        jdata = json.load(f)
                    jdata['test_acc_top1'] = acc
                    with open(arch_path, 'w') as f:
                        json.dump(jdata, f, indent=4)
                except:
                    pass
            else:
                print(f"   !!! Error: 无法从日志 {log_path} 中解析出 Accuracy")
                
        except subprocess.CalledProcessError as e:
            print(f"   !!! 训练进程出错，退出码: {e.returncode}")
        except KeyboardInterrupt:
            print("\n用户手动中断。")
            break
        
        # 休息一下，防止文件 I/O 冲突
        time.sleep(1)

if __name__ == "__main__":
    main()
