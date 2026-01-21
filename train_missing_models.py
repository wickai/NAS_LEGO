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
PROJECT_ROOT = "/home/weizixiang/dev/wk/github/NAS_LEGO"
ALL_RESULTS_CSV = os.path.join(PROJECT_ROOT, "gridsearch_output/all_results.csv")
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

def parse_list_string(s):
    """解析 CSV 中的字符串列表，例如 '[1, 2, 3]'"""
    try:
        return ast.literal_eval(s)
    except:
        return []

def ensure_json_config(row):
    """
    确保架构的 JSON 配置文件存在。
    如果文件丢失，使用 CSV 中的 op_codes 和 width_codes 重建它。
    """
    filename = row['filename']
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(filepath):
        return True
    
    print(f"Warning: 配置文件 {filename} 不存在，正在根据 CSV 内容重建...")
    
    try:
        # 重建 JSON 内容
        # 注意：这里根据你的 CSV 列名进行映射，如果训练脚本还需要其他参数，请在此添加
        config_data = {
            "id": row['id'],
            "n_blocks": int(row['n_blocks']),
            "op_codes": parse_list_string(row['op_codes']),
            "width_codes": parse_list_string(row['width_codes']),
            # 某些旧脚本可能需要 depth 或其他参数，这里尽量填充基础信息
            "created_at": time.time()
        }
        
        # 将 'test_acc_top1' 预设为 None
        config_data["test_acc_top1"] = None

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"【重建】已生成配置文件: {filepath}")
        return True
    except Exception as e:
        print(f"Error: 重建配置文件失败 {filename}: {e}")
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
        df = pd.read_csv(ALL_RESULTS_CSV)
        
        # 确保 ID 是字符串类型以便匹配
        run_id = str(run_id)
        df['id'] = df['id'].astype(str)
        
        mask = df['id'] == run_id
        if mask.any():
            df.loc[mask, "test_acc_top1"] = acc
            df.to_csv(ALL_RESULTS_CSV, index=False)
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
    backup_csv(ALL_RESULTS_CSV)
    
    # 2. 读取 CSV 并筛选未训练的模型
    df = pd.read_csv(ALL_RESULTS_CSV)
    
    # 将 test_acc_top1 转为 numeric，非数字转为 NaN
    df['test_acc_top1'] = pd.to_numeric(df['test_acc_top1'], errors='coerce')
    
    # 筛选 NaN 的行
    missing_df = df[df['test_acc_top1'].isna()]
    
    total_missing = len(missing_df)
    print(f"\n========================================")
    print(f"检测到总任务数: {len(df)}")
    print(f"需要补跑的模型数: {total_missing}")
    print(f"========================================\n")
    
    if total_missing == 0:
        print("所有模型均已有 Accuracy 数据，无需训练。")
        return

    # 3. 遍历训练
    for idx, (index, row) in enumerate(missing_df.iterrows()):
        run_id = str(row['id'])
        filename = row['filename']
        
        print(f"[{idx+1}/{total_missing}] 处理 ID: {run_id} (File: {filename})")
        
        # 3.1 确保 JSON 存在 (如果不存在则用 op_codes 重建)
        if not ensure_json_config(row):
            print(f"Skipping {run_id} due to config error.")
            continue
            
        # 3.2 构造日志路径和命令
        log_name = f"train_{run_id}.log"
        log_path = os.path.join(LOG_DIR, log_name)
        
        # 这里的命令调用 run_train.sh，传入 run_id 和 log_name
        # 假设 run_train.sh 会自动根据 run_id 去 gridsearch_output 下找 json
        cmd = ["bash", RUN_TRAIN_SCRIPT, f"--run_id={run_id}", f"--log_name={log_name}"]
        
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
                    json_path = os.path.join(OUTPUT_DIR, filename)
                    with open(json_path, 'r') as f:
                        jdata = json.load(f)
                    jdata['test_acc_top1'] = acc
                    with open(json_path, 'w') as f:
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
