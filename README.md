# Echo Lego Project

这是一个基于Python的项目，主要包含以下文件：

- `step2_0818_original.py`: 原始实现文件
- `step2_beamsearch.py`: 实现beam search算法的文件
- `step2_0818.py`: 2018年8月18日版本
- `step2_0819.py`: 2018年8月19日版本
- `step2_0819_training.py`: 训练相关实现
- `validation.py`: 验证功能实现
- `run.sh`: 运行脚本
- `update_arch_stats.py`: 计算网络结构的统计信息（Params, FLOPs, Acts）并写入 JSON 文件

## 运行环境

- Python 3.x
- 依赖库：`torch`, `fvcore`, `pandas` 等（建议使用 uv 管理环境）

## 使用方法

执行以下命令运行项目：

```bash
./run.sh
```

### 更新网络结构统计信息

`update_arch_stats.py` 脚本用于计算网络结构的 Params、FLOPs 和 Activations，并将这些信息回写到 JSON 文件中。

**更新单个 JSON 文件：**
```bash
uv run python update_arch_stats.py --arch_path ./gridsearch_output/search_nblk5_pop48_gen300__global_ea.json
```

**批量更新目录下的所有 JSON 文件：**
```bash
uv run python update_arch_stats.py --arch_path ./gridsearch_output --recursive
```
