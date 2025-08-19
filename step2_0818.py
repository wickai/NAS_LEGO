#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于SWAP分数的逐步网络构建实验（立刻见效版改进）

已实现的改进（对应你的“立刻见效”清单）：
1) 分类头自适应 current_channels（去掉固定 256 的 1x1 卷积）
2) 缓存“新增 block 的输入”并复用，支持 pin_memory + non_blocking
3) 轻度增强生成多视角 batch（随机水平翻转），提升 SWAP 稳定性（由命令行控制）
4) 断点恢复与可复现：支持 --resume_from，逐步保存 candidates 与 best 的 JSON

依赖：test.py 需提供
- set_seed, BaseSwap1, prepare_batches, mbconv_block, SEBlock
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import pickle
from typing import List, Dict, Any, Tuple, Optional

import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# 从你的工具文件导入
from test import (
    set_seed,
    BaseSwap1,
    prepare_batches,
    mbconv_block,
    SEBlock,
)


# -----------------------------
# 基本模块封装
# -----------------------------
class StepBlock(nn.Module):
    """可堆叠的网络块（MBConv 风格，可选 SE）"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        k: int = 3,
        exp: int = 3,
        se: bool = False,
    ):
        super().__init__()
        # mbconv_block: (in_c, out_c, k, exp, stride, se)
        self.block = mbconv_block(in_channels, out_channels, k, exp, stride, se)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.k = k
        self.exp = exp
        self.se = se

    def forward(self, x):
        return self.block(x)


class StackedNet(nn.Module):
    """可逐步堆叠的网络"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.base = BaseSwap1()
        self.blocks: nn.ModuleList = nn.ModuleList()

        # 探测 BaseSwap1 的输出通道数（假设 CIFAR-10 尺寸 3x32x32）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.base(dummy)
            assert out.ndim == 4, "BaseSwap1 的输出必须是 NCHW 格式的特征图"
            self.current_channels = int(out.shape[1])

        self.head = None
        self._update_head()

    def _update_head(self):
        """分类头自适应 current_channels（去掉固定 256 的 1x1 投影）"""
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.current_channels, self.num_classes),
        )

    def add_block(self, block: StepBlock):
        """添加新的块并更新网络结构"""
        if block.in_channels != self.current_channels:
            raise ValueError(
                f"in_channels 不匹配：候选 {block.in_channels}，当前 {self.current_channels}。"
                f"请按当前通道重新生成候选块。"
            )
        self.blocks.append(block)
        self.current_channels = block.out_channels
        self._update_head()

    def forward(self, x):
        x = self.base(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    @torch.no_grad()
    def forward_to_current(self, x):
        """前向到当前堆叠末端（得到新增 block 的输入特征）"""
        x = self.base(x)
        for block in self.blocks:
            x = block(x)
        return x


# -----------------------------
# 候选规格生成与实例化
# -----------------------------
def generate_candidate_specs(
    n: int,
    in_channels: int,
) -> List[Dict[str, Any]]:
    """
    生成 MobileNetV2 风格候选规格（(t, c, s) 稍做扰动），in_channels 由当前网络确定。
    """
    mobilenet_configs = [
        (1, 16, 1),
        (6, 24, 2),
        (6, 32, 2),
        (6, 64, 2),
        (6, 96, 1),
        (6, 160, 2),
        (6, 320, 1),
    ]

    specs: List[Dict[str, Any]] = []
    for _ in range(n):
        t, c, s = random.choice(mobilenet_configs)

        # 轻微随机扰动
        t = max(1, int(round(t * random.choice([1.0, 1.5, 0.75]))))
        c = max(16, int(round(c * random.uniform(0.75, 1.25))))
        k = random.choice([3, 5])
        se = (random.random() < 0.3)

        spec = dict(
            in_channels=in_channels,
            out_channels=c,
            stride=s,
            k=k,
            exp=t,
            se=se,
        )
        specs.append(spec)
    return specs


def instantiate_block(spec: Dict[str, Any]) -> StepBlock:
    """由规格实例化 StepBlock 模块"""
    return StepBlock(
        in_channels=spec["in_channels"],
        out_channels=spec["out_channels"],
        stride=spec["stride"],
        k=spec["k"],
        exp=spec["exp"],
        se=spec["se"],
    )


# -----------------------------
# 新的 SWAP 评估相关
# -----------------------------
class SampleWiseActivationPatterns:
    """高效的 SWAP 评估实现：按样本激活符号模式去重计数"""
    def __init__(self, device: torch.device):
        self.device = device
        self._collected: List[torch.Tensor] = []

    @torch.no_grad()
    def collect(self, feats: torch.Tensor):
        # feats: (N, D)
        self._collected.append(feats.sign().to(self.device))

    @torch.no_grad()
    def calc(self) -> int:
        if not self._collected:
            return 0
        acts = torch.cat(self._collected, dim=0)  # (sum_N, D)
        uniq = torch.unique(acts.t().contiguous(), dim=0).size(0)
        return int(uniq)

    def reset(self):
        self._collected.clear()


class BlockSWAP:
    """仅为“单个候选 Block”计算 SWAP"""
    def __init__(self, device: torch.device):
        self.device = device
        self._inter_feats: List[torch.Tensor] = []
        self._evaluator = SampleWiseActivationPatterns(device)

    def _hook_fn(self, module, inp, out):
        feats = out.detach().reshape(out.size(0), -1)
        self._inter_feats.append(feats)

    def _register_hooks(self, block: nn.Module):
        hooks = []
        for m in block.modules():
            if isinstance(m, (nn.ReLU, nn.ReLU6)):
                hooks.append(m.register_forward_hook(self._hook_fn))
        return hooks

    @staticmethod
    def _init_once(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_parameters()

    @torch.no_grad()
    def evaluate(self, block: nn.Module, inputs: List[torch.Tensor]) -> float:
        """
        对单个 block 做一次初始化，并在多批次 inputs 上前向，收集激活用于 SWAP 计算。
        inputs: 若干批次、已经是“新 block 的输入特征”（NCHW）
        """
        block.to(self.device)
        block.apply(self._init_once)
        block.eval()

        hooks = self._register_hooks(block)
        self._inter_feats.clear()
        self._evaluator.reset()

        try:
            for x in inputs:
                # non_blocking + pin_memory 友好：若在 CPU 则 pin 后再搬到 GPU
                if x.device.type == "cpu":
                    try:
                        x = x.pin_memory()
                    except Exception:
                        pass
                x = x.to(self.device, non_blocking=True)
                out = block(x)  # hooks 收集到 ReLU 的激活
                # 若无激活（极少见，或未用 ReLU），用 block 输出替代
                if len(self._inter_feats) == 0:
                    feats = out.detach().reshape(out.size(0), -1)
                    self._inter_feats.append(feats)
                feats_batch = (
                    torch.cat(self._inter_feats, dim=1)
                    if len(self._inter_feats) > 1
                    else self._inter_feats[0]
                )
                self._evaluator.collect(feats_batch)
                self._inter_feats.clear()

            score = float(self._evaluator.calc())
        finally:
            for h in hooks:
                h.remove()

        return score


# -----------------------------
# 数据批次：增强、多视角与缓存
# -----------------------------
@torch.no_grad()
def augment_batches(
    batches: List[torch.Tensor],
    times: int = 1,
    hflip_prob: float = 0.5,
) -> List[torch.Tensor]:
    """
    对原始图像 batch 做轻度增强，生成多视角样本。
    - 只做随机水平翻转（与 CIFAR/Imagenet 常见增强一致）
    - times=1 表示不复制；>1 表示重复 times 轮增强
    """
    if times <= 1:
        return batches

    outs: List[torch.Tensor] = []
    for _ in range(times):
        for x in batches:
            x_aug = x.clone()
            # 若张量在 GPU，直接在 GPU 上操作；在 CPU 则保持 CPU
            if torch.rand(1).item() < hflip_prob:
                x_aug = torch.flip(x_aug, dims=[-1])  # 水平翻转 W 维
            outs.append(x_aug)
    return outs


@torch.no_grad()
def make_block_inputs(
    net: StackedNet,
    raw_batches: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    让原始输入（图像）前向到“当前堆叠末端”，得到“新增 block 的输入特征”。
    - 自动处理 CPU -> GPU 的 non_blocking 传输与 pin_memory
    """
    net.eval().to(device)
    feats_batches: List[torch.Tensor] = []
    for x in raw_batches:
        if x.device.type == "cpu":
            try:
                x = x.pin_memory()
            except Exception:
                pass
        x = x.to(device, non_blocking=True)
        feats = net.forward_to_current(x)
        feats_batches.append(feats.detach())
    return feats_batches


@torch.no_grad()
def evaluate_block_swap(
    net: StackedNet,
    new_block: StepBlock,
    block_inputs: List[torch.Tensor],
    device: torch.device,
) -> float:
    """评估新 block 的 SWAP 分数（多批次）"""
    swapper = BlockSWAP(device)
    return swapper.evaluate(new_block, block_inputs)


# -----------------------------
# 实验主流程（含断点恢复）
# -----------------------------
def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("非法的 checkpoint 格式")
    return ckpt


def run_experiment(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 原始图像批次（由 prepare_batches 提供；可能在 CPU 或 GPU）
    raw_batches = prepare_batches(args.data_root, args.batches, args.batch_size, device)

    # 轻度增强（多视角）
    raw_batches_aug = augment_batches(
        raw_batches, times=args.augment_times, hflip_prob=args.hflip_prob
    )

    # 初始化/恢复网络与历史
    net = StackedNet(num_classes=args.num_classes).to(device).eval()
    history: List[Dict[str, Any]] = []
    start_step = 0

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path)
            net.load_state_dict(ckpt["state_dict"], strict=True)
            history = ckpt.get("history", [])
            start_step = int(ckpt.get("step", 0))
            print(f"[恢复] 从 step={start_step} 继续。历史长度={len(history)}")
        else:
            print(f"[警告] resume 文件不存在：{ckpt_path}，将从头开始。")

    # 主循环
    for step in range(start_step, args.max_steps):
        human_step = step + 1
        print(f"\n步骤 {human_step}/{args.max_steps} | 当前通道 = {net.current_channels}")

        # 先计算一次“新增 block 的输入特征”，避免为每个候选都重复跑 backbone
        block_inputs = make_block_inputs(net, raw_batches_aug, device)

        # 生成候选规格
        candidate_specs = generate_candidate_specs(args.n_candidates, net.current_channels)

        # 保存本步候选（可复现）
        save_json(
            candidate_specs,
            Path(args.output_dir) / f"step_{human_step:02d}_candidates.json",
        )

        best_score = float("-inf")
        best_spec: Optional[Dict[str, Any]] = None

        # 评估所有候选
        for spec in tqdm(candidate_specs, desc=f"评估候选块@step{human_step}"):
            candidate_block = instantiate_block(spec)
            score = evaluate_block_swap(net, candidate_block, block_inputs, device)
            if score > best_score:
                best_score = score
                best_spec = spec

        assert best_spec is not None, "未能产生有效候选规格。"
        best_block = instantiate_block(best_spec).to(device)
        net.add_block(best_block)  # 更新通道与 head

        # 记录
        step_info = {
            "step": human_step,
            "score": float(best_score),
            "chosen_spec": best_spec,
            "current_channels_after": net.current_channels,
        }
        history.append(step_info)

        print(f"步骤 {human_step} 最佳 Block SWAP: {best_score:.4f} | 规格: {best_spec}")

        # 保存 best 的 JSON
        save_json(
            step_info,
            Path(args.output_dir) / f"step_{human_step:02d}_best.json",
        )

        # 保存 checkpoint
        checkpoint = {
            "step": human_step,
            "state_dict": net.state_dict(),
            "history": history,
            "current_channels": net.current_channels,
        }
        torch.save(checkpoint, f"{args.output_dir}/checkpoint_step_{human_step}.pt")

    # 保存最终结果
    final_results = {
        "history": history,
        "final_score": history[-1]["score"] if history else float("nan"),
    }
    with open(f"{args.output_dir}/final_results.pkl", "wb") as f:
        pickle.dump(final_results, f)

    print("\n实验完成!")
    if history:
        print(f"最终 Block SWAP 分数: {history[-1]['score']:.4f}")


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    parser = argparse.ArgumentParser("基于 Block SWAP 的逐步网络构建实验（改进版）")
    parser.add_argument("--n_candidates", type=int, default=1000, help="每一步的候选块数量")
    parser.add_argument("--max_steps", type=int, default=5, help="最大堆叠步数")
    parser.add_argument("--batches", type=int, default=2, help="用于 SWAP 计算的原始批次数（>=1）")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_block_swap_immediate")
    parser.add_argument("--cuda", action="store_true", help="使用 CUDA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=10, help="分类类别数")

    # 轻度增强控制
    parser.add_argument("--augment_times", type=int, default=1, help="同一批数据复制增强的次数（1=不增强）")
    parser.add_argument("--hflip_prob", type=float, default=0.5, help="随机水平翻转概率")

    # 断点恢复
    parser.add_argument("--resume_from", type=str, default="", help="从某个 checkpoint 路径恢复")
    return parser


def main():
    args = build_parser().parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)

    # 运行实验
    tic = time.time()
    run_experiment(args)
    print(f"总用时: {time.time() - tic:.1f} 秒")


if __name__ == "__main__":
    main()
