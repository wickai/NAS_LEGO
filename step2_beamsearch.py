#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于SWAP分数的逐步网络构建实验（Beam Search 版）

改动要点：
1) 引入 Beam Search（束搜索）：每步保留K条最优部分架构并继续扩展；
2) 特征缓存：每个beam缓存“base+已选blocks”后的特征（按batch缓存），候选只在其上评估；
3) 仅评估“新增block”的SWAP，保持与原始方案一致；
4) 支持长度归一化排序，避免偏好更长或更短；
5) 可选二阶段重评估（final_batches + rescore_topL）；
6) 提供 materialize 方法，随时把Beam状态还原为可前向的 StackedNet。
"""

from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import pickle
from typing import List, Tuple, Dict, Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# 复用你的工具与组件
from test import (
    set_seed,
    BaseSwap1,
    prepare_batches,
    mbconv_block,
    SEBlock,
)

# -----------------------------
# 基本模块封装（与原版一致）
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
    """可逐步堆叠的网络（用于最终落盘/复现）"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.base = BaseSwap1()
        self.blocks: nn.ModuleList = nn.ModuleList()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.base(dummy)
            assert out.ndim == 4
            self.current_channels = int(out.shape[1])

        self.head = None
        self._update_head()

    def _update_head(self):
        self.head = nn.Sequential(
            nn.Conv2d(self.current_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.num_classes),
        )

    def add_block(self, block: StepBlock):
        if block.in_channels != self.current_channels:
            raise ValueError(f"in_channels mismatch: {block.in_channels} vs {self.current_channels}")
        self.blocks.append(block)
        self.current_channels = block.out_channels
        self._update_head()

    def forward(self, x):
        x = self.base(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# -----------------------------
# 规格与实例化
# -----------------------------
def generate_candidate_specs(
    n: int,
    in_channels: int,
) -> List[Dict[str, Any]]:
    """生成 MobileNetV2 风格的候选块规格列表"""
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
    return StepBlock(
        in_channels=spec["in_channels"],
        out_channels=spec["out_channels"],
        stride=spec["stride"],
        k=spec["k"],
        exp=spec["exp"],
        se=spec["se"],
    )


# -----------------------------
# SWAP 评估（仅对“新增block”）
# -----------------------------
class SampleWiseActivationPatterns:
    """高效的SWAP评估实现"""
    def __init__(self, device):
        self.device = device
        self.activations = None

    @torch.no_grad()
    def collect_activations(self, feats):
        self.activations = feats.sign().to(self.device)

    @torch.no_grad()
    def calc_swap(self):
        if self.activations is None:
            return 0
        self.activations = self.activations.t()  # => (features, N)
        unique_patterns = torch.unique(self.activations, dim=0).size(0)
        return unique_patterns


class BlockSWAP:
    """评估单个Block的SWAP"""
    def __init__(self, device):
        self.device = device
        self.inter_feats = []
        self.swap_evaluator = SampleWiseActivationPatterns(device)

    def _hook_fn(self, module, inp, out):
        feats = out.detach().reshape(out.size(0), -1)
        self.inter_feats.append(feats)

    def _clear_hooks(self, hooks):
        for h in hooks:
            h.remove()
        hooks.clear()

    @torch.no_grad()
    def evaluate_on_feats(self, block: nn.Module, x_feats: List[torch.Tensor]) -> float:
        """在缓存的特征列表上评估 block 的SWAP；返回平均分"""
        hooks = []
        for module in block.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                h = module.register_forward_hook(self._hook_fn)
                hooks.append(h)

        scores = []
        for x in x_feats:
            self.inter_feats = []
            y = block(x)  # 仅前向这个block
            if len(self.inter_feats) == 0:
                scores.append(0.0)
                continue
            block_feats = torch.cat(self.inter_feats, dim=1)  # (N, D)
            self.swap_evaluator.collect_activations(block_feats)
            scores.append(float(self.swap_evaluator.calc_swap()))

        self._clear_hooks(hooks)
        self.inter_feats = []
        return float(np.mean(scores)) if len(scores) > 0 else 0.0


def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.reset_parameters()


# -----------------------------
# Beam Search 结构体
# -----------------------------
class BeamState:
    """表示一条部分架构（blocks 序列）的状态"""
    def __init__(self, current_channels: int, blocks: List[StepBlock], x_feats: List[torch.Tensor], specs: List[Dict[str, Any]], score_sum: float):
        self.current_channels = current_channels
        self.blocks = blocks                    # 已选中的实际模块（含参数，用于继续前向）
        self.x_feats = x_feats                  # 缓存：每个batch在当前blocks之后的特征
        self.specs = specs                      # 保存规格（用于落盘/复现）
        self.score_sum = float(score_sum)       # 累计分
        self.length = len(blocks)

    def normalized_score(self, beta: float) -> float:
        L = max(1, self.length)
        return self.score_sum / (L ** beta)

    def clone_for_append(self, new_block: StepBlock, new_spec: Dict[str, Any], new_x_feats: List[torch.Tensor], delta_score: float):
        return BeamState(
            current_channels=new_block.out_channels,
            blocks=self.blocks + [new_block],
            x_feats=new_x_feats,
            specs=self.specs + [new_spec],
            score_sum=self.score_sum + float(delta_score)
        )


def materialize_stackednet_from_state(state: BeamState, num_classes: int) -> StackedNet:
    """把 BeamState 复原成可前向的 StackedNet（复制权重以便保存 state_dict）"""
    net = StackedNet(num_classes=num_classes)
    # 覆盖 base 的权重（使用新实例的 base 即可；搜索阶段未训练）
    # 把 blocks 复制到 net
    for b in state.blocks:
        # 以 spec 为蓝本重新实例化，再复制参数
        sb = StepBlock(b.in_channels, b.out_channels, b.stride, b.k, b.exp, b.se)
        sb.load_state_dict(deepcopy(b.state_dict()))
        net.add_block(sb)
    return net


# -----------------------------
# Beam Search 主流程
# -----------------------------
def run_experiment_beam(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    batches = prepare_batches(args.data_root, args.batches, args.batch_size, device)

    # 共享 base：先把每个 batch 过一遍 base，得到初始特征（所有beam共用）
    base = BaseSwap1().to(device).eval()
    with torch.no_grad():
        base_feats = [base(x.to(device)) for x in batches]
    current_channels = int(base_feats[0].shape[1])

    # 初始 beam（空 blocks）
    beams: List[BeamState] = [BeamState(current_channels, [], base_feats, [], 0.0)]
    history_all_steps = []

    swap_eval = BlockSWAP(device)

    for step in range(args.max_steps):
        step_candidates = []  # 汇集所有 beam 的扩展候选（新的 BeamState）
        print(f"\n[Beam Step {step+1}/{args.max_steps}] | beam_size={len(beams)} | current_channels in beams ~{sorted(set(b.current_channels for b in beams))}")

        # 对每条 beam 扩展
        for b_idx, beam in enumerate(beams):
            # 生成 M 个候选规格（每条 beam 各自基于其 current_channels 生成）
            cand_specs = generate_candidate_specs(args.expand_per_beam, beam.current_channels)

            # 评估每个候选（只新增block），并更新 x_feats 用于后续继续扩展
            for spec in cand_specs:
                # in_channels 一定要匹配
                spec["in_channels"] = beam.current_channels
                block = instantiate_block(spec).to(device).eval()
                # 只初始化一次
                for m in block.modules():
                    kaiming_init(m)

                # 评估新增块 SWAP（在该beam缓存的特征上）
                score = swap_eval.evaluate_on_feats(block, beam.x_feats)

                # 计算新的 x_feats（将缓存特征过新增块一次，作为新beam的缓存）
                with torch.no_grad():
                    new_x_feats = [block(x) for x in beam.x_feats]

                new_beam = beam.clone_for_append(block, deepcopy(spec), new_x_feats, score)
                step_candidates.append(new_beam)

        # 选前 K 个（按“长度归一化分数”排序）
        step_candidates.sort(key=lambda s: s.normalized_score(args.len_norm_beta), reverse=True)
        beams = step_candidates[: args.keep_per_step]

        # 记录本步 topK 的摘要
        topk_summary = [
            {
                "rank": i + 1,
                "norm_score": beams[i].normalized_score(args.len_norm_beta),
                "score_sum": beams[i].score_sum,
                "length": beams[i].length,
                "last_spec": beams[i].specs[-1] if beams[i].specs else None,
                "current_channels": beams[i].current_channels,
            }
            for i in range(len(beams))
        ]
        history_all_steps.append({"step": step + 1, "topk": topk_summary})
        print(f"[Step {step+1}] Top-1 norm_score={topk_summary[0]['norm_score']:.4f} | score_sum={topk_summary[0]['score_sum']:.1f} | len={topk_summary[0]['length']}")

    # ---- 可选：对最终 Top-L 做多 batch 重评估以稳定结果 ----
    final_pool = beams[: args.rescore_topL]
    if args.final_batches > args.batches:
        print(f"\n[Final Re-Scoring] Using {args.final_batches} batches (prev {args.batches}) on Top-{args.rescore_topL}")
        # 重新准备更多 batch
        final_batches = prepare_batches(args.data_root, args.final_batches, args.batch_size, device)
        # 为每条beam重建 base->blocks 的 x_feats
        with torch.no_grad():
            final_base_feats = [base(x.to(device)) for x in final_batches]

        rescored = []
        for beam in final_pool:
            # 逐块把 final_base_feats 过一遍，累加每块 SWAP 得分（保持“只新增块计分”的一致性）
            x_feats = final_base_feats
            total = 0.0
            for blk in beam.blocks:
                # 复制一个块并加载权重
                bcopy = StepBlock(blk.in_channels, blk.out_channels, blk.stride, blk.k, blk.exp, blk.se).to(device).eval()
                bcopy.load_state_dict(deepcopy(blk.state_dict()))
                sc = BlockSWAP(device).evaluate_on_feats(bcopy, x_feats)
                total += sc
                with torch.no_grad():
                    x_feats = [bcopy(x) for x in x_feats]
            # 生成一个新副本，仅更新 score_sum（不动 blocks 等）
            b2 = BeamState(beam.current_channels, beam.blocks, beam.x_feats, beam.specs, total)
            rescored.append(b2)

        # 以重评后的得分替换
        final_pool = rescored
        final_pool.sort(key=lambda s: s.normalized_score(args.len_norm_beta), reverse=True)

    best = final_pool[0]
    print("\n搜索完成！")
    print(f"最佳（按长度归一化）: norm_score={best.normalized_score(args.len_norm_beta):.4f} | score_sum={best.score_sum:.1f} | 长度={best.length}")

    # 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "args": vars(args),
        "history": history_all_steps,
        "best_norm_score": best.normalized_score(args.len_norm_beta),
        "best_score_sum": best.score_sum,
        "best_length": best.length,
        "best_specs": best.specs,
    }
    with open(f"{args.output_dir}/beam_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # 物化网络并保存 checkpoint（便于后续训练/评估）
    net_mat = materialize_stackednet_from_state(best, num_classes=args.num_classes).to(device).eval()
    torch.save({"state_dict": net_mat.state_dict(), "best_specs": best.specs, "args": vars(args)},
               f"{args.output_dir}/beam_best_checkpoint.pt")

    print(f"已保存：{args.output_dir}/beam_results.pkl 与 beam_best_checkpoint.pt")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser("基于Block SWAP的Beam Search网络构建")
    # 搜索规模
    parser.add_argument("--max_steps", type=int, default=5, help="最大堆叠步数")
    parser.add_argument("--beam_size", type=int, default=4, help="每步保留的beam数量（初始等于keep_per_step）")
    parser.add_argument("--expand_per_beam", type=int, default=64, help="每条beam在每步扩展的候选数 M")
    parser.add_argument("--keep_per_step", type=int, default=4, help="每步筛回的beam数量 K（通常=beam_size）")
    parser.add_argument("--len_norm_beta", type=float, default=0.7, help="长度归一化指数 beta，越大越惩罚更长的序列")

    # 数据与batch
    parser.add_argument("--batches", type=int, default=1, help="用于SWAP计算的batch数（Fast阶段）")
    parser.add_argument("--final_batches", type=int, default=1, help="Final重评估使用的batch数（>=batches时启用）")
    parser.add_argument("--rescore_topL", type=int, default=3, help="最终Top-L做重评估")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_root", type=str, default="./data")

    # 其它
    parser.add_argument("--output_dir", type=str, default="./results_beam")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=10)

    args = parser.parse_args()

    # 一些合理性约束
    args.keep_per_step = max(1, args.keep_per_step)
    args.beam_size = max(1, args.beam_size)
    if args.keep_per_step != args.beam_size:
        # 允许不同，但提醒一下
        print(f"[WARN] keep_per_step({args.keep_per_step}) != beam_size({args.beam_size}); 实际保留以 keep_per_step 为准。")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tic = time.time()
    run_experiment_beam(args)
    print(f"总用时: {time.time() - tic:.1f} 秒")


if __name__ == "__main__":
    main()
