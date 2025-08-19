#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于SWAP分数的逐步网络构建实验（强化版 / Two-Stage, Multi-Objective, Cached Features + fixes）

改进要点：
1) 两阶段评估：Fast（少batch, 少init）→ Final（多batch, 多init）；
2) 多目标打分：score = alpha*SWAP_norm - beta*log(FLOPs+1) - gamma*Params(M) - delta*DiversityPenalty；
3) 特征缓存：每个搜索步只计算一次 base+已选block 的特征，候选直接在其上评估；
4) 多样性：避免与上一步规格过于相似的候选；
5) 可选 FLOPs(需 thop)；无 thop 时自动降级仅用 Params；
6) Fast 阶段对 SWAP_norm 做 Z-score 标准化，稳定排序；
7) Final 阶段可选“快训重排”（只训新块+head，极少步数）帮助最终精度对齐；
8) 全局下采样（stride=2）约束，避免 CIFAR-10 32x32 过早失真。

依赖：复用 test.py 内的 set_seed / BaseSwap1 / prepare_batches / mbconv_block / SEBlock
"""

from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import pickle
from typing import List, Tuple, Dict, Any
from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# 可选 FLOPs 依赖
try:
    from thop import profile as thop_profile  # pip install thop
    HAS_THOP = True
except Exception:
    HAS_THOP = False

# 复用你的工具与组件（确保 test.py 在同目录或 PYTHONPATH 中）
from test import (
    set_seed,
    BaseSwap1,
    prepare_batches,
    mbconv_block,
    SEBlock,
)

# -----------------------------
# 基本模块
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
    """可逐步堆叠的网络"""
    def __init__(self, num_classes: int = 10, base_downsamples: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.base = BaseSwap1()
        self.blocks: nn.ModuleList = nn.ModuleList()

        # 探测 BaseSwap1 的输出通道数（假设 CIFAR-10 尺寸 3x32x32）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.base(dummy)
            assert out.ndim == 4, "BaseSwap1 的输出必须是 NCHW 特征图"
            self.current_channels = int(out.shape[1])
            print(f"BaseSwap1 输出通道数: {self.current_channels}")

        self.head = None
        self._update_head()

        # 全局下采样计数（若 Base 包含下采样，可把 base_downsamples 设为对应值）
        self.total_downsamples = int(base_downsamples)

    def _update_head(self):
        """更新分类头以匹配当前通道数"""
        self.head = nn.Sequential(
            nn.Conv2d(self.current_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.num_classes),
        )

    def add_block(self, block: StepBlock):
        """添加新的块并更新网络结构"""
        if block.in_channels != self.current_channels:
            raise ValueError(
                f"待添加块的 in_channels={block.in_channels} 与当前网络通道 "
                f"{self.current_channels} 不匹配。请按当前通道重新生成候选块。"
            )
        self.blocks.append(block)
        if getattr(block, "stride", 1) == 2:
            self.total_downsamples += 1
        self.current_channels = block.out_channels
        self._update_head()

    def forward(self, x):
        x = self.base(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# -----------------------------
# 候选规格生成与实例化
# -----------------------------
def generate_candidate_specs(
    n: int,
    in_channels: int,
    k_choices=(3, 5),
    exp_choices=(1, 3, 4, 6),
    out_scale=(0.75, 1.25),
    se_prob=0.3,
) -> List[Dict[str, Any]]:
    """生成 MobileNetV2 风格的候选块规格列表（可控搜索空间）"""
    ref_out_channels = [16, 24, 32, 40, 64, 80, 96, 112, 160, 192, 224, 320]

    specs: List[Dict[str, Any]] = []
    for _ in range(n):
        c = max(16, int(round(random.choice(ref_out_channels) * random.uniform(*out_scale))))
        k = random.choice(k_choices)
        exp = random.choice(exp_choices)
        stride = random.choice([1, 2])  # 原始候选，之后会被全局约束所修正
        se = (random.random() < se_prob)

        spec = dict(
            in_channels=in_channels,
            out_channels=c,
            stride=stride,
            k=k,
            exp=exp,
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
# SWAP 评估（仅针对新增 block）
# -----------------------------
class SampleWiseActivationPatterns:
    """SWAP 计算器：按符号二值化后的激活模式去重计数"""
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
    """用于评估单个Block的SWAP以及特征维度（用于归一化）"""
    def __init__(self, device):
        self.device = device
        self.inter_feats = []
        self.swap_evaluator = SampleWiseActivationPatterns(device)

    def evaluate(self, block, inputs):
        hooks = []
        for module in block.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                h = module.register_forward_hook(self._hook_fn)
                hooks.append(h)

        self.inter_feats = []
        with torch.no_grad():
            _ = block(inputs)

        if len(self.inter_feats) == 0:
            self._clear_hooks(hooks)
            return 0.0, 0  # swap, feat_dim

        block_feats = torch.cat(self.inter_feats, dim=1)  # (N, F)
        feat_dim = int(block_feats.shape[1])
        self.swap_evaluator.collect_activations(block_feats)
        swap_score = float(self.swap_evaluator.calc_swap())

        self._clear_hooks(hooks)
        self.inter_feats = []
        return swap_score, feat_dim

    def _hook_fn(self, module, inp, out):
        feats = out.detach().reshape(out.size(0), -1)
        self.inter_feats.append(feats)

    def _clear_hooks(self, hooks):
        for h in hooks:
            h.remove()
        hooks.clear()


def init_module_(m: nn.Module):
    """一次性合理初始化（Conv/BN/Linear）"""
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.reset_parameters()


@torch.no_grad()
def evaluate_block_swap_multi(
    new_block: StepBlock,
    cached_feats: List[torch.Tensor],  # 预计算的若干 batch 的输入特征
    inits: int,
    device,
) -> Tuple[float, float, float, int]:
    """
    对单个候选进行多 init、多 batch 的 SWAP 评估。
    返回：avg_swap, avg_swap_norm, avg_swap_per_sample, feat_dim
      - swap_norm = swap / feat_dim
      - swap_per_sample = swap / batch_size
    """
    block_swap = BlockSWAP(device)
    swaps, swaps_norm, swaps_per_sample = [], [], []
    feat_dim_ref = None

    for _ in range(inits):
        init_module_(new_block)
        # 多个 batch 汇总
        swap_sum, norm_sum, pps_sum, count = 0.0, 0.0, 0.0, 0
        for x in cached_feats:
            s, feat_dim = block_swap.evaluate(new_block, x)
            if feat_dim == 0:
                continue
            swap_sum += s
            norm_sum += (s / max(1, feat_dim))
            pps_sum += (s / max(1, x.shape[0]))
            count += 1
            if feat_dim_ref is None and feat_dim > 0:
                feat_dim_ref = feat_dim
        if count > 0:
            swaps.append(swap_sum / count)
            swaps_norm.append(norm_sum / count)
            swaps_per_sample.append(pps_sum / count)

    if len(swaps) == 0:
        return 0.0, 0.0, 0.0, (feat_dim_ref or 0)

    return (
        float(np.mean(swaps)),
        float(np.mean(swaps_norm)),
        float(np.mean(swaps_per_sample)),
        int(feat_dim_ref or 0),
    )


def try_profile_flops_params(block: nn.Module, input_feat: torch.Tensor) -> Tuple[float, float]:
    """
    估算单个 Block 的 FLOPs / Params。
    若无 thop，则仅返回 Params，FLOPs=0。
    """
    params_m = sum(p.numel() for p in block.parameters()) / 1e6
    if not HAS_THOP:
        return 0.0, params_m

    try:
        flops, _ = thop_profile(block, inputs=(input_feat,), verbose=False)
        flops_g = flops / 1e9  # 转成 G 次数便于 log 调权
        return flops_g, params_m
    except Exception:
        return 0.0, params_m


def diversity_penalty(spec: Dict[str, Any], prev_spec: Dict[str, Any] | None) -> float:
    """
    与上一层过于相似时施加微弱惩罚，促进结构多样性（可按需加权/扩展）。
    """
    if prev_spec is None:
        return 0.0
    p = 0.0
    # 同核大小/扩展率/步幅 → 小惩罚
    if spec["k"] == prev_spec["k"]:
        p += 0.2
    if spec["exp"] == prev_spec["exp"]:
        p += 0.2
    if spec["stride"] == prev_spec["stride"]:
        p += 0.1
    # 通道过近也惩罚（越接近惩罚越大）
    c1, c2 = spec["out_channels"], prev_spec["out_channels"]
    p += max(0.0, 0.5 - abs(c1 - c2) / max(16.0, c2))
    # 无/有 SE 一致性小惩罚
    if bool(spec["se"]) == bool(prev_spec["se"]):
        p += 0.1
    return p


# -----------------------------
# Final 阶段：可选“快训重排”
# -----------------------------
def quick_finetune_rank(
    net: StackedNet,
    block: StepBlock,
    cached_final_feats: List[torch.Tensor],
    steps: int = 120,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    只训练"新块 + head"，用当前 head 的软标签做极轻量蒸馏，返回一个越大越好的排名分。
    注意：我们不动 base 和已选旧块的参数，只让新块和 head 微调若干步。
    """
    # 确保当前网络在正确的设备上
    net = net.to(device)
    block = block.to(device)

    # 创建一个临时head以匹配新block的输出通道
    temp_head = nn.Sequential(
        nn.Conv2d(block.out_channels, 256, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, net.num_classes),
    ).to(device)

    # 冻结 base 与旧块
    for p in net.base.parameters():
        p.requires_grad_(False)
    for b in net.blocks:
        for p in b.parameters():
            p.requires_grad_(False)
    for p in net.head.parameters():
        p.requires_grad_(False)

    # 训练新 block 与 head
    for p in block.parameters():
        p.requires_grad_(True)
    for p in temp_head.parameters():
        p.requires_grad_(True)

    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.AdamW(list(block.parameters()) + list(temp_head.parameters()), lr=lr)

    block.train()
    temp_head.train()
    iters = 0
    running_loss = 0.0
    with torch.enable_grad():
        for f in cached_final_feats:
            f = f.detach()  # (N, C, H, W) features
            teacher_logits = net.head(f).detach()  # teacher 来自当前 head（不含新块）
            for _ in range(2):  # 每个 batch 重复几步
                out = block(f)
                logits = temp_head(out)  # 使用新head
                loss = criterion_kl(
                    torch.log_softmax(logits, dim=-1),
                    torch.softmax(teacher_logits, dim=-1),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += float(loss.item())
                iters += 1
                if iters >= steps:
                    break
            if iters >= steps:
                break

    # 返回 -loss 作为“越大越好”的加分项
    return - running_loss / max(1, iters)


# -----------------------------
# 实验主流程（两阶段）
# -----------------------------
def run_experiment(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    # 预加载数据 batch（尽量放到 CPU，再按需搬到 GPU）
    batches = prepare_batches(
        args.data_root,
        max(args.batches_fast, args.batches_final),
        args.batch_size,
        device
    )

    net = StackedNet(num_classes=args.num_classes, base_downsamples=args.base_downsamples).to(device)
    history: List[Dict[str, Any]] = []
    prev_chosen_spec: Dict[str, Any] | None = None

    for step in range(args.max_steps):
        print(f"\n步骤 {step + 1}/{args.max_steps} | 当前通道 = {net.current_channels} | 全局下采样数 = {net.total_downsamples}")

        # 1) 生成候选
        candidate_specs = generate_candidate_specs(
            args.n_candidates,
            net.current_channels,
            k_choices=tuple(args.k_choices),
            exp_choices=tuple(args.exp_choices),
            out_scale=tuple(args.out_scale),
            se_prob=args.se_prob,
        )

        # --- 全局下采样约束与本步 stride=2 配额控制 ---
        if net.total_downsamples >= args.max_total_downsamples:
            # 达上限：全部改为 stride=1
            candidate_specs = [{**s, "stride": 1} for s in candidate_specs]
        else:
            # 允许本步一小部分 stride=2（其余改为 stride=1）
            stride2_budget = max(1, int(len(candidate_specs) * args.stride2_ratio))
            kept = []
            for s in candidate_specs:
                if s["stride"] == 2 and stride2_budget > 0:
                    kept.append(s)
                    stride2_budget -= 1
                else:
                    kept.append({**s, "stride": 1})
            candidate_specs = kept

        # 2) 预计算 “已有网络”的输出特征（缓存），用于候选评估输入
        with torch.no_grad():
            cached_feats_fast: List[torch.Tensor] = []
            cached_feats_final: List[torch.Tensor] = []
            for i, x in enumerate(batches[: args.batches_final]):  # 至多到 final 所需
                x = x.to(device)
                feat = net.base(x)
                for blk in net.blocks:
                    feat = blk(feat)
                if i < args.batches_fast:
                    cached_feats_fast.append(feat.detach())
                cached_feats_final.append(feat.detach())

        # 3) Fast 阶段（快速筛选 Top-K）
        fast_raw: List[Tuple[float, Dict[str, Any], Dict[str, float]]] = []

        for spec in tqdm(candidate_specs, desc="Fast评估候选块"):
            # 规格与当前通道不匹配的直接跳过（理论上不会发生）
            if spec["in_channels"] != net.current_channels:
                continue

            block = instantiate_block(spec).to(device)
            # 估计 FLOPs / Params（用 fast 的第一批输入形状）
            if len(cached_feats_fast) > 0:
                flops_g, params_m = try_profile_flops_params(block, cached_feats_fast[0])
            else:
                flops_g, params_m = (0.0, sum(p.numel() for p in block.parameters())/1e6)

            # 多 init + 多 batch（快速阶段）
            avg_swap, avg_swap_norm, avg_swap_pps, feat_dim = evaluate_block_swap_multi(
                block, cached_feats_fast, inits=args.inits_fast, device=device
            )

            # 多样性惩罚
            div_pen = diversity_penalty(spec, prev_chosen_spec) * args.delta

            # 先把原始 meta 记下来，稍后统一做 Z-score 再打分
            meta = dict(
                swap=avg_swap,
                swap_norm=avg_swap_norm,
                swap_pps=avg_swap_pps,
                feat_dim=feat_dim,
                flops_g=flops_g,
                params_m=params_m,
                div_pen=div_pen,
                stage="fast",
            )
            fast_raw.append((0.0, spec, meta))

        # --- 对 SWAP_norm 做 Z-score 标准化，再合成 score ---
        fast_scores: List[Tuple[float, Dict[str, Any], Dict[str, float]]] = []
        if len(fast_raw) > 0:
            swaps = np.array([m["swap_norm"] for _, _, m in fast_raw], dtype=float)
            mu, sigma = float(swaps.mean()), float(swaps.std() + 1e-8)
            for (_, spec, meta), z in zip(fast_raw, ((swaps - mu) / sigma).tolist()):
                flops_g, params_m = meta["flops_g"], meta["params_m"]
                div_pen = meta["div_pen"]
                score = (
                    args.alpha * z
                    - args.beta * math.log1p(max(0.0, flops_g))
                    - args.gamma * max(0.0, params_m)
                    - div_pen
                )
                meta = {**meta, "swap_norm_z": z}
                fast_scores.append((score, spec, meta))

        # 选出 Top-K 进入 Final 阶段
        fast_scores.sort(key=lambda x: x[0], reverse=True)
        shortlist = fast_scores[: args.topk]

        # 4) Final 阶段（更严格的复评 + 可选快训重排）
        best_score = float("-inf")
        best_spec = None
        best_metrics = None

        for _, spec, fast_meta in tqdm(shortlist, desc="Final复评候选"):
            block = instantiate_block(spec).to(device)
            flops_g = fast_meta["flops_g"]
            params_m = fast_meta["params_m"]
            if flops_g == 0.0 and len(cached_feats_final) > 0:
                # 如果 fast 阶段没算成 FLOPs，这里再尝试一次
                f2, _ = try_profile_flops_params(block, cached_feats_final[0])
                if f2 > 0:
                    flops_g = f2

            avg_swap, avg_swap_norm, avg_swap_pps, feat_dim = evaluate_block_swap_multi(
                block, cached_feats_final, inits=args.inits_final, device=device
            )
            div_pen = diversity_penalty(spec, prev_chosen_spec) * args.delta

            score = (
                args.alpha * avg_swap_norm
                - args.beta * math.log1p(max(0.0, flops_g))
                - args.gamma * max(0.0, params_m)
                - div_pen
            )

            if args.enable_quick_rerank and len(cached_feats_final) > 0:
                # 直接使用当前网络和新block进行快速训练评估
                quick_rank = quick_finetune_rank(
                    net, block, cached_feats_final,
                    steps=args.quick_steps, lr=args.quick_lr, device=device
                )
                score = score + args.quick_weight * quick_rank

            if score > best_score:
                best_score = score
                best_spec = spec
                best_metrics = dict(
                    swap=avg_swap,
                    swap_norm=avg_swap_norm,
                    swap_pps=avg_swap_pps,
                    feat_dim=feat_dim,
                    flops_g=flops_g,
                    params_m=params_m,
                    div_pen=div_pen,
                    stage="final",
                )

        # 5) 添加最佳 block
        assert best_spec is not None
        best_block = instantiate_block(best_spec).to(device)
        net.add_block(best_block)
        prev_chosen_spec = best_spec

        # 6) 记录与保存
        step_info = {
            "step": step + 1,
            "score": float(best_score),
            "chosen_spec": best_spec,
            "metrics": best_metrics,
            "current_channels_after": net.current_channels,
            "total_downsamples": net.total_downsamples,
        }
        history.append(step_info)

        print(f"步骤 {step + 1} | 最佳综合分: {best_score:.5f}")
        print(f"  规格: {best_spec}")
        print(f"  指标: {best_metrics}")

        checkpoint = {
            "step": step + 1,
            "state_dict": net.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, f"{args.output_dir}/checkpoint_step_{step + 1}.pt")

    # 总结
    final_results = {
        "history": history,
        "final_score": history[-1]["score"] if history else float("nan"),
    }
    with open(f"{args.output_dir}/final_results.pkl", "wb") as f:
        pickle.dump(final_results, f)

    print("\n实验完成!")
    if len(history) > 0:
        print(f"最终综合分: {history[-1]['score']:.5f}")


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser("基于Block SWAP的逐步网络构建实验（强化版，含修复与快训重排）")
    # 搜索规模
    p.add_argument("--n_candidates", type=int, default=400, help="每一步的候选块数量")
    p.add_argument("--max_steps", type=int, default=5, help="最大堆叠步数")
    p.add_argument("--topk", type=int, default=12, help="Fast 阶段后进入 Final 的候选数量")
    # 两阶段评估强度
    p.add_argument("--batches_fast", type=int, default=1, help="Fast 阶段使用的 batch 数")
    p.add_argument("--inits_fast", type=int, default=1, help="Fast 阶段每个候选的初始化次数")
    p.add_argument("--batches_final", type=int, default=3, help="Final 阶段使用的 batch 数")
    p.add_argument("--inits_final", type=int, default=3, help="Final 阶段每个候选的初始化次数")
    # 数据
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--data_root", type=str, default="./data")
    # 目标与权重
    p.add_argument("--alpha", type=float, default=1.0, help="SWAP_norm 权重")
    p.add_argument("--beta", type=float, default=0.5, help="FLOPs 正则权重（对 log(1+FLOPs_G)）")
    p.add_argument("--gamma", type=float, default=0.05, help="参数量(M) 正则权重")
    p.add_argument("--delta", type=float, default=0.2, help="多样性惩罚权重（对 diversity_penalty 的整体系数）")
    # 设备与随机性
    p.add_argument("--cuda", action="store_true", help="使用 CUDA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_classes", type=int, default=10, help="分类类别数")
    # 输出
    p.add_argument("--output_dir", type=str, default="./results_plus")
    # 搜索空间可控项
    p.add_argument("--se_prob", type=float, default=0.3, help="SE 使用概率")
    p.add_argument("--k_choices", type=int, nargs="+", default=[3, 5], help="卷积核候选")
    p.add_argument("--exp_choices", type=int, nargs="+", default=[1, 3, 4, 6], help="扩展率候选")
    p.add_argument("--out_scale", type=float, nargs=2, default=[0.85, 1.25], help="出通道缩放范围（相对参考通道）")
    # 全局下采样约束
    p.add_argument("--base_downsamples", type=int, default=0, help="BaseSwap1 内部已包含的下采样次数估计")
    p.add_argument("--max_total_downsamples", type=int, default=3, help="全网最大下采样次数（stride=2 次数）")
    p.add_argument("--stride2_ratio", type=float, default=0.10, help="每步允许的 stride=2 候选占比（其余强制改为1）")
    # Final 快训重排
    p.add_argument("--enable_quick_rerank", action="store_true", help="开启 Final 阶段‘快训重排’")
    p.add_argument("--quick_steps", type=int, default=80, help="快训步数（极少）")
    p.add_argument("--quick_lr", type=float, default=5e-4, help="快训学习率")
    p.add_argument("--quick_weight", type=float, default=0.05, help="快训重排得分的权重")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 目录与随机性
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tic = time.time()
    run_experiment(args)
    print(f"总用时: {time.time() - tic:.1f} 秒")


if __name__ == "__main__":
    main()
