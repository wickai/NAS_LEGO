#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证SWAP搜索得到的网络架构在CIFAR-10上的性能。
"""

import os
import sys
import time
import logging
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from thop import profile

from test import (
    set_seed,
    BaseSwap1,
    mbconv_block,
    SEBlock,
)


class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def mixup_data(x, y, alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_architecture(result_path):
    """加载搜索结果并解析架构"""
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    
    # 获取历史记录中的所有block规格
    history = results['history']
    block_specs = []
    
    # 从第一步开始收集所有选中的规格
    # 创建一个临时网络来获取实际的base通道数
    temp_net = BaseSwap1()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32)
        out = temp_net(dummy)
        current_channels = int(out.shape[1])
        print(f"检测到BaseSwap1输出通道数: {current_channels}")
    
    for step_info in history:
        spec = step_info['chosen_spec']
        assert spec['in_channels'] == current_channels, f"通道数不匹配：期望{current_channels}，实际{spec['in_channels']}"
        block_specs.append(spec)
        current_channels = spec['out_channels']
    
    return block_specs


class SearchedNet(nn.Module):
    """搜索得到的网络结构"""
    def __init__(self, block_specs, num_classes=10):
        super().__init__()
        # Base layers (from BaseSwap1)
        self.base = BaseSwap1()  # 输出64通道
        
        # 根据规格构建blocks
        self.blocks = nn.ModuleList()
        for spec in block_specs:
            block = nn.Sequential(
                mbconv_block(
                    spec['in_channels'],
                    spec['out_channels'],
                    k=spec['k'],
                    exp=spec['exp'],
                    stride=spec['stride'],
                    se=spec['se']
                )
            )
            self.blocks.append(block)
        
        # 获取最后一层的输出通道数
        final_channels = block_specs[-1]['out_channels'] if block_specs else 64
        
        # Classification head
        self.head = nn.Sequential(
            nn.Conv2d(final_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.base(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@torch.no_grad()
def get_model_complexity(model, device):
    """计算模型的参数量和FLOPs"""
    # 创建一个示例输入
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(dummy_input,))
    
    # 转换为更易读的格式
    if flops > 1e9:
        flops_str = f"{flops/1e9:.2f}G"
    elif flops > 1e6:
        flops_str = f"{flops/1e6:.2f}M"
    else:
        flops_str = f"{flops/1e3:.2f}K"
    
    if params > 1e6:
        params_str = f"{params/1e6:.2f}M"
    else:
        params_str = f"{params/1e3:.2f}K"
    
    return {
        'flops': flops,
        'params': params,
        'flops_str': flops_str,
        'params_str': params_str
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """计算Top-1和Top-5准确率"""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # Top-5
        _, pred_topk = outputs.topk(5, dim=1, largest=True, sorted=True)
        
        # Top-1
        correct_top1 += (pred_topk[:, 0] == labels).sum().item()
        
        # Top-5
        for i in range(labels.size(0)):
            if labels[i].item() in pred_topk[i].tolist():
                correct_top5 += 1
        
        total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc


def get_cifar10_dataloaders(root, batch_size, num_workers=2,
                           use_cutout=False, cutout_length=16,
                           val_ratio=0.1):
    """构造CIFAR-10的训练、验证和测试数据加载器"""
    transform_list = [
        transforms.RandAugment(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if use_cutout:
        transform_list.append(Cutout(n_holes=1, length=cutout_length))
    transform_list.append(
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                           [0.2023, 0.1994, 0.2010])
    )
    transform_train = transforms.Compose(transform_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                           [0.2023, 0.1994, 0.2010]),
    ])

    # Train/Val split
    full_train_ds_aug = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    full_train_ds_plain = datasets.CIFAR10(root, train=True, download=True, transform=transform_test)

    total_count = len(full_train_ds_aug)
    val_count = int(total_count * val_ratio)
    train_count = total_count - val_count

    generator = torch.Generator()
    generator.manual_seed(0)
    indices = torch.randperm(total_count, generator=generator).tolist()
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_ds = torch.utils.data.Subset(full_train_ds_aug, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds_plain, val_indices)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


def train_and_eval(model, train_loader, val_loader, test_loader, device, args):
    """完整训练过程"""
    epochs = args.train_epochs
    lr = args.lr
    mixup_alpha = args.mixup_alpha
    label_smoothing = args.label_smoothing
    weight_decay = args.weight_decay

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if mixup_alpha > 0.:
                mixed_x, y_a, y_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
                outputs = model(mixed_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc_top1 = correct_top1 / total if total > 0 else 0.

        val_top1, val_top5 = evaluate(model, val_loader, device)
        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                    f"Loss={train_loss:.3f}, "
                    f"Train@1={train_acc_top1*100:.2f}%, "
                    f"Val@1={val_top1*100:.2f}%, Val@5={val_top5*100:.2f}%")

    final_top1, final_top5 = evaluate(model, test_loader, device)
    logging.info(f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
    return final_top1


def setup_logger(log_path, log_name):
    os.makedirs(log_path, exist_ok=True)
    print("--- Set up basicConfig for logger ---")

    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_path, log_name), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser("验证SWAP搜索得到的网络架构")
    parser.add_argument("--arch_path", type=str, required=True,
                       help="搜索结果文件路径（final_results.pkl）")
    parser.add_argument("--log_path", default="./logs", type=str,
                       help="日志保存路径")
    parser.add_argument("--log_name", default="validate.log", type=str,
                       help="日志文件名")
    parser.add_argument("--data_path", default="./data", type=str,
                       help="CIFAR-10数据集路径")
    parser.add_argument("--cuda", action="store_true",
                       help="是否使用CUDA")
    parser.add_argument("--seed", default=42, type=int,
                       help="随机种子")
    
    # 训练相关参数
    parser.add_argument("--train_batch", default=128, type=int,
                       help="训练batch size")
    parser.add_argument("--train_epochs", default=200, type=int,
                       help="训练轮数")
    parser.add_argument("--lr", default=0.05, type=float,
                       help="初始学习率")
    parser.add_argument("--num_classes", default=10, type=int,
                       help="类别数")
    
    # cutout参数
    parser.add_argument("--use_cutout", action="store_true", default=True,
                       help="是否使用cutout数据增强")
    parser.add_argument("--cutout_length", default=16, type=int,
                       help="cutout区域大小")
    
    # mixup & label smoothing
    parser.add_argument("--mixup_alpha", default=0.2, type=float,
                       help="mixup alpha参数")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                       help="标签平滑因子")
    parser.add_argument("--weight_decay", default=5e-4, type=float,
                       help="权重衰减")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger(args.log_path, args.log_name)
    logging.info(f"Args: {args}")
    
    # 设置随机种子
    set_seed(args.seed)
    logging.info(f"Set random seed to {args.seed}")
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 加载搜索到的架构
    block_specs = load_architecture(args.arch_path)
    logging.info(f"Loaded architecture with {len(block_specs)} blocks")
    for i, spec in enumerate(block_specs, 1):
        logging.info(f"Block {i}: in={spec['in_channels']}, out={spec['out_channels']}, "
                    f"k={spec['k']}, exp={spec['exp']}, stride={spec['stride']}, "
                    f"se={spec['se']}")
    
    # 构建搜索得到的网络
    net = SearchedNet(block_specs, num_classes=args.num_classes)
    logging.info("Created searched network")
    
    # 计算模型复杂度
    net = net.to(device)  # 需要先将模型移到正确的设备上
    complexity_info = get_model_complexity(net, device)
    logging.info(f"Model complexity: {complexity_info['params_str']} parameters, {complexity_info['flops_str']} FLOPs")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        root=args.data_path,
        batch_size=args.train_batch,
        num_workers=2,
        use_cutout=args.use_cutout,
        cutout_length=args.cutout_length
    )
    logging.info("Prepared dataloaders")
    
    # 训练和评估
    logging.info("Starting training...")
    tic = time.time()
    final_top1 = train_and_eval(
        net,
        train_loader,
        val_loader,
        test_loader,
        device=device,
        args=args
    )
    training_time = time.time() - tic
    logging.info(f"Training completed in {training_time:.1f} seconds")
    logging.info(f"Final Test Accuracy: {final_top1*100:.2f}%")


if __name__ == "__main__":
    main()