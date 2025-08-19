import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def set_seed(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class BaseSwap1(nn.Module):
    """基础网络结构"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

def prepare_batches(data_root, num_batches, batch_size, device):
    """准备用于SWAP计算的数据批次"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    batches = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        batches.append(images)
    
    return batches

def mbconv_block(in_c, out_c, k, exp, stride, se):
    """MobileNetV2风格的反转残差块"""
    layers = []
    
    # Expansion phase
    exp_c = in_c * exp
    if exp > 1:
        layers.extend([
            nn.Conv2d(in_c, exp_c, 1, bias=False),
            nn.BatchNorm2d(exp_c),
            nn.ReLU6(inplace=True)
        ])
    
    # Depthwise phase
    layers.extend([
        nn.Conv2d(exp_c, exp_c, k, stride, padding=k//2, groups=exp_c, bias=False),
        nn.BatchNorm2d(exp_c),
        nn.ReLU6(inplace=True)
    ])
    
    # SE block if requested
    if se:
        layers.append(SEBlock(exp_c))
    
    # Projection phase
    layers.extend([
        nn.Conv2d(exp_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c)
    ])
    
    return nn.Sequential(*layers)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y
