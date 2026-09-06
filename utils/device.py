"""
设备与随机种子 — CUDA 自动检测、复现性设置。
"""

import random

import numpy as np
import torch
from loguru import logger


def get_device(prefer: str = "auto") -> torch.device:
    """选择计算设备。

    "auto" 策略：先看 torch.cuda.is_available()，再做一次真实的小算力
    验证（创建 CUDA 张量做矩阵乘），避免驱动版本不匹配等场景下
    is_available() 误报 True、实际 kernel 调用抛错。验证失败回退 CPU。

    Args:
        prefer: "auto" / "cuda" / "cpu"

    Returns:
        torch.device
    """
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        try:
            # 真实算力验证
            t = torch.zeros(2, 2, device="cuda")
            t = t @ t
            torch.cuda.synchronize()
            del t
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"CUDA 可用: {name} ({mem:.1f} GB)，使用 GPU 训练")
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"CUDA 验证失败（{e}），回退 CPU")
    else:
        logger.info("CUDA 不可用，使用 CPU 训练（速度较慢）")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """固定全部随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        # TF32 加速，精度足够（checkpoint 往返断言容差 1e-5）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
