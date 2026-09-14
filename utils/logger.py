"""
日志模块 — 基于 Loguru，支持文件滚动和终端彩色输出。
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = "logs/quantlab2.log",
                 rotation: str = "10 MB", retention: str = "30 days"):
    """初始化全局日志配置。

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 日志文件路径
        rotation: 日志滚动策略
        retention: 日志保留时间
    """
    # 移除默认 handler
    logger.remove()

    # 终端输出 — 彩色格式
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # 文件输出 — 完整格式
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    return logger


# 默认启动 logger
setup_logger()
