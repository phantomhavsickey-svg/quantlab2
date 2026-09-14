"""
交易成本模型 — A股真实的佣金、印花税、滑点。

与 quantlab 完全一致,保证 LightGBM vs Transformer 对比公平。
"""

import pandas as pd
import numpy as np
from loguru import logger


class TransactionCostModel:
    """A股交易成本计算。

    成本构成：
        1. 佣金（买卖双向）：默认万三(0.03%)，最低5元
        2. 印花税（仅卖出）：万五(0.05%)，2023年8月起
        3. 滑点（买卖双向）：默认0.1%

    可以配置：
        - 固定费率 + 最低佣金
        - 滑点可以按固定费率或波动率调整
    """

    def __init__(self,
                 commission_rate: float = 0.0003,
                 min_commission: float = 5.0,
                 stamp_tax_rate: float = 0.0005,
                 slippage_rate: float = 0.001):
        """
        Args:
            commission_rate: 佣金费率（默认万三）
            min_commission: 最低佣金（元）
            stamp_tax_rate: 印花税率（仅卖出，默认万五）
            slippage_rate: 滑点费率
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate

    def commission(self, trade_amount: float) -> float:
        """单边佣金。

        Args:
            trade_amount: 成交金额

        Returns:
            佣金（元）
        """
        fee = trade_amount * self.commission_rate
        return max(fee, self.min_commission)

    def stamp_tax(self, trade_amount: float, side: str) -> float:
        """印花税（仅卖出）。

        Args:
            trade_amount: 成交金额
            side: "buy" / "sell"

        Returns:
            印花税（元）
        """
        if side == "sell":
            return trade_amount * self.stamp_tax_rate
        return 0.0

    def slippage(self, trade_amount: float) -> float:
        """滑点成本。

        Args:
            trade_amount: 成交金额

        Returns:
            滑点（元）
        """
        return trade_amount * self.slippage_rate

    def total_cost(self, trade_amount: float, side: str) -> float:
        """单笔交易总成本。

        Args:
            trade_amount: 成交金额
            side: "buy" / "sell"

        Returns:
            总成本（元）
        """
        return (self.commission(trade_amount) +
                self.stamp_tax(trade_amount, side) +
                self.slippage(trade_amount))

    def apply_costs(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """对交易记录批量计算成本。

        Args:
            trades_df: DataFrame with columns [amount, side]

        Returns:
            增加 'commission', 'stamp_tax', 'slippage', 'total_cost' 列
        """
        df = trades_df.copy()
        df["commission"] = df["amount"].apply(self.commission)
        df["stamp_tax"] = df.apply(
            lambda r: self.stamp_tax(r["amount"], r["side"]), axis=1)
        df["slippage"] = df["amount"].apply(self.slippage)
        df["total_cost"] = (df["commission"] + df["stamp_tax"] +
                            df["slippage"])
        return df

    def effective_cost_rate(self, side: str = "buy") -> float:
        """单边有效成本率（费前）。"""
        base = self.commission_rate + self.slippage_rate
        if side == "sell":
            base += self.stamp_tax_rate
        return base

    def round_lot(self, quantity: int, lot_size: int = 100) -> int:
        """将股数向下取整到整手数。

        Args:
            quantity: 目标股数
            lot_size: 一手股数（A股=100）

        Returns:
            整手股数
        """
        return (quantity // lot_size) * lot_size

    def __repr__(self) -> str:
        return (f"TransactionCost(佣金={self.commission_rate*10000:.0f}‱, "
                f"最低{self.min_commission}元, "
                f"印花税={self.stamp_tax_rate*10000:.0f}‱, "
                f"滑点={self.slippage_rate*100:.2f}%)")
