"""
绩效指标 — 收益率、风险、风险调整后收益。

与 quantlab 完全一致,保证 LightGBM vs Transformer 对比公平。
"""

import pandas as pd
import numpy as np
from loguru import logger


class PerformanceMetrics:
    """回测绩效指标计算。

    所有指标基于日频净值序列计算。
    """

    def __init__(self, periods_per_year: int = 252,
                 risk_free_rate: float = 0.02):
        """
        Args:
            periods_per_year: 年化交易日数（A股≈252）
            risk_free_rate: 无风险利率（默认2%）
        """
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    # ==================== 收益率 ====================

    def cumulative_return(self, equity_curve: pd.Series) -> float:
        """累计收益率。"""
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    def annual_return(self, equity_curve: pd.Series) -> float:
        """年化收益率 (CAGR)。"""
        total_return = self.cumulative_return(equity_curve)
        n_years = len(equity_curve) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return float((1 + total_return) ** (1 / n_years) - 1)

    def annual_volatility(self, daily_returns: pd.Series) -> float:
        """年化波动率。"""
        return float(daily_returns.std() * np.sqrt(self.periods_per_year))

    # ==================== 风险指标 ====================

    def max_drawdown(self, equity_curve: pd.Series) -> dict:
        """最大回撤。

        Returns:
            {drawdown: float, peak_date: date, trough_date: date, peak_value: float}
        """
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()
        # 峰值是 trough 之前的历史最高点
        peak_idx = equity_curve[:trough_idx].idxmax()

        return {
            "drawdown": float(abs(max_dd)),
            "peak_date": peak_idx,
            "trough_date": trough_idx,
            "peak_value": float(equity_curve[peak_idx]),
        }

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """最长回撤恢复期（天数）。"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        underwater = drawdown < 0
        if not underwater.any():
            return 0

        # 计算连续亏损天数
        streak = 0
        max_streak = 0
        for is_under in underwater:
            if is_under:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return max_streak

    # ==================== 风险调整后收益 ====================

    def sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """夏普比率 = (年化收益 - 无风险利率) / 年化波动率。"""
        ann_ret = self.annual_return(
            self._returns_to_equity(daily_returns))
        ann_vol = self.annual_volatility(daily_returns)
        if ann_vol == 0:
            return 0.0
        return float((ann_ret - self.risk_free_rate) / ann_vol)

    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """Calmar 比率 = 年化收益 / 最大回撤。"""
        ann_ret = self.annual_return(equity_curve)
        dd_info = self.max_drawdown(equity_curve)
        if dd_info["drawdown"] == 0:
            return 0.0
        return float(ann_ret / dd_info["drawdown"])

    def sortino_ratio(self, daily_returns: pd.Series) -> float:
        """Sortino 比率（仅用下行波动率）。"""
        ann_ret = self.annual_return(
            self._returns_to_equity(daily_returns))
        neg_returns = daily_returns[daily_returns < 0]
        if len(neg_returns) == 0:
            return float("inf")
        downside_vol = neg_returns.std() * np.sqrt(self.periods_per_year)
        if downside_vol == 0:
            return 0.0
        return float((ann_ret - self.risk_free_rate) / downside_vol)

    def information_ratio(self, daily_returns: pd.Series,
                          benchmark_daily_returns: pd.Series) -> float:
        """信息比率 = mean(超额收益) / std(超额收益) * sqrt(252)。"""
        excess = daily_returns - benchmark_daily_returns.fillna(0)
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() *
                     np.sqrt(self.periods_per_year))

    # ==================== 胜率与盈亏比 ====================

    def win_rate(self, daily_returns: pd.Series) -> float:
        """日胜率。"""
        return float((daily_returns > 0).mean())

    def win_rate_by_trade(self, trades_df: pd.DataFrame) -> float:
        """按交易订单的胜率（基于买卖价差）。"""
        if trades_df.empty:
            return 0.0
        # 计算每笔交易的盈亏
        sells = trades_df[trades_df["side"] == "sell"]
        if sells.empty:
            return 0.0
        return float((sells["net_proceeds"] > 0).mean())

    def profit_factor(self, trades_df: pd.DataFrame) -> float:
        """利润因子 = 总盈利 / 总亏损。"""
        if trades_df.empty:
            return 0.0
        sells = trades_df[trades_df["side"] == "sell"]
        if sells.empty:
            return 0.0
        gross_profit = sells[sells["net_proceeds"] > 0]["net_proceeds"].sum()
        gross_loss = abs(sells[sells["net_proceeds"] < 0]["net_proceeds"].sum())
        if gross_loss == 0:
            return float("inf")
        return float(gross_profit / gross_loss)

    # ==================== 月度收益表 ====================

    def monthly_returns_table(self, equity_curve: pd.Series) -> pd.DataFrame:
        """生成月度收益热力图数据。

        Returns:
            DataFrame (index=年份, columns=月份)
        """
        monthly = equity_curve.resample("ME").last().pct_change()
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.year, monthly.index.month],
            names=["year", "month"]
        )
        table = monthly.unstack(level="month")
        # 百分比形式
        return table * 100

    # ==================== 汇总 ====================

    def compute_all(self, equity_curve: pd.Series,
                    benchmark_curve: pd.Series | None = None,
                    trades: pd.DataFrame | None = None) -> dict:
        """计算所有绩效指标。

        Args:
            equity_curve: 策略净值曲线
            benchmark_curve: 基准净值曲线（可选）
            trades: 交易记录 DataFrame（可选）

        Returns:
            指标字典
        """
        daily_returns = equity_curve.pct_change().fillna(0)

        dd_info = self.max_drawdown(equity_curve)

        metrics = {
            "cumulative_return": self.cumulative_return(equity_curve),
            "annual_return": self.annual_return(equity_curve),
            "annual_volatility": self.annual_volatility(daily_returns),
            "max_drawdown": dd_info["drawdown"],
            "max_drawdown_peak_date": dd_info["peak_date"],
            "max_drawdown_trough_date": dd_info["trough_date"],
            "max_drawdown_duration": self.max_drawdown_duration(equity_curve),
            "sharpe_ratio": self.sharpe_ratio(daily_returns),
            "calmar_ratio": self.calmar_ratio(equity_curve),
            "sortino_ratio": self.sortino_ratio(daily_returns),
            "win_rate": self.win_rate(daily_returns),
        }

        # 基准相关
        if benchmark_curve is not None:
            bm_returns = benchmark_curve.pct_change().fillna(0)
            metrics["benchmark_cumulative_return"] = \
                self.cumulative_return(benchmark_curve)
            metrics["benchmark_annual_return"] = \
                self.annual_return(benchmark_curve)
            metrics["excess_return"] = (metrics["cumulative_return"] -
                                        metrics["benchmark_cumulative_return"])
            metrics["information_ratio"] = \
                self.information_ratio(daily_returns, bm_returns)

        # 交易统计
        if trades is not None and not trades.empty:
            metrics["total_trades"] = len(trades)
            metrics["total_cost"] = trades["cost"].sum() \
                if "cost" in trades.columns else 0
            metrics["n_buys"] = (trades["side"] == "buy").sum()
            metrics["n_sells"] = (trades["side"] == "sell").sum()
            metrics["win_rate_by_trade"] = self.win_rate_by_trade(trades)
            metrics["profit_factor"] = self.profit_factor(trades)

        return metrics

    # ==================== 辅助 ====================

    @staticmethod
    def _returns_to_equity(returns: pd.Series) -> pd.Series:
        """日收益率序列 → 净值序列。"""
        return (1 + returns).cumprod()
