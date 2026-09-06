"""
向量化回测引擎 — 模拟多股票多期调仓，计算净值曲线。

与 quantlab 完全一致,保证 LightGBM vs Transformer 对比公平。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from tqdm import tqdm

from backtest.cost import TransactionCostModel
from backtest.metrics import PerformanceMetrics


class BacktestEngine:
    """向量化回测引擎。

    关键假设：
        - 日频回测，信号在调仓日收盘后生成
        - 实际成交在下一个交易日的开盘价
        - 交易成本包含佣金、印花税（仅卖）、滑点
        - T+1 规则：当天买的股票次日才能卖
        - 仅做多（long only）
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 rebalance_frequency: str = "monthly",
                 max_positions: int = 30,
                 cost_model: TransactionCostModel | None = None):
        """
        Args:
            initial_capital: 初始资金
            rebalance_frequency: 调仓频率 daily/weekly/monthly
            max_positions: 最大持仓数
            cost_model: 交易成本模型
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.cost = cost_model or TransactionCostModel()
        self.metrics = PerformanceMetrics()

    # ==================== 主回测循环 ====================

    def run(self, data_dict: dict,
            signals: pd.DataFrame,
            benchmark_prices: pd.Series | None = None) -> dict:
        """执行回测。

        Args:
            data_dict: {symbol: DataFrame(日期, 开盘, 收盘, ...)} 日线数据字典
            signals: 交易信号（来自 signals_from_predictions）
                     MultiIndex (date, symbol), 含 'weight' 列
            benchmark_prices: 基准指数价格序列（可选，用于相对绩效）

        Returns:
            dict with keys:
                - equity_curve: 每日净值 Series
                - daily_returns: 每日收益率 Series
                - trades: 所有成交记录 DataFrame
                - positions: 每日持仓快照 DataFrame
                - metrics: 绩效指标 dict
                - benchmark_curve: 基准净值 Series (if provided)
        """
        logger.info(f"开始回测: 初始资金={self.initial_capital:,.0f}, "
                     f"持仓上限={self.max_positions}, "
                     f"调仓频率={self.rebalance_frequency}")

        # 准备价格面板
        price_df = self._build_price_panel(data_dict)

        # 获取调仓日期（signal 中实际存在的日期）
        signal_dates = sorted(signals.index.get_level_values("date").unique())

        if self.rebalance_frequency == "monthly":
            # 每月最后一个 signal date
            rebalance_dates = self._get_rebalance_dates(signal_dates, "monthly")
        elif self.rebalance_frequency == "weekly":
            rebalance_dates = self._get_rebalance_dates(signal_dates, "weekly")
        else:
            rebalance_dates = signal_dates  # daily

        logger.info(f"调仓日数量: {len(rebalance_dates)}")

        # 初始化
        cash = self.initial_capital
        positions = {}        # {symbol: shares}
        equity_curve = []     # [(date, total_value)]
        all_trades = []       # 记录每笔成交
        all_positions = []    # 每日持仓快照

        all_dates = sorted(price_df.index)
        prev_positions_set = set()

        # 主循环：每个调仓日
        for i, rebal_date in enumerate(tqdm(rebalance_dates, desc="回测进行中")):
            # --- Step 1: 获取当日的目标组合 ---
            try:
                day_signals = signals.xs(rebal_date, level="date")
            except KeyError:
                continue

            target_weights = day_signals[day_signals["weight"] > 0]
            target_symbols = set(target_weights.index)

            # --- Step 2: 找到下一个交易日（成交日） ---
            exec_date = self._next_date(all_dates, rebal_date)
            if exec_date is None:
                continue

            # --- Step 3: 卖出不在目标组合中的持仓 ---
            to_sell = prev_positions_set - target_symbols
            for sym in to_sell:
                if sym not in positions or positions[sym] <= 0:
                    continue
                if sym not in data_dict:
                    continue

                df_sym = data_dict[sym].set_index("日期")
                if exec_date not in df_sym.index:
                    continue

                sell_price = df_sym.loc[exec_date, "开盘"]
                shares = positions[sym]
                amount = sell_price * shares

                # 计算成本
                cost = self.cost.total_cost(amount, "sell")
                proceeds = amount - cost

                cash += proceeds

                all_trades.append({
                    "date": exec_date,
                    "symbol": sym,
                    "side": "sell",
                    "price": sell_price,
                    "shares": shares,
                    "amount": amount,
                    "cost": cost,
                    "net_proceeds": proceeds,
                })

                del positions[sym]

            # --- Step 4: 买入目标组合 ---
            if target_symbols:
                # 计算买入金额
                buy_capital = cash / max(len(target_symbols), 1)
                # 等权分配
                for sym, row in target_weights.iterrows():
                    if sym not in data_dict:
                        continue

                    df_sym = data_dict[sym].set_index("日期")
                    if exec_date not in df_sym.index:
                        continue

                    buy_price = df_sym.loc[exec_date, "开盘"]

                    # 整手买入（100股的整数倍）
                    target_amount = buy_capital * 1.0  # equal weight
                    shares = self.cost.round_lot(int(target_amount / buy_price))
                    if shares <= 0:
                        continue

                    amount = buy_price * shares
                    cost = self.cost.total_cost(amount, "buy")
                    total_cost = amount + cost

                    if total_cost > cash:
                        # 钱不够，减少股数
                        shares = self.cost.round_lot(
                            int((cash * 0.99 - self.cost.min_commission) / buy_price)
                        )
                        if shares <= 0:
                            continue
                        amount = buy_price * shares
                        cost = self.cost.total_cost(amount, "buy")
                        total_cost = amount + cost

                    cash -= total_cost
                    positions[sym] = positions.get(sym, 0) + shares

                    all_trades.append({
                        "date": exec_date,
                        "symbol": sym,
                        "side": "buy",
                        "price": buy_price,
                        "shares": shares,
                        "amount": amount,
                        "cost": cost,
                        "net_proceeds": -total_cost,
                    })

            prev_positions_set = set(positions.keys())

            # --- Step 5: 每日盯市（从上个调仓日到当前调仓日） ---
            if i == 0:
                # 首个调仓日之前没有持仓：从首个成交日开始盯市，
                # 避免用首个组合回标历史价格产生虚假的期初波动
                start_idx = all_dates.index(exec_date)
            else:
                # 从上个成交日的次日起（上个成交日已在上一轮盯市）
                start_idx = all_dates.index(
                    self._next_date(all_dates, rebalance_dates[i-1])
                    or all_dates[0]) + 1
            end_idx = all_dates.index(exec_date) + 1
            if i == len(rebalance_dates) - 1:
                end_idx = len(all_dates)

            for j in range(start_idx, end_idx):
                d = all_dates[j]
                # 计算持仓市值
                total_market_value = 0.0
                for sym, shares in positions.items():
                    if sym in data_dict:
                        df_sym = data_dict[sym].set_index("日期")
                        if d in df_sym.index:
                            close_price = df_sym.loc[d, "收盘"]
                            total_market_value += close_price * shares

                total_value = cash + total_market_value

                equity_curve.append({
                    "date": d,
                    "total_value": total_value,
                    "cash": cash,
                    "market_value": total_market_value,
                    "n_positions": len(positions),
                })

                # 持仓快照
                for sym, shares in positions.items():
                    if sym in data_dict:
                        df_sym = data_dict[sym].set_index("日期")
                        if d in df_sym.index:
                            all_positions.append({
                                "date": d,
                                "symbol": sym,
                                "shares": shares,
                                "price": df_sym.loc[d, "收盘"],
                            })

        # ==================== 最终输出 ====================

        equity_df = pd.DataFrame(equity_curve)
        if equity_df.empty:
            logger.error("回测未生成任何净值数据")
            return {}

        equity_df = equity_df.set_index("date")
        equity_df["equity"] = equity_df["total_value"] / self.initial_capital

        # 日收益率
        daily_returns = equity_df["total_value"].pct_change().fillna(0)
        daily_returns.name = "daily_return"

        # 交易记录
        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values(["date", "symbol"])

        # 持仓记录
        positions_df = pd.DataFrame(all_positions)

        # 绩效指标
        bm_curve = None
        if benchmark_prices is not None:
            bm_returns = benchmark_prices.pct_change().fillna(0)
            bm_curve = (1 + bm_returns).cumprod()
            # 对齐日期
            bm_curve = bm_curve.reindex(equity_df.index).ffill()
            perf = self.metrics.compute_all(
                equity_df["equity"], bm_curve, trades_df)
        else:
            perf = self.metrics.compute_all(equity_df["equity"],
                                             trades=trades_df)

        # 打印结果
        self._print_summary(perf)

        return {
            "equity_curve": equity_df["equity"],
            "daily_returns": daily_returns,
            "trades": trades_df,
            "positions": positions_df,
            "metrics": perf,
            "benchmark_curve": bm_curve,
        }

    # ==================== 辅助方法 ====================

    @staticmethod
    def _build_price_panel(data_dict: dict) -> pd.DataFrame:
        """将 per-symbol 价格字典转为面板 (date x symbol)。"""
        panels = {}
        for sym, df in data_dict.items():
            df = df.set_index("日期")
            if "收盘" in df.columns:
                panels[sym] = df["收盘"]
        return pd.DataFrame(panels).sort_index()

    @staticmethod
    def _next_date(dates: list, ref_date) -> object | None:
        """找到严格大于 ref_date 的下一个日期。"""
        ref = pd.Timestamp(ref_date)
        for d in dates:
            if d > ref:
                return d
        return None

    @staticmethod
    def _get_rebalance_dates(signal_dates: list, freq: str) -> list:
        """从 signal 日期中筛选调仓日。"""
        if freq == "monthly":
            # 每月最后一个日期
            df = pd.DataFrame({"date": signal_dates})
            df["ym"] = df["date"].dt.strftime("%Y-%m")
            return df.groupby("ym")["date"].last().sort_values().tolist()
        elif freq == "weekly":
            # 每周五
            return [d for d in signal_dates if d.weekday() == 4]
        return signal_dates

    def _print_summary(self, metrics: dict):
        """打印回测摘要。"""
        print("\n" + "=" * 60)
        print("  回测结果摘要")
        print("=" * 60)
        print(f"  累计收益率:   {metrics.get('cumulative_return', 0)*100:8.2f}%")
        print(f"  年化收益率:   {metrics.get('annual_return', 0)*100:8.2f}%")
        print(f"  年化波动率:   {metrics.get('annual_volatility', 0)*100:8.2f}%")
        print(f"  夏普比率:     {metrics.get('sharpe_ratio', 0):8.2f}")
        print(f"  最大回撤:     {metrics.get('max_drawdown', 0)*100:8.2f}%")
        print(f"  Calmar比率:   {metrics.get('calmar_ratio', 0):8.2f}")
        print(f"  胜率:         {metrics.get('win_rate', 0)*100:8.2f}%")
        if "information_ratio" in metrics:
            print(f"  信息比率:     {metrics.get('information_ratio', 0):8.2f}")
        print(f"  总交易次数:   {metrics.get('total_trades', 0)}")
        print(f"  交易总成本:   {metrics.get('total_cost', 0):,.0f} 元")
        print("=" * 60 + "\n")
