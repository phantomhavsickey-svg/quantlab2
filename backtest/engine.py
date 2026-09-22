"""
回测引擎 — 月度调仓撮合 + 逐日盯市,含交易成本与可交易性约束。

2026-09 执行层修复后的四条口径变化(相对 quantlab 的同名引擎):
    1. 建仓按"目标市值 - 现有市值"的差额下单。旧实现是
       `buy_capital = cash / len(target)` 再按 DataFrame 行序逐个买到没钱,
       低换手的月份里新入选股票只分到卖出回笼资金的 1/N,等权是名义上的。
    2. 盯市区间改用**该区间实际持有的组合**。旧实现先更新持仓、再回补上一
       个月的每日净值,等于每个月都用下一个月末才决定的组合给上个月估值,
       日收益/夏普/最大回撤里混进一个月的前视。
    3. 涨停不买、跌停不卖、停牌或当日无 bar 不成交 —— 与实盘共用
       utils/market_rules.py 的判定,不再"回测假设成交、实盘被拒单"。
    4. 持仓股票当日缺 bar 时沿用最近有效收盘估值,市值不再从净值里凭空消失;
       另加显式 T+1:当日买入的股份不可当日卖出。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from loguru import logger
from tqdm import tqdm

from backtest.cost import TransactionCostModel
from backtest.metrics import PerformanceMetrics
from utils.market_rules import can_fill, DATE_COL
from utils.position_policy import (PolicyConfig, apply_fills, plan as policy_plan,
                                   sync_intent_shares)
from utils.sizing import rebalance_plan, scale_buys_to_budget


class BacktestEngine:
    """事件循环回测引擎(逐调仓日撮合,逐交易日盯市)。

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
                 cost_model: TransactionCostModel | None = None,
                 lot_size: int = 100,
                 policy: PolicyConfig | None = None):
        """
        Args:
            initial_capital: 初始资金
            rebalance_frequency: 调仓频率 daily/weekly/monthly
                分数带位策略下这是**策略评估频率**(补仓/减仓的触发节奏)
            max_positions: 最大持仓数(信号端已按 Top-K 截断,此处仅记录;
                策略模式下由 policy.max_names 管住新仓,该值不生效)
            cost_model: 交易成本模型
            lot_size: 一手股数
            policy: PolicyConfig → 用分数带位建仓/补仓/减仓;None = 等权 Top-K
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.lot_size = lot_size
        self.cost = cost_model or TransactionCostModel()
        self.policy = policy
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
                - marks: 每日盯市明细 DataFrame(净值/现金/市值/持仓数)
                - daily_returns: 每日收益率 Series
                - trades: 所有成交记录 DataFrame
                - positions: 每日持仓快照 DataFrame
                - metrics: 绩效指标 dict
                - benchmark_curve: 基准净值 Series (if provided)
                - execution: 执行层诊断(持有期/换手/被挡指令/滞留现金)
        """
        logger.info(f"开始回测: 初始资金={self.initial_capital:,.0f}, "
                     f"持仓上限={self.max_positions}, "
                     f"调仓频率={self.rebalance_frequency}")
        if self.policy is not None:
            if "score" not in signals.columns:
                raise ValueError("分数带位策略需要 signals 带 score 列"
                                 "(用 scores_from_predictions 生成全截面分数)")
            logger.info(f"仓位策略已启用: 建仓线 {self.policy.buy_score:.4f} / "
                        f"清仓线 {self.policy.sell_score:.4f} / "
                        f"{self.policy.base_weight:.0%}→"
                        f"{self.policy.max_entry_weight:.0%} 建仓, "
                        f"单票上限 {self.policy.max_position_weight:.0%}, "
                        f"新仓上限 {self.policy.max_names} 只, "
                        f"评估频率 {self.rebalance_frequency}")

        # 价格面板 + 按日期预索引(旧实现在每个交易日里反复 set_index)
        px = self._index_by_date(data_dict)
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
        cash = float(self.initial_capital)
        positions = {}        # {symbol: shares}
        entry_date = {}       # {symbol: 建仓交易日} —— 持有期统计用
        locked = {}           # {symbol: 当日买入股数} —— T+1 按股份而不是按票
        last_exec = None      # 上一次撮合日,用于跨日解锁 locked
        last_price = {}       # {symbol: 最近有效收盘} —— 缺行估值兜底
        states = {}           # {symbol: NameState} —— 分数带位策略状态
        equity_curve = []     # [(date, total_value)]
        all_trades = []       # 记录每笔成交
        all_positions = []    # 每日持仓快照

        all_dates = list(pd.DatetimeIndex(sorted(price_df.index)))
        date_pos = {d: k for k, d in enumerate(all_dates)}
        cursor = 0            # 下一个待盯市的日期下标
        started = False       # 首个成交日之前不写净值(避免回标历史)

        blocked_buy = Counter()
        blocked_sell = Counter()
        hold_days = []
        turnover = []
        idle_cash = []
        policy_actions = Counter()
        policy_gross = []
        policy_names = []

        # 主循环：每个调仓日
        for rebal_date in tqdm(rebalance_dates, desc="回测进行中"):
            # --- Step 1: 获取当日的目标组合 ---
            try:
                day_signals = signals.xs(rebal_date, level="date")
            except KeyError:
                continue

            ws = day_signals[day_signals["weight"] > 0]["weight"]
            weights = {str(s): float(v) for s, v in ws.items()}
            # 策略模式要的是**全截面分数**(signals 的 score 列):Top-K 之外的名字
            # 可能是"分数过建仓线"的候选,已持仓的名字更不该因为掉出 Top-K 而丢掉
            # 分数 —— 那会被误当成"分数跌破清仓线"。
            scores = ({str(s): float(v)
                       for s, v in day_signals["score"].items()
                       if pd.notna(v)} if self.policy is not None else {})

            # --- Step 2: 找到下一个交易日（成交日） ---
            exec_date = self._next_date(all_dates, rebal_date)
            if exec_date is None or exec_date not in date_pos:
                continue
            ei = date_pos[exec_date]
            if exec_date != last_exec:       # 跨日解锁:昨日买入今日可卖
                locked.clear()
                last_exec = exec_date

            # --- Step 3: 先把 [cursor, ei) 用"这段时间实际持有的组合"盯市 ---
            # (修复:旧实现先更新持仓再回补上一区间,等于用下个月的组合给
            #  这个月估值,日收益里混进一个调仓周期的前视)
            if started:
                for k in range(cursor, ei):
                    self._mark_day(all_dates[k], positions, last_price, px,
                                   cash, equity_curve, all_positions)
            cursor = ei

            # --- Step 4: 撮合价与可成交性(全部取自 exec_date 当日 bar) ---
            # 建仓候选只需覆盖"分数过建仓线"的名字,全截面里低分名字不必取 bar
            cands = (set(weights) if self.policy is None else
                     {s for s, x in scores.items() if x >= self.policy.buy_score})
            universe = cands | set(positions)
            rows = {s: self._row(px, s, exec_date) for s in universe}
            for s in universe:
                cp = self._close_of(rows[s])
                if cp is not None:
                    last_price[s] = cp

            def exec_price(sym):
                o = self._open_of(rows.get(sym))
                return o if o is not None else last_price.get(sym)

            prices = {s: exec_price(s) for s in universe
                      if exec_price(s) is not None}
            equity = cash + sum(q * prices.get(s, last_price.get(s, 0.0))
                                for s, q in positions.items())

            # --- Step 5: 目标市值 - 现有市值 → 买卖差额(与实盘同一份数学) ---
            pol = None
            if self.policy is None:
                plan = rebalance_plan(weights, positions, prices, equity,
                                      lot_size=self.lot_size)
            else:
                pol = policy_plan(scores, positions, prices, states, equity,
                                  self.policy, lot_size=self.lot_size,
                                  asof=exec_date)
                plan = pol.plan
                for k, v in pol.actions.items():
                    policy_actions[k] += v
                if pol.gross_weight > 0:
                    policy_gross.append(pol.gross_weight)
                for s, why in pol.notes.items():
                    logger.debug(f"未动作 {exec_date.date()} {s}: {why}")

            before = dict(positions)
            fill_price = {}

            # --- Step 6: 卖出(先卖后买;跌停/停牌/T+1 挡下的留在持仓里) ---
            for sym in sorted(plan.sells):
                avail = positions.get(sym, 0) - locked.get(sym, 0)
                if avail <= 0:
                    if positions.get(sym, 0) > 0:
                        blocked_sell["T+1 当日买入"] += 1
                    continue
                qty = min(plan.sells[sym], avail)
                if qty <= 0:
                    continue
                ok, why = can_fill(rows.get(sym), sym, "sell")
                if not ok:
                    blocked_sell[why] += 1
                    continue
                sell_price = prices[sym]
                amount = sell_price * qty
                cost = self.cost.total_cost(amount, "sell")
                cash += amount - cost
                fill_price[sym] = sell_price
                if entry_date.get(sym) is not None:
                    hold_days.append(self._holding_days(
                        date_pos, entry_date[sym], exec_date))
                if qty >= positions[sym]:
                    del positions[sym]
                    entry_date.pop(sym, None)
                else:
                    positions[sym] -= qty

                all_trades.append({
                    "date": exec_date,
                    "symbol": sym,
                    "side": "sell",
                    "price": sell_price,
                    "shares": qty,
                    "amount": amount,
                    "cost": cost,
                    "net_proceeds": amount - cost,
                })

            # --- Step 7: 买入(钱不够时按剩余现金等比缩量,与下单顺序无关) ---
            scale_buys_to_budget(plan, cash, prices, lot_size=self.lot_size,
                                 fee_rate_buy=self.cost.effective_cost_rate(
                                     "buy"))
            if pol is not None:
                # 现金缩量是"这一档只能买到这么多",按缩量后的股数推进状态
                sync_intent_shares(pol.intents, plan)
            traded = 0.0
            for sym in sorted(plan.buys):
                shares = plan.buys[sym]
                ok, why = can_fill(rows.get(sym), sym, "buy")
                if not ok:
                    blocked_buy[why] += 1
                    continue
                buy_price = prices[sym]
                amount = buy_price * shares
                cost = self.cost.total_cost(amount, "buy")
                if amount + cost > cash:
                    blocked_buy["现金不足"] += 1
                    continue

                cash -= amount + cost
                positions[sym] = positions.get(sym, 0) + shares
                locked[sym] = locked.get(sym, 0) + shares   # T+1:今日买入不可卖
                fill_price[sym] = buy_price
                entry_date.setdefault(sym, exec_date)
                traded += amount

                all_trades.append({
                    "date": exec_date,
                    "symbol": sym,
                    "side": "buy",
                    "price": buy_price,
                    "shares": shares,
                    "amount": amount,
                    "cost": cost,
                    "net_proceeds": -(amount + cost),
                })

            if equity > 0:
                turnover.append(traded / equity)
            invested = sum(q * prices.get(s, last_price.get(s, 0.0))
                           for s, q in positions.items())
            if equity > 0 and invested / equity < 0.98:
                idle_cash.append(cash)
            started = True

            # --- Step 7.5: 按**实际股数变化**提交策略状态 ---
            # 涨停挡买、跌停挡卖、T+1 挡卖、现金缩量掉的部分都不推进参考分数,
            # 下一轮同一档条件仍然成立 → 自动补做,不会出现"钱花了仓位没记上"。
            if pol is not None:
                for s, msg in apply_fills(states, pol.intents, before,
                                          positions, fill_price,
                                          asof=exec_date).items():
                    logger.debug(f"策略状态 {exec_date.date()} {s}: {msg}")
                policy_names.append(len(positions))

            # --- Step 8: 成交日当天用撮合后的新组合盯市 ---
            self._mark_day(exec_date, positions, last_price, px, cash,
                           equity_curve, all_positions)
            cursor = ei + 1

        # 最后一个调仓日之后继续盯市到数据末端
        if started:
            for k in range(cursor, len(all_dates)):
                self._mark_day(all_dates[k], positions, last_price, px, cash,
                               equity_curve, all_positions)

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

        # 执行层诊断(把"学 20 日 / 持有到月末"这类口径差量化出来)
        execution = {
            "blocked_buy": dict(blocked_buy),
            "blocked_sell": dict(blocked_sell),
            "n_blocked_buy": int(sum(blocked_buy.values())),
            "n_blocked_sell": int(sum(blocked_sell.values())),
            "median_hold_days": float(np.median(hold_days)) if hold_days
            else float("nan"),
            "p90_hold_days": float(np.percentile(hold_days, 90))
            if hold_days else float("nan"),
            "n_exit_trades": len(hold_days),
            "mean_turnover": float(np.mean(turnover)) if turnover else 0.0,
            "n_underinvested_rebalances": len(idle_cash),
            "mean_idle_cash": float(np.mean(idle_cash)) if idle_cash else 0.0,
            "final_cash": float(cash),
        }
        if self.policy is not None:
            # 策略诊断:动作构成、实际用掉的仓位、持仓只数
            execution["policy_actions"] = dict(policy_actions)
            execution["policy_mean_gross_weight"] = (
                float(np.mean(policy_gross)) if policy_gross else float("nan"))
            execution["policy_mean_names"] = (
                float(np.mean(policy_names)) if policy_names else 0.0)
            execution["policy_final_names"] = len(positions)

        # 打印结果
        self._print_summary(perf, execution)

        return {
            "equity_curve": equity_df["equity"],
            "marks": equity_df,
            "daily_returns": daily_returns,
            "trades": trades_df,
            "positions": positions_df,
            "metrics": perf,
            "benchmark_curve": bm_curve,
            "execution": execution,
            "policy_states": dict(states),
        }

    # ==================== 盯市 ====================

    @staticmethod
    def _mark_day(d, positions, last_price, px, cash,
                  equity_curve, all_positions):
        """用**当前**持仓给某天估值;当日缺 bar 的股票沿用最近有效收盘。"""
        total_market_value = 0.0
        for sym, shares in positions.items():
            cp = BacktestEngine._close_of(BacktestEngine._row(px, sym, d))
            if cp is not None:
                last_price[sym] = cp
            price = last_price.get(sym)
            if price is None:
                continue          # 从未有过有效收盘 → 无法计入(不应发生)
            total_market_value += price * shares
            all_positions.append({
                "date": d,
                "symbol": sym,
                "shares": shares,
                "price": price,
            })

        total_value = cash + total_market_value
        equity_curve.append({
            "date": d,
            "total_value": total_value,
            "cash": cash,
            "market_value": total_market_value,
            "n_positions": len(positions),
        })

    # ==================== 辅助方法 ====================

    @staticmethod
    def _index_by_date(data_dict: dict) -> dict:
        """{symbol: 以日期为索引的 DataFrame},整场回测只建一次。"""
        out = {}
        for sym, df in data_dict.items():
            if df is None or len(df) == 0 or "日期" not in df.columns:
                continue
            s = df.set_index("日期")
            s.index = pd.DatetimeIndex(s.index)
            out[str(sym)] = s[~s.index.duplicated(keep="last")].sort_index()
        return out

    @staticmethod
    def _row(px: dict, sym, d):
        df = px.get(str(sym))
        if df is None or d not in df.index:
            return None
        return df.loc[d]

    @staticmethod
    def _open_of(row):
        if row is None or "开盘" not in row.index:
            return None
        v = row["开盘"]
        return None if v is None or pd.isna(v) or float(v) <= 0 else float(v)

    @staticmethod
    def _close_of(row):
        if row is None or "收盘" not in row.index:
            return None
        v = row["收盘"]
        return None if v is None or pd.isna(v) or float(v) <= 0 else float(v)

    @staticmethod
    def _holding_days(date_pos: dict, entry, exit_) -> int:
        a = date_pos.get(pd.Timestamp(entry))
        b = date_pos.get(pd.Timestamp(exit_))
        return b - a if a is not None and b is not None else 0

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

    def _print_summary(self, metrics: dict, execution: dict | None = None):
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
        if execution:
            print("  --- 执行层诊断 ---")
            print(f"  中位持有交易日: {execution['median_hold_days']:.0f}"
                  f" (p90 {execution['p90_hold_days']:.0f},"
                  f" 减仓/平仓 {execution['n_exit_trades']} 笔)")
            print(f"  平均单次换手率: {execution['mean_turnover']*100:.1f}%")
            print(f"  买入被挡 {execution['n_blocked_buy']} 笔: "
                  f"{execution['blocked_buy'] or '无'}")
            print(f"  卖出被挡 {execution['n_blocked_sell']} 笔: "
                  f"{execution['blocked_sell'] or '无'}")
            print(f"  欠配调仓次数: {execution['n_underinvested_rebalances']},"
                  f" 该些次平均滞留现金 "
                  f"{execution['mean_idle_cash']:,.0f} 元")
            if "policy_actions" in execution:
                print(f"  策略动作: {execution['policy_actions']}")
                print(f"  平均目标仓位: {execution['policy_mean_gross_weight']*100:.1f}%,"
                      f" 平均持仓 {execution['policy_mean_names']:.1f} 只,"
                      f" 期末 {execution['policy_final_names']} 只")
        print("=" * 60 + "\n")
