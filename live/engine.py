"""
实时交易引擎 — 盘中轮询行情 + 定时调仓 + 风控 + 多券商执行。

流程:
    启动(交易日盘中前台运行,或 --once 单次执行):
    ├─ 加载最新 checkpoint + SequenceStore + 预测器
    ├─ 主循环(每 poll_interval 秒):
    │    ├─ 拉取 持仓 ∪ 目标 股票实时行情
    │    ├─ 到调仓时间(默认 09:35)且今日是调仓日(默认月末)且今日未调仓:
    │    │    信号(因子截止上一交易日,lag-1 口径) → 目标权重
    │    │    → make_orders → 风控过滤 → 按后端执行
    │    │    simulate: 下单 + 当日 bar 撮合 + 状态落盘
    │    │    qmt:      --confirm 才真实下单,否则仅导出指令 CSV
    │    │    none:     仅导出指令 CSV
    │    └─ 收盘后(15:05)盯市 + 净值快照
    └─ Ctrl+C 退出

--once 模式:单次完整执行(信号→指令→风控→执行→盯市)后退出,
任何一天都可运行(测试/计划任务用),不检查调仓日。
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from live.broker import Broker, OrderSide, OrderStatus
from live.journal import TradeJournal
from live.orders import make_orders, export_orders
from live.risk import RiskManager
from live.__init__ import SinaQuoteFeed, Quote
from utils.market_rules import at_limit_up, at_limit_down


def _num(row, col, default=None):
    """从日线一行里取数,缺失/NaN 时给默认值(行情退化时用)。"""
    try:
        v = row[col]
    except (KeyError, IndexError):
        return default
    if v is None or pd.isna(v):
        return default
    return float(v)


class LiveEngine:
    """实时交易引擎。"""

    def __init__(self, config: dict, broker: Broker, *,
                 confirm: bool = False, once: bool = False,
                 asof: str | None = None):
        """
        Args:
            config: 完整 config dict
            broker: 已创建的 Broker 实例(simulate/qmt/none)
            confirm: qmt 模式是否真实下单(默认 False=仅导出指令)
            once: 单次执行后退出(默认 False=盘中轮询)
            asof: 信号基准日期(默认今天;once 模式建议显式传历史日)
        """
        self.config = config
        self.broker = broker
        self.confirm = confirm
        self.once = once
        self.asof = asof or datetime.now().strftime("%Y-%m-%d")

        live_cfg = config["live"]
        self.rebalance_time = live_cfg.get("rebalance_time", "09:35")
        self.poll_interval = float(live_cfg.get("poll_interval", 5))
        self.rebalance_day = live_cfg.get("rebalance_day", "month_end")
        # 实际选股名额取自 predict 段(generate_signals 用的就是它),
        # 之前读 backtest.max_positions 只是日志里数字对得上而已
        self.top_k = int(config["predict"].get(
            "top_k", config["backtest"]["max_positions"]))
        self.order_dir = live_cfg.get("order_dir", "live/output")

        self.risk = RiskManager(live_cfg.get("risk", {}))
        self.journal = TradeJournal(output_dir=live_cfg.get(
            "order_dir", "live/output"))
        self.feed = SinaQuoteFeed()
        # 初始资金取自配置(与具体券商解耦;none 模式无 initial_cash 属性)
        self.initial_capital = float(
            config["backtest"].get("initial_capital", 1_000_000))
        # 买入单边费率:资金不足时按"含费"缩量,与回测引擎同一条算式
        mkt = config.get("market", {})
        self.fee_rate_buy = float(mkt.get("commission_rate", 0.0003)) + \
            float(mkt.get("slippage_rate", 0.001))

        self._load_model()
        self.done_rebalance_today = False
        self.done_m2m_today = False
        self.last_signal_date = None
        self.daily_values: dict[pd.Timestamp, float] = {}

    # ==================== 模型与数据加载 ====================

    def _load_model(self):
        """加载最新 checkpoint + 构建 store/预测器(约 10 秒)。"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from main import build_store, latest_checkpoint
        from data.loader import load_factor_panel, EXEC_DAILY_COLS
        from utils.market_rules import build_tradable_mask
        from models.trainer import TransformerTrainer
        from models.predictor import TransformerPredictor
        from utils.device import get_device

        self.device = get_device()
        ckpt = latest_checkpoint(self.config["model"]["save_dir"])
        if ckpt is None:
            raise RuntimeError("未找到模型 checkpoint,请先运行 "
                               "python main.py train")

        self.model, meta = TransformerTrainer.load_checkpoint(
            ckpt, self.device)
        self.panel = load_factor_panel(
            self.config["data"]["factor_panel"])
        self.store, self.daily = build_store(
            self.config, self.panel, columns=EXEC_DAILY_COLS)
        self.predictor = TransformerPredictor(
            self.model, self.store, self.config, self.device)
        logger.info(f"实时引擎就绪: checkpoint={ckpt}, "
                    f"股票池={len(self.store.symbols)}")

    # ==================== 信号 ====================

    def _signal_asof(self, ref_date: pd.Timestamp) -> str:
        """信号基准日 = ref_date 的上一交易日(因子截止该日收盘,lag-1 口径)。

        盘中模式下 ref_date=今天,今天未收盘的行情绝不进入因子。
        """
        dates = self.store.global_dates
        prev = dates[dates < np.datetime64(ref_date)]
        if len(prev) == 0:
            return str(ref_date.date())
        return str(pd.Timestamp(prev[-1]).date())

    def _target_weights(self, ref_date: pd.Timestamp) -> dict[str, float]:
        """生成目标权重 {symbol → weight}。

        严格口径:因子截止上一交易日收盘(与训练 lag-1 一致,无未来函数)。
        """
        asof = self._signal_asof(ref_date)
        preds = self.predictor.predict_asof(asof)
        # 信号日已停牌/无成交的股票不占 Top-K 名额(与回测同一份掩码口径)
        signals = self.predictor.generate_signals(
            predictions=preds,
            tradable=build_tradable_mask(
                self.daily,
                preds.index.get_level_values("date").unique()))
        self.last_signal_date = preds.index.get_level_values("date")[0]
        day = signals[signals["weight"] > 0]
        # 权重 key 取 symbol 层级(索引是 (date, symbol) MultiIndex)
        weights = dict(zip(day.index.get_level_values("symbol"),
                           day["weight"].values))
        logger.info(f"信号生成: 因子截至 {self.last_signal_date.date()}, "
                    f"目标 {len(weights)} 只, Top-K={self.top_k}")
        return weights

    # ==================== 行情与参考价 ====================

    def _watch_symbols(self, target: dict[str, float]) -> list[str]:
        """行情订阅集合 = 持仓 ∪ 目标。"""
        syms = set(self.broker.positions_dict().keys()) | set(target.keys())
        return sorted(syms)

    def _ref_prices(self, quotes: dict[str, Quote],
                    fallback: dict[str, float]) -> dict[str, float]:
        """参考价:实时行情最新价;缺失回退日线最后收盘。"""
        ref = {}
        for sym in set(quotes.keys()) | set(fallback.keys()):
            if sym in quotes and quotes[sym].price > 0:
                ref[sym] = quotes[sym].price
            elif sym in fallback:
                ref[sym] = fallback[sym]
        return ref

    def _fallback_closes(self) -> dict[str, float]:
        """日线最后收盘价(行情不可用时的参考价兜底)。"""
        closes = {}
        for sym, df in self.daily.items():
            if len(df) > 0:
                closes[sym] = float(df["收盘"].iloc[-1])
        return closes

    def _build_market_snapshot(self, trade_date,
                               quotes: dict[str, Quote],
                               target: dict[str, float]
                               ) -> dict:
        """构造 process_daily 需要的当日 bar 数据。

        持仓 ∪ 目标的每只股票:实时行情可用 → 用实时 bar;
        否则回退日线最后一条(close 用最后收盘)。
        """
        closes = self._fallback_closes()
        snapshot = {}
        for sym in set(self.broker.positions_dict().keys()) \
                | set(target.keys()):
            if sym in quotes:
                q = quotes[sym]
                snapshot[sym] = {
                    "open": q.open, "high": q.high, "low": q.low,
                    "close": q.price, "volume": q.volume,
                    "at_limit_up": at_limit_up(sym, q.change_pct),
                    "at_limit_down": at_limit_down(sym, q.change_pct),
                }
            elif sym in self.daily:
                row = self.daily[sym].iloc[-1]
                close = _num(row, "收盘")
                snapshot[sym] = {
                    # 日线缺 OHLC 时退化成"按收盘价成交"的退化 bar
                    "open": _num(row, "开盘", close),
                    "high": _num(row, "最高", close),
                    "low": _num(row, "最低", close),
                    "close": close,
                    "volume": _num(row, "成交量", 0.0),
                    "at_limit_up": at_limit_up(sym, _num(row, "涨跌幅")),
                    "at_limit_down": at_limit_down(sym, _num(row, "涨跌幅")),
                }
        return snapshot

    # ==================== 调仓 ====================

    def _rebalance(self, trade_date):
        """执行一次完整调仓。"""
        if hasattr(trade_date, "date"):
            trade_date = trade_date.date()  # 归一化为 date(去时间戳)
        logger.info(f"===== 调仓开始: {trade_date} =====")

        # 1. 信号(因子截止上一交易日,与训练 lag-1 口径一致)
        target = self._target_weights(pd.Timestamp(self.asof))

        # 2. 行情 + 参考价
        syms = self._watch_symbols(target)
        quotes = self.feed.fetch(syms)
        ref_prices = self._ref_prices(quotes, self._fallback_closes())

        # 3. 指令构造(先卖后买)
        positions = self.broker.positions_dict()
        cash = self.broker.get_cash()
        if self.broker.__class__.__name__ == "NoneBroker":
            # none 模式:用初始资金估算"从零开始"的全新建仓指令
            # (真实券商没有持仓/资金查询,这正是指令文件的用途)
            cash = self.initial_capital
        orders = make_orders(target, positions, cash, ref_prices,
                             lot_size=self.config["market"].get(
                                 "lot_size", 100),
                             max_total_pct=self.risk.max_total_pct,
                             fee_rate_buy=self.fee_rate_buy)

        # 4. 风控过滤(阻断式)
        total_value = self.broker.get_total_value()
        passed, alerts = self.risk.filter_orders(
            orders, total_value=total_value, cash=cash,
            positions=positions, quotes=quotes)

        # 5. 执行
        snapshot = self._build_market_snapshot(trade_date, quotes, target)
        is_qmt = self.broker.__class__.__name__ == "QMTBroker"
        filled = []

        for req in passed:
            result = self.broker.place_order(req)
            self.journal.log_order_result(req, result)

            if is_qmt and not self.confirm:
                continue  # 实盘安全模式:只记录不成交

            if result.status == OrderStatus.PENDING:
                filled.extend(self.broker.process_daily(trade_date, snapshot))

        # 模拟盘成交记账
        for order in filled:
            self.journal.log_trade(
                order_id=order.order_id, symbol=order.symbol,
                side=order.side.value, quantity=order.filled_quantity,
                price=order.filled_price, commission=order.commission,
                stamp_tax=order.stamp_tax, slippage=order.slippage,
                trade_date=trade_date)

        # 导出指令文件(实盘安全模式/none 模式的核心产物)
        if is_qmt and not self.confirm or \
                self.broker.__class__.__name__ == "NoneBroker":
            os.makedirs(self.order_dir, exist_ok=True)
            export_orders(passed, os.path.join(
                self.order_dir,
                f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))

        # 6. 盯市 + 快照
        self._mark_to_market(trade_date, quotes, target)
        self.done_rebalance_today = True
        logger.info(f"===== 调仓完成: 目标 {len(target)} 只, "
                    f"成交 {len(filled)} 笔 =====")

    # ==================== 盯市 ====================

    def _mark_to_market(self, trade_date, quotes=None, target=None):
        """盯市 + 净值快照(收盘后或调仓后)。"""
        if hasattr(trade_date, "date"):
            trade_date = trade_date.date()
        if target is None:
            target = {}
        snapshot_data = self._build_market_snapshot(
            trade_date, quotes or {}, target)
        self.broker.process_daily(trade_date, snapshot_data)

        cash = self.broker.get_cash()
        total = self.broker.get_total_value()
        n_pos = len(self.broker.get_positions())
        self.daily_values[pd.Timestamp(trade_date)] = total

        peak = max(self.daily_values.values())
        drawdown = 1 - total / peak if peak > 0 else 0.0
        self.journal.log_daily_snapshot(trade_date, {
            "cash": cash, "market_value": self.broker.get_market_value(),
            "total_value": total, "n_positions": n_pos,
            "pnl": total - self.initial_capital,
            "drawdown": drawdown,
        })
        logger.info(f"盯市 {trade_date}: 总资产 {total:,.0f} 元, "
                    f"现金 {cash:,.0f}, 持仓 {n_pos} 只, "
                    f"回撤 {drawdown:.1%}")

    # ==================== 主循环 ====================

    def _is_rebalance_day(self, today: pd.Timestamp) -> bool:
        """调仓日判断。

        month_end: 本月最后一个交易日(基于交易日历,盘中实时判断
                   不依赖因子面板日期,新交易日也能正确识别)
        weekly:    每周五
        """
        if self.rebalance_day == "weekly":
            return today.weekday() == 4

        from utils.calendar import get_trading_calendar
        month_end = (today + pd.offsets.MonthEnd(0)).date()
        try:
            days = get_trading_calendar(str(today.date()),
                                        str(month_end))
        except Exception as e:
            logger.warning(f"交易日历获取失败({e}),按周末判断")
            days = [d for d in pd.bdate_range(today.date(), month_end)]
        return len(days) > 0 and today.date() == days[-1]

    def run(self):
        """启动引擎(once=True 单次执行后退出;否则盘中轮询)。"""
        broker_name = self.broker.__class__.__name__
        mode = "单次执行" if self.once else f"盘中轮询({self.poll_interval}s)"
        logger.info(f"实时引擎启动: broker={broker_name}, "
                    f"confirm={self.confirm}, 模式={mode}, "
                    f"asof={self.asof}")

        self.broker.connect()

        if self.once:
            today = pd.Timestamp(self.asof)
            self._rebalance(today)
            self.broker.disconnect()
            self.journal.export_csv(
                f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(self.journal.generate_report())
            logger.info("单次执行完成")
            return

        # 盘中轮询
        try:
            while True:
                now = datetime.now()
                today = pd.Timestamp(now.date())

                # 到调仓时间且今天是调仓日且今日未调仓 → 调仓
                if (not self.done_rebalance_today
                        and now.strftime("%H:%M") >= self.rebalance_time
                        and (self.rebalance_day == "daily"
                             or self._is_rebalance_day(today))):
                    self._rebalance(now.strftime("%Y-%m-%d"))

                # 收盘后盯市
                if now.strftime("%H:%M") >= "15:05" and not self.done_m2m_today:
                    self._mark_to_market(now.date())
                    self.done_m2m_today = True

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("引擎停止(Ctrl+C)")
        finally:
            self.broker.disconnect()
            self.journal.export_csv(
                f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(self.journal.generate_report())
