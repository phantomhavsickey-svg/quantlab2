"""
实时交易引擎 — 盘中轮询行情 + 定时调仓 + 风控 + 多券商执行。

流程:
    启动(交易日盘中前台运行,或 --once 单次执行):
    ├─ 加载最新 checkpoint + SequenceStore + 预测器
    ├─ 主循环(每 poll_interval 秒):
    │    ├─ 拉取 持仓 ∪ 候选 股票实时行情
    │    ├─ 到调仓时间(默认 09:35)且今日是调仓日(默认月末)且今日未调仓:
    │    │    信号(因子截止上一交易日,lag-1 口径)
    │    │    ├─ 等权模式: Top-K → 目标权重 → make_orders
    │    │    └─ position_policy.enabled: 全截面分数 → plan_orders
    │    │         (建仓线/补仓档/单票上限/减仓价,与回测同一份状态机)
    │    │    → 风控过滤 → 按后端执行 → apply_fills 提交策略状态并落盘
    │    │    simulate: 下单 + 当日 bar 撮合 + 状态落盘
    │    │    qmt:      --confirm 才真实下单,否则仅导出指令 CSV
    │    │    none:     仅导出指令 CSV
    │    └─ 收盘后(15:05)盯市 + 净值快照
    └─ Ctrl+C 退出

--once 模式:单次完整执行(信号→指令→风控→执行→盯市)后退出,
任何一天都可运行(测试/计划任务用),不检查调仓日。

策略状态(每只票的建仓分数/参考分数/减仓价)落在
live/state/policy_state.json,重启后继续按同一档结算 —— 补仓/减仓的触发
是"相对上一次动作的分数变化",没有状态就等于每轮从零开始。
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from live.broker import Broker, OrderSide, OrderStatus
from live.journal import TradeJournal
from live.orders import export_orders, make_orders, plan_orders
from live.risk import RiskManager
from live.__init__ import SinaQuoteFeed, Quote
from utils.market_rules import at_limit_up, at_limit_down
from utils.position_policy import (apply_fills, load_states,
                                   policy_from_config, save_states)


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

        # --- 分数带位策略:与回测引擎共用 utils/position_policy ---
        self.policy = policy_from_config(config)
        self.policy_enabled = self.policy is not None
        self.policy_state_path = os.path.join(
            live_cfg.get("state_dir", "live/state"), "policy_state.json")
        self.states = load_states(self.policy_state_path) \
            if self.policy_enabled else {}
        if self.policy_enabled:
            logger.info(f"仓位策略已启用: 建仓线 {self.policy.buy_score:.4f} / "
                        f"清仓线 {self.policy.sell_score:.4f},单票 "
                        f"{self.policy.base_weight:.0%}→"
                        f"{self.policy.max_position_weight:.0%},新仓上限 "
                        f"{self.policy.max_names} 只,总仓位上限 "
                        f"{self.policy.max_total_pct:.0%};已恢复 "
                        f"{len(self.states)} 只持仓的策略状态")

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

    def _policy_scores(self, ref_date: pd.Timestamp) -> dict[str, float]:
        """生成**全截面**分数 {symbol → score}(策略要的是分数,不是 Top-K)。

        口径与等权路径一致:因子截止上一交易日收盘(无未来函数),信号日没有
        成交的股票直接摘掉 —— 不建仓也不因"分数消失"被误清仓(那只在回测里
        同样是缺席,由 keep 兜住)。
        """
        asof = self._signal_asof(ref_date)
        preds = self.predictor.predict_asof(asof)
        self.last_signal_date = preds.index.get_level_values("date")[0]
        sig_dates = preds.index.get_level_values("date").unique()
        tradable = build_tradable_mask(self.daily, sig_dates)
        allowed = set(tradable[tradable].index.get_level_values("symbol"))
        preds = preds[[s in allowed
                       for s in preds.index.get_level_values("symbol")]]
        scores = {str(s): float(x) for s, x in preds.items() if x == x}
        n_buy = sum(1 for x in scores.values()
                    if x >= self.policy.buy_score)
        logger.info(f"策略分数: 因子截至 {self.last_signal_date.date()}, "
                    f"全截面 {len(scores)} 只,过建仓线 {n_buy} 只")
        return scores

    # ==================== 行情与参考价 ====================

    def _watch_symbols(self, target: dict[str, float]) -> list[str]:
        """行情订阅集合 = 持仓 ∪ 候选。

        策略模式下 target 是全截面分数(几千只),逐秒轮询这个集合既打满新浪
        接口又没必要:进仓按分数降序排队,能真正建仓的最多 max_names 只,取
        过线的 2×max_names 名就够(剩下的名额本来也进不去)。
        """
        syms = {str(s) for s in self.broker.positions_dict().keys()}
        if self.policy is None:
            syms |= {str(s) for s in target}
            return sorted(syms)
        cand = {str(s): x for s, x in target.items()
                if x >= self.policy.buy_score}
        limit = 2 * self.policy.max_names
        syms |= sorted(cand, key=lambda s: (-cand[s], s))[:limit]
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
        #    基准日 = **本次撮合日**,不是进程启动那天写死的 self.asof —— 盘中
        #    轮询跨过零点后还拿启动日的因子选股,就是拿两天前的信号下单。
        ref = pd.Timestamp(trade_date)
        target = (self._policy_scores(ref) if self.policy_enabled
                  else self._target_weights(ref))

        # 2. 行情 + 参考价
        syms = self._watch_symbols(target)
        quotes = self.feed.fetch(syms)
        closes = self._fallback_closes()
        held = {str(s) for s in self.broker.positions_dict().keys()}
        # 日线兜底价只给持仓用(算市值、卖单要有价):没盯盘的候选拿不到今日价,
        # 就不该按昨天的收盘价去建仓 —— 缺价的候选由策略判成"本轮不碰"。
        ref_prices = self._ref_prices(
            quotes, {s: c for s, c in closes.items() if s in held})

        # 3. 指令构造(先卖后买)
        positions = self.broker.positions_dict()
        cash = self.broker.get_cash()
        if self.broker.__class__.__name__ == "NoneBroker":
            # none 模式:用初始资金估算"从零开始"的全新建仓指令
            # (真实券商没有持仓/资金查询,这正是指令文件的用途)
            cash = self.initial_capital
        before = {str(s): int(getattr(p, "shares", 0) or 0)
                  for s, p in positions.items()}
        lot = int(self.config["market"].get("lot_size", 100))
        pol = None
        if self.policy is None:
            orders = make_orders(target, positions, cash, ref_prices,
                                 lot_size=lot,
                                 max_total_pct=self.risk.max_total_pct,
                                 fee_rate_buy=self.fee_rate_buy)
        else:
            orders, pol = plan_orders(target, positions, cash, ref_prices,
                                      self.states, self.policy,
                                      lot_size=lot, asof=trade_date)
            for s, why in sorted(pol.notes.items()):
                # "为什么今天没单"必须能从日志里直接回答,而不是让人去猜策略状态
                logger.debug(f"未动作 {s}: {why}")

        # 4. 风控过滤(阻断式)
        total_value = self.broker.get_total_value()
        passed, alerts = self.risk.filter_orders(
            orders, total_value=total_value, cash=cash,
            positions=positions, quotes=quotes)

        # 5. 执行(watch = 本轮真正可能成交的名字:持仓 ∪ 候选)
        watch = dict.fromkeys(syms, 1.0)
        snapshot = self._build_market_snapshot(trade_date, quotes, watch)
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

        # 5.5 策略状态提交:只认**实际股数变化**。风控拦下、券商拒单、部分成交
        # 都不推进参考分数,下一轮同一档条件仍成立会自动补做(与回测同一提交点)。
        if pol is not None:
            after = {str(s): int(getattr(p, "shares", 0) or 0)
                     for s, p in self.broker.positions_dict().items()}
            fills = {str(o.symbol): float(o.filled_price) for o in filled}
            for s, msg in apply_fills(self.states, pol.intents, before,
                                      after, fills, asof=trade_date).items():
                logger.debug(f"策略状态 {s}: {msg}")
            save_states(self.policy_state_path, self.states)

        # 导出指令文件(实盘安全模式/none 模式的核心产物)
        if is_qmt and not self.confirm or \
                self.broker.__class__.__name__ == "NoneBroker":
            os.makedirs(self.order_dir, exist_ok=True)
            export_orders(passed, os.path.join(
                self.order_dir,
                f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))

        # 6. 盯市 + 快照
        self._mark_to_market(trade_date, quotes, watch)
        self.done_rebalance_today = True
        n_target = (len(self.states) if self.policy_enabled
                    else len(self.broker.positions_dict()))
        logger.info(f"===== 调仓完成: 持仓/在建 {n_target} 只, "
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

        daily:     每个交易日都评估(分数带位策略的推荐节奏:补仓/减仓的触发
                   是分数相对变化,等月末会把 20 日窗口内攒出的档位全丢掉)
        month_end: 本月最后一个交易日(基于交易日历,盘中实时判断
                   不依赖因子面板日期,新交易日也能正确识别)
        weekly:    每周五
        """
        if self.rebalance_day == "daily":
            return True
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
