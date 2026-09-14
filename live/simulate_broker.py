"""
模拟券商 — 模拟 A 股交易执行(移植自 quantlab paper_trade/broker.py 并改进)。

改进点:
    1. 继承统一 Broker 接口(place_order 接受 OrderRequest、返回 OrderResult)
    2. T+1 规则可配置(t_plus_1 参数,读取 config market.t_plus_1)
    3. 佣金/印花税/滑点复用 backtest/cost.py 的 TransactionCostModel
    4. 状态持久化内置到构造参数 state_path(自动 save_state 于每次成交后)
"""

import json
import os
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from backtest.cost import TransactionCostModel
from live.broker import (
    Broker, OrderRequest, OrderResult, PositionInfo,
    OrderSide, OrderType, OrderStatus,
)


@dataclass
class Order:
    """内部订单对象(比 OrderResult 更详细,模拟盘撮合用)。"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_date: date | None = None
    filled_date: date | None = None
    commission: float = 0.0
    stamp_tax: float = 0.0
    slippage: float = 0.0
    notes: str = ""


@dataclass
class Position:
    """内部持仓对象。"""
    symbol: str
    shares: int = 0
    available_shares: int = 0
    avg_cost: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    locked_shares: int = 0


class SimulateBroker(Broker):
    """模拟券商 — T+1、整手、涨跌停不可交易、状态持久化。"""

    def __init__(self,
                 initial_cash: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 min_commission: float = 5.0,
                 stamp_tax_rate: float = 0.0005,
                 slippage_rate: float = 0.001,
                 lot_size: int = 100,
                 t_plus_1: bool = True,
                 state_path: str = "live/state/simulate_state.json"):
        self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.lot_size = lot_size
        self.t_plus_1 = t_plus_1
        self.state_path = state_path
        self.cost = TransactionCostModel(
            commission_rate, min_commission, stamp_tax_rate, slippage_rate)

        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.order_counter = 0
        self.trade_date: date | None = None
        self._last_unlock_date: date | None = None  # T+1 解锁幂等保护

        self.load_state()

    # ==================== Broker 接口 ====================

    def connect(self) -> None:
        pass  # 模拟盘无连接

    def disconnect(self) -> None:
        self.save_state()

    def place_order(self, req: OrderRequest) -> OrderResult:
        """提交订单(自动向下取整到整手;卖空/资金检查)。"""
        self.order_counter += 1
        order_id = f"ORD{self.order_counter:06d}"
        side = OrderSide(req.side)

        # 整手取整
        quantity = (req.quantity // self.lot_size) * self.lot_size

        def reject(msg: str) -> OrderResult:
            self.orders.append(Order(
                order_id=order_id, symbol=req.symbol, side=side,
                quantity=quantity, order_type=OrderType(req.order_type),
                limit_price=req.ref_price,
                status=OrderStatus.REJECTED, notes=msg))
            logger.warning(f"拒单 {req.symbol} {req.side}: {msg}")
            return OrderResult(order_id=order_id,
                               status=OrderStatus.REJECTED, message=msg)

        if quantity <= 0:
            return reject("数量不足1手")

        if side == OrderSide.SELL:
            pos = self.positions.get(req.symbol)
            avail = pos.available_shares if pos else 0
            if avail < quantity:
                return reject(f"可卖数量不足(需要{quantity},可用{avail})")

        order = Order(
            order_id=order_id, symbol=req.symbol, side=side,
            quantity=quantity, order_type=OrderType(req.order_type),
            limit_price=req.ref_price,
            created_date=self.trade_date)
        self.orders.append(order)
        logger.info(f"订单已提交: {order_id} {req.side} {req.symbol} "
                    f"x{quantity} @ {req.ref_price or '市价'}")
        return OrderResult(order_id=order_id, status=OrderStatus.PENDING,
                           message="已提交,待撮合")

    def cancel_order(self, order_id: str) -> bool:
        for order in self.orders:
            if order.order_id == order_id:
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    order.notes = "用户撤单"
                    logger.info(f"订单已撤销: {order_id}")
                    return True
                logger.warning(f"订单 {order_id} 状态为 {order.status},无法撤销")
                return False
        logger.warning(f"未找到订单: {order_id}")
        return False

    def get_cash(self) -> float:
        return self.cash

    def get_total_value(self) -> float:
        return self.cash + self.get_market_value()

    def get_positions(self) -> list[PositionInfo]:
        return [PositionInfo(
            symbol=p.symbol, shares=p.shares,
            available_shares=p.available_shares,
            locked_shares=p.locked_shares,
            avg_cost=p.avg_cost, market_price=p.market_price,
            unrealized_pnl=p.unrealized_pnl,
            pnl_pct=(p.market_price / p.avg_cost - 1) * 100
            if p.avg_cost > 0 else 0.0,
        ) for p in self.positions.values()]

    def get_pending_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status == OrderStatus.PENDING]

    # ==================== 每日撮合 ====================

    def process_daily(self, trade_date, market_data: dict) -> list[Order]:
        """按当日 bar 撮合待成交订单 + 更新市价。

        market_data: {symbol: {open, high, low, close, volume,
                               at_limit_up, at_limit_down}}
        规则:
            - T+1 到期解锁(昨日买入今日可卖)
            - 涨停不可买、跌停不可卖、停牌(volume<=0)跳过
            - 市价单以当日开盘价成交;限价单触价成交
            - 买入资金不足 → 减一手重试,仍不足留在 PENDING
        """
        self.trade_date = trade_date
        filled_today = []

        # Step 1: T+1 解锁(同一天多次 process_daily 只解锁一次:
        # engine 调仓与盯市会各调一次,重复解锁会破坏 T+1)
        if self.t_plus_1 and trade_date != self._last_unlock_date:
            for pos in self.positions.values():
                pos.available_shares += pos.locked_shares
                pos.locked_shares = 0
            self._last_unlock_date = trade_date

        # Step 2: 撮合待成交订单
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            if order.symbol not in market_data:
                continue

            bar = market_data[order.symbol]

            if order.side == OrderSide.BUY and bar.get("at_limit_up", False):
                continue  # 涨停买不到
            if order.side == OrderSide.SELL and bar.get("at_limit_down", False):
                continue  # 跌停卖不掉
            if bar.get("volume", 0) <= 0:
                continue  # 停牌

            # --- 撮合 ---
            fill_price = None
            if order.order_type == OrderType.MARKET:
                fill_price = bar["open"]
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY:
                    if bar["low"] <= order.limit_price:
                        fill_price = order.limit_price
                else:
                    if bar["high"] >= order.limit_price:
                        fill_price = order.limit_price

            if fill_price is None or fill_price <= 0:
                continue

            amount = fill_price * order.quantity
            total_cost = self.cost.total_cost(amount, order.side.value)

            if order.side == OrderSide.BUY:
                total_deduction = amount + total_cost
                if total_deduction > self.cash:
                    # 资金不足:减一手重试
                    reduced_qty = order.quantity - self.lot_size
                    if reduced_qty <= 0:
                        continue
                    order.quantity = reduced_qty
                    amount = fill_price * order.quantity
                    total_cost = self.cost.total_cost(
                        amount, order.side.value)
                    total_deduction = amount + total_cost
                    if total_deduction > self.cash:
                        continue

                self.cash -= total_deduction

                pos = self.positions.get(order.symbol)
                if pos is None:
                    pos = Position(symbol=order.symbol)
                    self.positions[order.symbol] = pos
                total_shares = pos.shares + order.quantity
                pos.avg_cost = ((pos.avg_cost * pos.shares)
                                + (fill_price * order.quantity)) / total_shares \
                    if total_shares > 0 else 0.0
                pos.shares = total_shares
                if self.t_plus_1:
                    pos.locked_shares += order.quantity  # T+1 锁定
                else:
                    pos.available_shares += order.quantity
                pos.market_price = fill_price

            else:  # SELL
                pos = self.positions.get(order.symbol)
                if pos is None:
                    continue
                self.cash += amount - total_cost
                pos.shares -= order.quantity
                pos.available_shares -= order.quantity
                if pos.shares <= 0:
                    del self.positions[order.symbol]

            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.filled_date = trade_date
            order.commission = self.cost.commission(amount)
            order.stamp_tax = self.cost.stamp_tax(amount, order.side.value)
            order.slippage = self.cost.slippage(amount)
            order.status = OrderStatus.FILLED
            filled_today.append(order)

        # Step 3: 更新市价
        for sym, pos in self.positions.items():
            if sym in market_data:
                pos.market_price = market_data[sym]["close"]
                pos.unrealized_pnl = (pos.market_price - pos.avg_cost) \
                    * pos.shares

        # 成交后自动落盘(盘中进程被杀不丢状态)
        self.save_state()
        return filled_today

    # ==================== 状态持久化 ====================

    def save_state(self, path: str | None = None):
        """保存持仓与现金状态。"""
        path = path or self.state_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "trade_date": str(self.trade_date) if self.trade_date else None,
            "positions": {
                s: {"shares": p.shares,
                    "available_shares": p.available_shares,
                    "locked_shares": p.locked_shares,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price}
                for s, p in self.positions.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"持仓状态已保存: {path}")

    def load_state(self, path: str | None = None):
        """从状态文件恢复(损坏/缺失回退初始资金并告警)。"""
        path = path or self.state_path
        if not os.path.exists(path):
            logger.info(f"状态文件不存在: {path},使用初始资金 "
                        f"{self.initial_cash:,.0f}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.cash = float(state.get("cash", self.initial_cash))
            self.initial_cash = float(
                state.get("initial_cash", self.initial_cash))
            self.positions = {}
            for s, p in state.get("positions", {}).items():
                self.positions[s] = Position(
                    symbol=s,
                    shares=int(p.get("shares", 0)),
                    available_shares=int(
                        p.get("available_shares", p.get("shares", 0))),
                    avg_cost=float(p.get("avg_cost", 0.0)),
                    market_price=float(p.get("market_price", 0.0)),
                    locked_shares=int(p.get("locked_shares", 0)),
                )
            logger.info(f"持仓状态已恢复: {len(self.positions)} 只持仓, "
                        f"现金 {self.cash:,.0f}")
        except Exception as e:
            logger.error(f"状态文件损坏({e}),回退初始资金 "
                         f"{self.initial_cash:,.0f}")
            self.cash = self.initial_cash
            self.positions = {}
