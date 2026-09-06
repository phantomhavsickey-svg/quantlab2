"""
统一券商接口 — Broker 抽象基类 + 中立数据结构 + 工厂。

设计目标(吸取 quantlab 的教训):
    - quantlab 没有公共基类,靠鸭子类型约定(place_market_order 返回类型
      不一致、QMTBroker 缺 get_cash/process_daily、journal 直接摸 broker
      内部属性),导致模拟盘/实盘切换脆弱。
    - quantlab2 用正式 ABC 统一方法签名与返回类型;新增券商只需继承
      Broker 实现方法并在 create_broker 注册一行。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


# ==================== 枚举与数据结构 ====================

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderRequest:
    """中立下单指令(信号层 → 执行层的唯一契约)。

    ref_price: 参考价(模拟盘: 撮合/资金估算参考;实盘: FIX_PRICE 限价
               的下单价,即滑点保护价)
    """
    symbol: str
    side: str                    # "buy" / "sell"
    quantity: int                # 目标股数(自动向下取整到整手)
    ref_price: float | None = None
    order_type: str = "market"   # "market" / "limit"


@dataclass
class OrderResult:
    """下单结果(所有券商统一返回)。"""
    order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    message: str = ""


@dataclass
class PositionInfo:
    """持仓信息(9 字段统一超集,模拟盘全量,实盘尽量填全)。"""
    symbol: str
    shares: int = 0
    available_shares: int = 0    # 可卖股数(T+1 锁仓不计入)
    locked_shares: int = 0       # 锁仓股数(当日买入)
    avg_cost: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    name: str = ""


# ==================== 抽象基类 ====================

class Broker(ABC):
    """统一券商接口。

    模拟盘(SimulateBroker)与实盘(QMTBroker)必须实现全部方法;
    实盘不需要日撮合(process_daily 实现为 no-op)。
    """

    @abstractmethod
    def connect(self) -> None:
        """连接券商(模拟盘 no-op)。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接(模拟盘 no-op)。"""

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult:
        """提交订单。"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销待成交订单。"""

    @abstractmethod
    def get_cash(self) -> float:
        """可用资金。"""

    @abstractmethod
    def get_total_value(self) -> float:
        """总资产 = 现金 + 持仓市值。"""

    @abstractmethod
    def get_positions(self) -> list[PositionInfo]:
        """当前持仓列表。"""

    def process_daily(self, trade_date, market_data: dict) -> list:
        """每日撮合/盯市。

        模拟盘: 按当日 bar 撮合待成交订单 + 更新市价。
        实盘:   no-op(成交回报以柜台为准,返回空列表)。
        """
        return []

    # ---- 便捷方法(基于抽象方法的默认实现,无需子类覆写) ----

    def get_market_value(self) -> float:
        """持仓总市值。"""
        return sum(p.shares * p.market_price for p in self.get_positions())

    def positions_dict(self) -> dict[str, PositionInfo]:
        """symbol → PositionInfo 字典。"""
        return {p.symbol: p for p in self.get_positions()}

    def get_pending_orders(self) -> list:
        """待成交订单(默认无;模拟盘覆写)。"""
        return []


# ==================== 工厂 ====================

def create_broker(kind: str, config: dict, **kwargs) -> Broker:
    """按类型创建券商实例。

    Args:
        kind: "simulate" / "qmt" / "none"
        config: 完整 config dict(取 market/live 段)
        kwargs: 透传给具体券商构造(如 initial_cash)

    Returns:
        Broker 实例
    """
    kind = (kind or config["live"]["broker"]).lower()
    market_cfg = config["market"]
    live_cfg = config["live"]

    if kind == "simulate":
        from live.simulate_broker import SimulateBroker
        return SimulateBroker(
            initial_cash=kwargs.get("initial_cash",
                                    config["backtest"]["initial_capital"]),
            commission_rate=market_cfg["commission_rate"],
            min_commission=market_cfg["min_commission"],
            stamp_tax_rate=market_cfg["stamp_tax_rate"],
            slippage_rate=market_cfg["slippage_rate"],
            lot_size=market_cfg.get("lot_size", 100),
            t_plus_1=market_cfg.get("t_plus_1", True),
            state_path=kwargs.get("state_path",
                                  live_cfg.get("state_dir", "live/state")
                                  + "/simulate_state.json"),
        )
    elif kind == "qmt":
        from live.qmt_broker import QMTBroker
        return QMTBroker(live_cfg["qmt"])
    elif kind == "none":
        from live.none_broker import NoneBroker
        return NoneBroker()
    else:
        raise ValueError(f"未知券商类型: {kind} (可选 simulate/qmt/none)")
