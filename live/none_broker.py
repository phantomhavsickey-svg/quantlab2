"""
空券商 — 不执行任何订单(仅生成指令文件供人工检查/其他系统消费)。
"""

from loguru import logger

from live.broker import Broker, OrderRequest, OrderResult, OrderStatus


class NoneBroker(Broker):
    """none 模式: 只记录指令,不下单、不成交。"""

    def __init__(self):
        self._orders: list[OrderRequest] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def place_order(self, req: OrderRequest) -> OrderResult:
        self._orders.append(req)
        logger.info(f"[none] 指令记录: {req.side} {req.symbol} "
                    f"x{req.quantity} @ {req.ref_price}")
        return OrderResult(order_id="NONE", status=OrderStatus.PENDING,
                           message="none 模式:仅记录,不下单")

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_cash(self) -> float:
        return 0.0

    def get_total_value(self) -> float:
        return 0.0

    def get_positions(self) -> list:
        return []

    def get_pending_orders(self) -> list:
        return list(self._orders)
