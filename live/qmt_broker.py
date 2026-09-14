"""
QMT 券商实盘适配器(miniQMT/迅投 XtQuant)— 移植自 quantlab live/qmt_broker.py,
适配统一 Broker 接口。

安全设计:
    - xtquant 依赖可选:未安装时 connect() 抛出带安装说明的 RuntimeError
    - mini_qmt_path/account_id 为空 → 拒绝启动真实柜台
    - place_order 实际下 FIX_PRICE 限价单(ref_price 为下单价,滑点保护)
    - 真实下单前引擎层必须通过 --confirm 确认(见 engine.py)
"""

import os
from loguru import logger

from live.broker import (
    Broker, OrderRequest, OrderResult, PositionInfo, OrderStatus,
)

try:
    from xtquant import xttrader
    from xtquant.xttype import StockAccount
    from xtquant.xtconstant import STOCK_BUY, STOCK_SELL, FIX_PRICE
    HAS_XTQUANT = True
except ImportError:
    HAS_XTQUANT = False


class QMTBroker(Broker):
    """QMT 真实柜台适配器。

    依赖: pip install xtquant(仅实盘需要,安装说明见 README)
    需要本机运行 miniQMT 客户端(国金QMT等),路径与资金账号在 config 配置。
    """

    def __init__(self, qmt_config: dict):
        if not HAS_XTQUANT:
            raise RuntimeError(
                "xtquant 未安装,无法启动 QMT 实盘接口。\n"
                "安装: pip install xtquant\n"
                "或使用模拟盘: python main.py live --broker simulate")

        self.mini_qmt_path = qmt_config.get("mini_qmt_path", "")
        self.account_id = qmt_config.get("account_id", "")
        if not self.mini_qmt_path or not self.account_id:
            raise RuntimeError(
                "QMT 实盘配置不完整: 请设置 config.yaml 的 "
                "live.qmt.mini_qmt_path(如 D:\\国金QMT\\userdata_mini) "
                "与 live.qmt.account_id(资金账号)")

        self.xt_trader = xttrader.XtQuantTrader(self.mini_qmt_path, 1)
        self.account = StockAccount(self.account_id)
        self.cash = 0.0
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        self.xt_trader.start()
        connect_result = self.xt_trader.connect()
        if connect_result != 0:
            raise RuntimeError(
                f"QMT 连接失败(错误码 {connect_result}): "
                f"请确认 miniQMT 客户端已登录并开启交易服务")
        self._connected = True
        logger.info(f"QMT 已连接: 账号 {self.account_id}")

    def disconnect(self) -> None:
        if self._connected:
            self.xt_trader.stop()
            self._connected = False
            logger.info("QMT 已断开")

    # ==================== 交易 ====================

    @staticmethod
    def _with_exchange(symbol: str) -> str:
        """6 位代码 → 带交易所后缀: 600519 -> 600519.SH。"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith(("60", "68")):
            return f"{symbol}.SH"
        elif symbol.startswith(("4", "8")):
            return f"{symbol}.BJ"
        else:
            return f"{symbol}.SZ"

    def place_order(self, req: OrderRequest) -> OrderResult:
        """下单(FIX_PRICE 限价单,ref_price 为限价,滑点保护)。

        注意:QMT 只接受限价单;无 ref_price 时以当前市价兜底并警告。
        """
        self.connect()
        code = self._with_exchange(req.symbol)
        price = req.ref_price or self._get_latest_price(req.symbol)
        if not price or price <= 0:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message=f"无法获取 {req.symbol} 参考价")

        stock_type = STOCK_BUY if req.side == "buy" else STOCK_SELL
        order_id = self.xt_trader.order_stock(
            self.account, code, stock_type, req.quantity,
            FIX_PRICE, price, "quantlab2", "rebalance")

        if order_id < 0:
            return OrderResult(order_id="", status=OrderStatus.REJECTED,
                               message=f"QMT 下单失败(错误码 {order_id})")
        logger.info(f"QMT 下单成功: {order_id} {req.side} {code} "
                    f"x{req.quantity} @ {price}")
        return OrderResult(order_id=str(order_id),
                           status=OrderStatus.PENDING,
                           message="已提交 QMT,待成交")

    def _get_latest_price(self, symbol: str) -> float:
        """从行情查询接口获取最新价(失败返回 0)。"""
        try:
            code = self._with_exchange(symbol)
            quote = self.xt_trader.query_stock_quote(self.account, code)
            if quote is not None:
                return float(getattr(quote, "lastPrice", 0) or 0)
        except Exception as e:
            logger.warning(f"QMT 行情查询失败 {symbol}: {e}")
        return 0.0

    def cancel_order(self, order_id: str) -> bool:
        self.connect()
        try:
            return self.xt_trader.cancel_order_stock(
                self.account, int(order_id)) == 0
        except Exception as e:
            logger.warning(f"QMT 撤单失败 {order_id}: {e}")
            return False

    # ==================== 查询 ====================

    def get_cash(self) -> float:
        try:
            asset = self.xt_trader.query_stock_asset(self.account)
            if asset is not None:
                self.cash = float(getattr(asset, "cash", 0) or 0)
        except Exception as e:
            logger.warning(f"QMT 资金查询失败: {e}(使用缓存 {self.cash:,.0f})")
        return self.cash

    def get_total_value(self) -> float:
        try:
            asset = self.xt_trader.query_stock_asset(self.account)
            if asset is not None:
                return float(getattr(asset, "total_asset", 0) or 0)
        except Exception as e:
            logger.warning(f"QMT 资产查询失败: {e}")
        return self.get_cash() + self.get_market_value()

    def get_positions(self) -> list[PositionInfo]:
        try:
            raw = self.xt_trader.query_stock_positions(self.account)
        except Exception as e:
            logger.warning(f"QMT 持仓查询失败: {e}")
            return []

        infos = []
        for p in raw or []:
            sym = str(getattr(p, "stock_code", "")).split(".")[0]
            if not sym:
                continue
            shares = int(getattr(p, "volume", 0) or 0)
            avail = int(getattr(p, "can_use_volume", shares) or shares)
            cost = float(getattr(p, "open_price", 0) or 0)
            price = float(getattr(p, "market_value", 0) or 0)
            if shares > 0:
                price = price / shares
            infos.append(PositionInfo(
                symbol=sym, shares=shares, available_shares=avail,
                locked_shares=max(shares - avail, 0),
                avg_cost=cost, market_price=price,
            ))
        return infos

    def get_pending_orders(self) -> list:
        return []  # 实盘成交回报以柜台为准,engine 不做 pending 管理
