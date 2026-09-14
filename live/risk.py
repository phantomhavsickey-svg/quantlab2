"""
风控 — 下单前阻断式检查(quantlab 是"告警式",quantlab2 升级为阻断式)。

每条指令在下单前逐条检查,违反 → 该订单跳过并记录告警,继续其余订单。
默认阻断(block=true),可配置关闭为仅告警。
"""

from dataclasses import dataclass

from loguru import logger

from live.broker import OrderRequest, OrderSide


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20      # 单股最大仓位(预估成交后)
    max_total_pct: float = 0.95         # 总仓位上限(现金留底)
    max_positions: int = 50             # 最大持仓数
    block: bool = True                  # True=违反跳过订单;False=仅告警


class RiskManager:
    """下单前风控检查。

    规则:
        1. 卖出数量 ≤ available_shares(T+1 可卖)
        2. 买入金额 ≤ 可用资金
        3. 涨停不买 / 跌停不卖(行情提供时)
        4. 停牌跳过(volume<=0,行情提供时)
        5. 单股仓位 ≤ max_position_pct(预估成交后市值)
        6. 持仓数 ≤ max_positions
        7. 总仓位 ≤ max_total_pct
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.max_position_pct = cfg.get("max_position_pct", 0.20)
        self.max_total_pct = cfg.get("max_total_pct", 0.95)
        self.max_positions = cfg.get("max_positions", 50)
        self.block = cfg.get("block", True)

    def check_order(self, req: OrderRequest, *, total_value: float,
                    cash: float, positions: dict, quotes: dict | None = None
                    ) -> tuple[bool, str]:
        """检查单条指令。

        Args:
            req: 待检查指令
            total_value: 当前总资产
            cash: 当前可用资金
            positions: {symbol: PositionInfo}
            quotes: {symbol: Quote}(可选,用于涨跌停/停牌判断)

        Returns:
            (通过?, 说明)
        """
        sym = req.symbol
        price = req.ref_price or 0.0

        # 1. 卖出可用检查
        if req.side == OrderSide.SELL.value:
            pos = positions.get(sym)
            avail = pos.available_shares if pos else 0
            if req.quantity > avail:
                return False, f"卖出 {sym}: 数量({req.quantity})超过可卖({avail})"

        # 2. 涨停不买 / 跌停不卖
        if quotes and sym in quotes:
            q = quotes[sym]
            if req.side == OrderSide.BUY.value and q.change_pct >= 9.5:
                return False, f"买入 {sym}: 涨停({q.change_pct:+.1f}%)不可买"
            if req.side == OrderSide.SELL.value and q.change_pct <= -9.5:
                return False, f"卖出 {sym}: 跌停({q.change_pct:+.1f}%)不可卖"
            if q.volume <= 0:
                return False, f"{sym}: 停牌/无成交,跳过"

        # 3. 买入资金检查
        if req.side == OrderSide.BUY.value:
            amount = price * req.quantity
            if amount > cash:
                return False, (f"买入 {sym}: 金额({amount:,.0f})超过"
                               f"可用资金({cash:,.0f})")

            # 单股仓位(预估成交后)
            if total_value > 0:
                pct = (amount + (positions.get(sym).market_price
                                 * positions.get(sym).shares
                                 if sym in positions else 0)) / total_value
                if pct > self.max_position_pct:
                    return False, (f"买入 {sym}: 预估仓位 {pct:.1%} "
                                   f"超限({self.max_position_pct:.0%})")

        # 4. 持仓数限制(新增持仓时)
        if req.side == OrderSide.BUY.value and sym not in positions:
            if len(positions) >= self.max_positions:
                return False, (f"买入 {sym}: 持仓数已达上限 "
                               f"({self.max_positions})")

        # 5. 总仓位上限(买入后)
        if req.side == OrderSide.BUY.value and total_value > 0:
            pos_value = sum(p.shares * p.market_price
                            for p in positions.values())
            after = (pos_value + price * req.quantity) / total_value
            if after > self.max_total_pct:
                return False, (f"买入 {sym}: 总仓位 {after:.1%} "
                               f"超限({self.max_total_pct:.0%})")

        return True, "ok"

    def filter_orders(self, orders: list[OrderRequest], *,
                      total_value: float, cash: float, positions: dict,
                      quotes: dict | None = None
                      ) -> tuple[list[OrderRequest], list[str]]:
        """批量过滤指令,返回 (通过列表, 告警列表)。"""
        passed, alerts = [], []
        for req in orders:
            ok, msg = self.check_order(
                req, total_value=total_value, cash=cash,
                positions=positions, quotes=quotes)
            if ok:
                passed.append(req)
            else:
                alerts.append(msg)
                if self.block:
                    logger.warning(f"[风控阻断] {msg}")
                else:
                    logger.warning(f"[风控告警] {msg}")
        return passed, alerts
