"""
指令构造 — 目标权重 + 持仓 + 现金 + 参考价 → 中立下单指令。

输入/输出都与具体券商无关,是信号层 → 执行层的唯一契约。
"目标权重 → 股数"这一步与回测共用 utils/sizing.rebalance_plan,
两端不会再出现"回测按目标建仓、实盘按现金摊派"的口径分叉。
"""

from loguru import logger

from live.broker import OrderRequest, OrderSide
from utils.sizing import rebalance_plan, scale_buys_to_budget


def make_orders(target_weights: dict[str, float],
                positions: dict[str, object],
                cash: float,
                ref_price: dict[str, float],
                lot_size: int = 100,
                max_total_pct: float = 1.0,
                fee_rate_buy: float = 0.0) -> list[OrderRequest]:
    """从目标权重生成买卖指令(先卖后买,下目标市值与现有市值的差额)。

    Args:
        target_weights: {symbol: 目标权重},权重 > 0 即应持有,缺席即应清仓
        positions: {symbol: 持仓对象},需有 shares / available_shares
        cash: 当前可用资金
        ref_price: {symbol: 参考价}(缺价的股票保留原状,不下单)
        lot_size: 一手股数
        max_total_pct: 投资总额上限(取风控的 max_total_pct,让"超限被拒单"
                       变成"一开始就按这个上限算量")
        fee_rate_buy: 买入单边费率(佣金+滑点)。资金不足时按"含费"缩量,
                      与回测引擎同一条算式,否则两端会差出最后一手

    Returns:
        list[OrderRequest]: 卖出在前、买入在后;买入总额已按可用资金等比缩量
    """
    target = {s: w for s, w in (target_weights or {}).items()
              if w is not None and w > 0}
    if not target and not positions:
        return []

    held = {s: int(getattr(p, "shares", 0) or 0) for s, p in positions.items()}
    prices = {s: float(p) for s, p in (ref_price or {}).items()
              if p is not None and float(p) > 0}
    mv = sum(q * prices.get(
        s, float(getattr(positions[s], "market_price", 0.0) or 0.0))
        for s, q in held.items())
    total_value = cash + mv

    plan = rebalance_plan(target, held, prices, total_value,
                          lot_size=lot_size, max_total_pct=max_total_pct)

    orders: list[OrderRequest] = []

    # --- 先卖:清仓或减到目标权重(受 T+1 可卖数量约束;清仓允许零股) ---
    proceeds = 0.0
    for sym in sorted(plan.sells):
        pos = positions[sym]
        avail = int(getattr(pos, "available_shares", 0) or 0)
        want = min(plan.sells[sym], held.get(sym, 0))
        qty = min(want, avail)
        if qty < held.get(sym, 0):     # 只减不清仓 → 按整手卖,零股留着
            qty = (qty // lot_size) * lot_size
        if qty <= 0:
            if plan.sells[sym] > 0:
                logger.debug(f"卖出 {sym}: 可卖 {avail} 不足 1 手,跳过")
            continue
        proceeds += qty * prices[sym]
        orders.append(OrderRequest(
            symbol=sym, side=OrderSide.SELL.value, quantity=qty,
            ref_price=prices[sym]))

    # --- 后买:建仓或加到目标权重(受可用资金约束) ---
    scale_buys_to_budget(plan, cash + proceeds, prices, lot_size=lot_size,
                         fee_rate_buy=fee_rate_buy)
    for sym in sorted(plan.buys):
        qty = plan.buys[sym]
        if qty <= 0:
            continue
        orders.append(OrderRequest(
            symbol=sym, side=OrderSide.BUY.value, quantity=qty,
            ref_price=prices[sym]))

    logger.info(f"指令构造完成: {sum(1 for o in orders if o.side=='sell')} 卖 "
                f"+ {sum(1 for o in orders if o.side=='buy')} 买 "
                f"(目标 {len(target)} 只, 总资产 {total_value:,.0f})")
    return orders


def export_orders(orders: list[OrderRequest], path: str):
    """导出指令为 CSV(实盘安全模式/干跑检查用)。"""
    import os
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "symbol", "side", "quantity", "ref_price", "order_type"])
        writer.writeheader()
        for o in orders:
            writer.writerow({
                "symbol": o.symbol, "side": o.side,
                "quantity": o.quantity,
                "ref_price": o.ref_price if o.ref_price else "",
                "order_type": o.order_type,
            })
    logger.info(f"指令已导出: {path} ({len(orders)} 条)")
