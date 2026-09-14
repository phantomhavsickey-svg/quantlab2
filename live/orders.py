"""
指令构造 — 目标权重 + 持仓 + 现金 + 参考价 → 中立下单指令。

与 quantlab daily_signal.make_orders 同语义:
    输入/输出都与具体券商无关,是信号层 → 执行层的唯一契约。
"""

from loguru import logger

from live.broker import OrderRequest, OrderSide


def make_orders(target_weights: dict[str, float],
                positions: dict[str, object],
                cash: float,
                ref_price: dict[str, float],
                lot_size: int = 100) -> list[OrderRequest]:
    """从目标权重生成买卖指令(先卖后买)。

    Args:
        target_weights: {symbol: 目标权重},权重 > 0 即应持有
        positions: {symbol: 持仓对象},需有 shares / available_shares
        cash: 可用资金
        ref_price: {symbol: 参考价}(缺省 symbol 时跳过买入)
        lot_size: 一手股数

    Returns:
        list[OrderRequest]:
            - 卖出: 持仓中不在目标集合(或目标权重为 0)的,全部卖出可用股数
            - 买入: 目标集合中的,按等权分配 cash/len(target) 买入
              (目标权重已由 signals_from_predictions 归一化,此处按
               目标数量等分现金)
    """
    target = {s: w for s, w in target_weights.items() if w > 0}
    held = set(positions.keys())
    orders: list[OrderRequest] = []

    # --- 先卖:不在目标组合中的持仓全部卖出 ---
    for sym in sorted(held - set(target.keys())):
        pos = positions[sym]
        avail = getattr(pos, "available_shares", 0) \
            if hasattr(pos, "available_shares") else 0
        if avail <= 0:
            continue
        orders.append(OrderRequest(
            symbol=sym, side=OrderSide.SELL.value, quantity=avail,
            ref_price=ref_price.get(sym)))

    # --- 后买:目标组合中未持有(或持有不足)的按等权买入 ---
    if target:
        buy_budget = cash / len(target)
        for sym in sorted(target.keys()):
            price = ref_price.get(sym)
            if not price or price <= 0:
                logger.warning(f"买入 {sym}: 无参考价,跳过")
                continue
            qty = int(buy_budget / price)
            qty = (qty // lot_size) * lot_size  # 整手
            if qty <= 0:
                logger.debug(f"买入 {sym}: 资金不足1手,跳过")
                continue
            orders.append(OrderRequest(
                symbol=sym, side=OrderSide.BUY.value, quantity=qty,
                ref_price=price))

    logger.info(f"指令构造完成: {sum(1 for o in orders if o.side=='sell')} 卖 "
                f"+ {sum(1 for o in orders if o.side=='buy')} 买")
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
