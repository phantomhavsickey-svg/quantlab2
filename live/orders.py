"""
指令构造 — 目标/分数 + 持仓 + 现金 + 参考价 → 中立下单指令。

输入/输出都与具体券商无关,是信号层 → 执行层的唯一契约。
"目标权重 → 股数"这一步与回测共用 utils/sizing.rebalance_plan,
两端不会再出现"回测按目标建仓、实盘按现金摊派"的口径分叉。

两条路径共用同一段"Plan → 指令"的收尾逻辑:
    make_orders  等权 / 信号强度加权(直接给目标权重)
    plan_orders  分数带位策略(建仓线、补仓档、单票上限、减仓价)
策略模式下指令只负责下单,状态推进统一由
utils.position_policy.apply_fills 在成交回报之后做 —— 与回测引擎同一个提交点。
"""

from loguru import logger

from live.broker import OrderRequest, OrderSide
from utils.position_policy import plan as policy_plan
from utils.position_policy import sync_intent_shares
from utils.sizing import rebalance_plan, scale_buys_to_budget


def _position_maps(positions: dict, ref_price: dict):
    """持仓对象(键统一成字符串) / 股数 / 有效参考价 / 持仓市值。

    参考价缺失时用券商给的对象市值价兜底估总资产 —— 那只股票本来也下不了单,
    但它的钱不能从预算里凭空消失。
    """
    positions = {str(s): p for s, p in (positions or {}).items()}
    held = {s: int(getattr(p, "shares", 0) or 0) for s, p in positions.items()}
    prices = {str(s): float(p) for s, p in (ref_price or {}).items()
              if p is not None and float(p) > 0}
    mv = sum(q * prices.get(
        s, float(getattr(positions[s], "market_price", 0.0) or 0.0))
        for s, q in held.items())
    return positions, held, prices, mv


def _to_requests(plan, positions, held, prices, cash, lot_size,
                 fee_rate_buy, intents=None):
    """Plan → 指令列表(先卖后买)。给了 intents 就把现金缩量回写进意图。"""
    orders: list[OrderRequest] = []

    # --- 先卖:清仓或减到目标权重(受 T+1 可卖数量约束;清仓允许零股) ---
    proceeds = 0.0
    for sym in sorted(plan.sells):
        pos = positions.get(sym)
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
    if intents:
        # 现金缩量是"这一档只能买这么多",按缩量后的股数推进状态。不回写的话,
        # 补仓每轮都被判成未足量成交、参考分数永不推进、仓位反复补到上限。
        sync_intent_shares(intents, plan)
    for sym in sorted(plan.buys):
        qty = plan.buys[sym]
        if qty <= 0:
            continue
        orders.append(OrderRequest(
            symbol=sym, side=OrderSide.BUY.value, quantity=qty,
            ref_price=prices[sym]))
    return orders


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

    positions, held, prices, mv = _position_maps(positions, ref_price)
    total_value = cash + mv
    plan = rebalance_plan(target, held, prices, total_value,
                          lot_size=lot_size, max_total_pct=max_total_pct)
    orders = _to_requests(plan, positions, held, prices, cash, lot_size,
                          fee_rate_buy)

    logger.info(f"指令构造完成: {sum(1 for o in orders if o.side == 'sell')} 卖 "
                f"+ {sum(1 for o in orders if o.side == 'buy')} 买 "
                f"(目标 {len(target)} 只, 总资产 {total_value:,.0f})")
    return orders


def plan_orders(scores: dict[str, float],
                positions: dict[str, object],
                cash: float,
                ref_price: dict[str, float],
                states: dict,
                policy,
                *,
                lot_size: int = 100,
                asof=None,
                max_total_pct_override: float | None = None,
                entry_allowed: bool = True):
    """分数带位策略版指令构造(与回测引擎走同一个 utils.position_policy.plan)。

    Args:
        scores: {symbol: 分数},**全截面**(低于建仓线的候选也要在里面)
        states: {symbol: NameState} 就地不变;成交回报后由 apply_fills 推进
        policy: utils.position_policy.PolicyConfig
        max_total_pct_override / entry_allowed: 组合级暴露层(utils/exposure.py)
                的结果,与回测引擎同一口径;不传 = 纯分数带

    Returns:
        (orders, PolicyPlan) —— 撮合回报到手后调用
        apply_fills(states, pol.intents, before, after, fill_price, asof)
        再 save_states 落盘。
    """
    positions, held, prices, mv = _position_maps(positions, ref_price)
    total_value = cash + mv
    pol = policy_plan(scores or {}, held, prices, states, total_value,
                      policy, lot_size=lot_size, asof=asof,
                      max_total_pct_override=max_total_pct_override,
                      entry_allowed=entry_allowed)
    orders = _to_requests(pol.plan, positions, held, prices, cash, lot_size,
                          policy.fee_rate_buy, pol.intents)
    n_names = len([w for w in pol.weights.values() if w > 0])
    cap = (policy.max_total_pct if max_total_pct_override is None
           else min(float(max_total_pct_override), policy.max_total_pct))
    logger.info(
        f"策略指令: {sum(1 for o in orders if o.side == 'sell')} 卖 + "
        f"{sum(1 for o in orders if o.side == 'buy')} 买,动作 {pol.actions or '无'},"
        f" 目标 {n_names} 只 / 仓位 {pol.gross_weight:.1%}(本轮上限 {cap:.1%}"
        f"{'' if entry_allowed else ',禁新仓'}),"
        f" 总资产 {total_value:,.0f}")
    return orders, pol


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
