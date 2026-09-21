"""
调仓数学 — 目标权重 → 买卖差额,回测与实盘共用一份实现。

为什么要单独抽出来:改之前回测是 `buy_capital = cash / len(target)` 再按
DataFrame 行序逐个买入(实盘 orders.py 同构),于是低换手的月份里新入选的
股票只分到"卖出回笼资金 / 50",而不是目标权重的市值 —— 等权只是名义上的。
这里按 **目标市值 - 现有市值** 下单,一次解决两端口径。

不做的事(留给调用方,因为那是持仓状态而非算术):
    - T+1 可卖数量限制
    - 涨跌停/停牌的成交可行性
    - 整手以外的最小交易额、资金校验后的拒单
"""

from dataclasses import dataclass, field


@dataclass
class Plan:
    """一次调仓的计划量(股数,已整手取整)。"""
    target_shares: dict = field(default_factory=dict)
    sells: dict = field(default_factory=dict)   # sym → 卖出股数
    buys: dict = field(default_factory=dict)    # sym → 买入股数
    buy_value: float = 0.0                      # 名义买入金额(费前)
    sell_value: float = 0.0

    @property
    def n_trades(self) -> int:
        return len(self.sells) + len(self.buys)


def _lot(qty: int, lot_size: int) -> int:
    return (int(qty) // lot_size) * lot_size


def rebalance_plan(weights: dict, held: dict, prices: dict,
                   total_value: float, *, lot_size: int = 100,
                   max_total_pct: float = 1.0,
                   fee_rate_buy: float = 0.0,
                   fee_rate_sell: float = 0.0) -> Plan:
    """按目标市值与现有市值的差额生成买卖量。

    Args:
        weights: {symbol: 目标权重},权重 > 0 即应持有;缺席即应清仓
        held: {symbol: 当前股数}
        prices: {symbol: 成交价}(回测传次日开盘价,实盘传实时价)。
                缺价的股票保持原持仓不动 —— 没有价格就没法下单
        total_value: 调仓时点总资产(现金 + 持仓市值),按同一价格基准
        lot_size: 一手股数
        max_total_pct: 投资总额上限(0.95 = 留 5% 现金缓冲)
        fee_rate_buy / fee_rate_sell: 单边费率,仅用于买入预算的可行性缩放

    Returns:
        Plan
    """
    held = {s: int(q) for s, q in held.items() if int(q) > 0}
    w = {str(s): float(v) for s, v in weights.items()
         if v is not None and float(v) > 0}
    # 只对有价格的名字分配权重;无价名字保留现有持仓,不参与再分配
    priced = {s: v for s, v in w.items() if _valid_price(prices, s)}
    sw = sum(priced.values())
    plan = Plan()
    if sw <= 0 or total_value <= 0:
        plan.sells = {}
        plan.buys = {}
        return plan

    investable = total_value * min(max(max_total_pct, 0.0), 1.0)
    tgt = {}
    for s, v in priced.items():
        tgt[s] = _lot(investable * (v / sw) / float(prices[s]), lot_size)
    # 目标集合里缺席、但有价格的名字 → 清仓
    for s, q in held.items():
        if s not in w and _valid_price(prices, s):
            tgt[s] = 0
    plan.target_shares = tgt

    for s, q in held.items():
        t = tgt.get(s, q)          # 无价名字 → 目标 = 现有,不动
        if t < q:
            plan.sells[s] = q - t
            plan.sell_value += (q - t) * float(prices[s])
    for s, t in tgt.items():
        q = held.get(s, 0)
        if t > q:
            plan.buys[s] = t - q
            plan.buy_value += (t - q) * float(prices[s])

    return plan


def scale_buys_to_budget(plan: Plan, cash: float, prices: dict, *,
                         lot_size: int = 100,
                         fee_rate_buy: float = 0.0) -> Plan:
    """买入总额超过可用资金时按比例缩量(不改变相对权重)。

    旧实现是"按 DataFrame 行序买到没钱为止",排在后面的股票被系统性欠配;
    按比例缩让所有目标名字同等承担缺口,与下单顺序无关。

    Args:
        plan: rebalance_plan 的结果
        cash: 可动用资金(实盘=账户可用,回测=卖出回笼后的现金)
        prices: {symbol: 成交价},与 rebalance_plan 同一基准
        lot_size: 一手股数
        fee_rate_buy: 买入单边费率
    """
    if not plan.buys or cash <= 0:
        return plan
    cost = sum(q * float(prices[s]) * (1 + fee_rate_buy)
               for s, q in plan.buys.items())
    if cost <= cash:
        return plan
    k = cash / cost
    buys, value = {}, 0.0
    for s, q in plan.buys.items():
        qty = _lot(q * k, lot_size)
        if qty > 0:
            buys[s] = qty
            value += qty * float(prices[s])
    plan.buys = buys
    plan.buy_value = value
    return plan


def _valid_price(prices: dict, sym: str) -> bool:
    p = prices.get(sym)
    return p is not None and p == p and float(p) > 0  # 非 None/NaN/<=0
