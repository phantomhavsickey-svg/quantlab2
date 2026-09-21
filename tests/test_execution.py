"""执行层(选股→指令→成交→盯市)的口径测试。

每条测试只钉住一处修复,数据全部合成且刻意把"停牌/缺行/涨跌停/低换手"
摆在最容易踩到的位置。不需要 torch、不需要真实缓存,所以也不需要重训。
"""

import numpy as np
import pandas as pd
import pytest

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from live.orders import make_orders
from models.predictor import signals_from_predictions
from utils.market_rules import (at_limit_down, at_limit_up, can_fill,
                                get_limit_pct, build_tradable_mask)
from utils.sizing import rebalance_plan, scale_buys_to_budget

CAP = 1_000_000
DATES = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=110))
BUY_FEE = TransactionCostModel().effective_cost_rate("buy")


# ==================== 合成数据 ====================

def daily_frame(closes, vols=None, chgs=None, opens=None):
    n = len(closes)
    o = list(closes) if opens is None else list(opens)
    return pd.DataFrame({
        "日期": DATES[:n],
        "开盘": o,
        "收盘": list(closes),
        "最高": [x * 1.01 for x in o],
        "最低": [x * 0.99 for x in o],
        "成交量": [1e6] * n if vols is None else list(vols),
        "涨跌幅": [0.0] * n if chgs is None else list(chgs),
    })


def flat(level):
    return [float(level)] * len(DATES)


def idx_of(d):
    return list(DATES).index(pd.Timestamp(d))


def signals_from(weights_by_date):
    rows = []
    for d, ws in weights_by_date.items():
        for s, w in ws.items():
            rows.append({"date": d, "symbol": s, "weight": w, "score": w})
    df = pd.DataFrame(rows).set_index(["date", "symbol"])
    df.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(0)),
         df.index.get_level_values(1).astype(object)],
        names=["date", "symbol"])
    return df


def rebal_dates():
    """与引擎同口径:每月最后一个交易日。"""
    df = pd.DataFrame({"date": DATES})
    df["ym"] = df["date"].dt.strftime("%Y-%m")
    return df.groupby("ym")["date"].last().sort_values().tolist()


REBAL = rebal_dates()

# 每个调仓日的成交日 = 次一交易日;月末是数据末端时没有次日
EXEC_DAYS = [d + pd.offsets.BDay(1) for d in REBAL
             if (d + pd.offsets.BDay(1)) in set(DATES)]


def flag_on_exec_days(value):
    """在每一个成交日上打同一个涨跌幅标记(一字涨停/跌停天天如此)。"""
    chgs = flat(0.0)
    for d in EXEC_DAYS:
        chgs[idx_of(d)] = value
    return chgs


def run_bt(weights_by_date, data, freq="monthly", top_k=None):
    sig = signals_from(weights_by_date)
    if top_k is None:
        first = next(iter(weights_by_date.values()))
        top_k = len(first)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency=freq,
                         max_positions=top_k, cost_model=TransactionCostModel())
    return eng.run(data, sig)


# ==================== sizing ====================

def test_rebalance_plan_targets_equal_weight():
    weights = {f"60000{i}": 0.1 for i in range(10)}
    prices = {f"60000{i}": 10.0 for i in range(10)}
    plan = rebalance_plan(weights, {}, prices, CAP)
    assert sum(plan.buys.values()) * 10.0 == pytest.approx(CAP, rel=2e-3)
    assert all(q % 100 == 0 for q in plan.buys.values())
    assert not plan.sells


def test_rebalance_plan_trims_overweight_and_tops_up_underweight():
    weights = {"600001": 0.5, "600002": 0.5}
    held = {"600001": 100_000, "600002": 25_000}    # 100万 vs 25万
    prices = {"600001": 10.0, "600002": 10.0}
    plan = rebalance_plan(weights, held, prices, total_value=1_250_000)
    assert plan.target_shares == {"600001": 62_500, "600002": 62_500}
    assert plan.sells == {"600001": 37_500}
    assert plan.buys == {"600002": 37_500}


def test_rebalance_plan_leaves_unpriced_names_alone():
    plan = rebalance_plan({"600001": 0.5, "600002": 0.5},
                          {"600009": 1000}, {"600001": 10.0, "600002": 10.0},
                          total_value=100_000)
    assert "600009" not in plan.sells and "600009" not in plan.target_shares


def test_scale_buys_to_budget_is_order_independent():
    """钱不够时等比缩量,结果与字典顺序无关(旧实现按行序买到没钱)。"""
    prices = {f"60000{i}": 10.0 for i in range(10)}
    weights = {s: 0.1 for s in prices}
    a = rebalance_plan(weights, {}, prices, CAP)
    b = rebalance_plan(dict(reversed(list(weights.items()))), {},
                       dict(reversed(list(prices.items()))), CAP)
    scale_buys_to_budget(a, 300_000.0, prices, fee_rate_buy=BUY_FEE)
    scale_buys_to_budget(b, 300_000.0, prices, fee_rate_buy=BUY_FEE)
    assert sorted(a.buys.items()) == sorted(b.buys.items())
    assert sum(q * 10.0 for q in a.buys.values()) * (1 + BUY_FEE) <= 300_000.0


# ==================== market_rules ====================

def test_limit_pct_by_board():
    """创业板/科创板 20%,主板 10% —— 替掉实盘里写死的 9.5。"""
    assert get_limit_pct("600001") == 0.10
    assert get_limit_pct("000001") == 0.10
    assert get_limit_pct("300750") == 0.20
    assert get_limit_pct("688981") == 0.20
    assert at_limit_up("300750", 9.9) is False       # 主板的涨幅在创业板不是涨停
    assert at_limit_up("600001", 9.9) is True
    assert at_limit_down("688981", -12.0) is False
    assert at_limit_down("688981", -19.9) is True


def test_can_fill_blocks_no_bar_zero_volume_and_limit():
    assert can_fill(None, "600001", "buy")[0] is False
    assert can_fill(pd.Series({"收盘": 10.0, "成交量": 0.0, "涨跌幅": 0.0}),
                    "600001", "buy")[0] is False
    ok, why = can_fill(pd.Series({"收盘": 10.0, "成交量": 1e6, "涨跌幅": 10.0}),
                       "600001", "sell")
    assert ok and why == "ok"
    ok, why = can_fill(pd.Series({"收盘": 10.0, "成交量": 1e6, "涨跌幅": 10.0}),
                       "600001", "buy")
    assert not ok and "涨停" in why


# ==================== tradable 接线 ====================

def test_tradable_mask_drops_no_volume_days():
    df = daily_frame(flat(10))
    df.loc[3, "成交量"] = 0.0
    mask = build_tradable_mask({"600001": df}, DATES[:5])
    seen = set(mask.index.get_level_values("date"))
    assert DATES[3] not in seen
    assert seen == {DATES[0], DATES[1], DATES[2], DATES[4]}


def test_suspended_names_do_not_consume_top_k_slots():
    d0, d1 = DATES[0], REBAL[0]
    preds = pd.Series(
        [5.0, 4.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [(d0, "600001"), (d0, "600002"), (d0, "600003")],
            names=["date", "symbol"]))
    tradable = pd.Series(
        True,
        index=pd.MultiIndex.from_tuples(
            [(d0, "600001"), (d0, "600003")], names=["date", "symbol"]))
    held = signals_from_predictions(preds, top_k=2,
                                   tradable=tradable)
    held = held[held["weight"] > 0]
    assert set(held.index.get_level_values("symbol")) == {"600001", "600003"}


# ==================== 回测引擎 ====================

def test_first_rebalance_is_equal_weight_and_fully_invested():
    names = [f"60000{i}" for i in range(10)]
    data = {s: daily_frame(flat(10.0 + i)) for i, s in enumerate(names)}
    res = run_bt({d: {s: 0.1 for s in names} for d in REBAL}, data)
    tr = res["trades"]
    buys = tr[tr["date"] == tr["date"].min()]
    buys = buys[buys["side"] == "buy"]
    assert len(buys) == 10
    amounts = buys["amount"].to_numpy()
    assert amounts.max() / amounts.min() < 1.35
    assert res["execution"]["n_underinvested_rebalances"] == 0


def test_low_turnover_rebalance_still_targets_equal_weight():
    """钉住 F0:换 1 只时新入选的要拿到约 1/N 仓位。

    旧实现 buy_capital = cash/len(target):低换手月份只回笼 1/N 的钱,
    于是新名字被建成 (1/N)² 的仓位 —— 这个塌陷在这里被直接测出来。
    """
    names = [f"60000{i}" for i in range(10)]
    data = {s: daily_frame(flat(10.0)) for s in names}
    data["301010"] = daily_frame(flat(10.0))
    w1 = {s: 0.1 for s in names}
    w2 = {s: 0.1 for s in names[:-1]}
    w2["301010"] = 0.1
    res = run_bt({REBAL[0]: w1, REBAL[1]: w2, REBAL[2]: w2}, data)
    tr = res["trades"]
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    new_buy = tr[(tr["date"] == exec2) & (tr["symbol"] == "301010")]
    assert len(new_buy) == 1, "新入选股票没被建仓"
    equity = float(res["equity_curve"].iloc[0]) * CAP
    amt = float(new_buy["amount"].iloc[0])
    assert 0.07 * equity < amt < 0.13 * equity, (amt, equity)


def test_limit_up_blocks_the_buy_and_leaves_cash():
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)),
            b: daily_frame(flat(10.0), chgs=flag_on_exec_days(10.0))}
    res = run_bt({d: {a: 0.5, b: 0.5} for d in REBAL}, data)
    assert (res["trades"]["symbol"] == b).sum() == 0, "涨停股不该成交"
    assert res["execution"]["blocked_buy"].get("涨停 +10.00%", 0) >= 1
    tail = res["marks"]
    assert float(tail["cash"].iloc[-1]) > 0.4 * CAP, "买不进的钱要留在现金里"
    assert res["execution"]["n_underinvested_rebalances"] >= 1


def test_limit_down_keeps_the_position_valued():
    a, b = "600001", "600002"
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    exec3 = REBAL[2] + pd.offsets.BDay(1)
    chgs = flat(0.0)
    chgs[idx_of(exec2)] = -10.0
    data = {a: daily_frame(flat(10.0), chgs=chgs), b: daily_frame(flat(10.0))}
    keep, drop = {a: 0.5, b: 0.5}, {b: 1.0}
    res = run_bt({REBAL[0]: keep, REBAL[1]: drop, REBAL[2]: drop}, data)
    tr = res["trades"]
    sold_a = tr[(tr["symbol"] == a) & (tr["side"] == "sell")]
    assert exec2 not in set(sold_a["date"]), "跌停日卖不出去"
    assert set(sold_a["date"]) == {exec3}, "解除跌停后的下一次调仓该卖掉"
    assert res["execution"]["blocked_sell"].get("跌停 -10.00%") == 1
    pos = res["positions"]
    assert (pos["symbol"] == a).sum() > 0, "卖不掉的持仓要继续计在净值里"


def test_missing_bars_do_not_zero_out_market_value():
    """钉住 F1:已建仓的 b 断档 6 个交易日,市值沿用最近有效收盘而不是归零。"""
    a, b = "600001", "600002"
    gap = daily_frame(flat(20.0))
    cut = idx_of(REBAL[0]) + 2          # 成交日(REBAL[0]+1)之后的 6 天没有日线
    gap = gap.drop(index=range(cut, cut + 6)).reset_index(drop=True)
    data = {a: daily_frame(flat(10.0)), b: gap}
    res = run_bt({d: {a: 0.5, b: 0.5} for d in REBAL}, data)
    window = res["marks"].loc[DATES[cut]: DATES[cut + 5]]
    assert len(window) == 6
    mv = window["market_value"]
    assert mv.min() > 0.99 * mv.max(), mv.to_dict()
    assert (window["n_positions"] == 2).all()
    assert mv.iloc[0] > 0.97 * CAP


def test_daily_marks_use_the_portfolio_actually_held_that_month():
    """钉住前视修复:下一个月末才决定的组合不能给这一个月估值。

    A 全期不动,B 每天 +1%。第一个月只持 A,第二个月末换成 B。
    旧实现先更新持仓再回补上一区间 → 第一个月的净值里就冒出 B 的涨幅。
    """
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)),
            b: daily_frame([10.0 * 1.01 ** i for i in range(len(DATES))])}
    res = run_bt({REBAL[0]: {a: 1.0}, REBAL[1]: {b: 1.0},
                  REBAL[2]: {b: 1.0}}, data)
    eq = res["marks"]
    exec1 = REBAL[0] + pd.offsets.BDay(1)
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    month1 = eq.loc[exec1: exec2 - pd.offsets.BDay(1)]
    assert len(month1) > 5
    tv = month1["total_value"].to_numpy()
    assert np.allclose(tv, tv[0]), "第一个月净值里混进了第二个月的持仓变动"
    after = eq.loc[exec2:]
    assert after["total_value"].iloc[-1] > tv[0], "换到 B 之后才该吃到涨幅"


def test_no_same_day_round_trip_under_daily_rebalancing():
    """daily 调仓 + 目标每天换:任何一只股票都不能同一天既买又卖。"""
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)), b: daily_frame(flat(10.0))}
    weights = {d: ({a: 1.0} if i % 2 == 0 else {b: 1.0})
               for i, d in enumerate(DATES[:-1])}
    res = run_bt(weights, data, freq="daily")
    tr = res["trades"]
    per_day = tr.groupby(["symbol", "date"])["side"].apply(set)
    assert not any("buy" in s and "sell" in s for s in per_day), per_day
    assert len(tr) > 0


def test_execution_diagnostics_report_holding_period():
    names = [f"60000{i}" for i in range(5)]
    data = {s: daily_frame(flat(10.0)) for s in names}
    first = {s: 0.2 for s in names}
    later = {s: 0.25 for s in names[:-1]}      # 第二次换仓:踢掉最后一只
    res = run_bt({d: (first if d == REBAL[0] else later) for d in REBAL}, data)
    ex = res["execution"]
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    tr = res["trades"]
    assert ex["n_exit_trades"] == 1
    assert 15 <= ex["median_hold_days"] <= 30, ex
    assert ex["n_blocked_buy"] == 0 and ex["n_blocked_sell"] == 0
    # 目标稳定下来之后不该再有任何成交(首轮建仓把 mean_turnover 摊薄,不能直接看它)
    assert (tr["date"] > exec2).sum() == 0, tr[tr["date"] > exec2]
    assert ex["n_underinvested_rebalances"] == 0


def test_backtest_and_live_size_the_same_plan():
    """钉住两端一致:同一份权重/价格/资产,回测与实盘给出同样的股数。"""
    names = [f"60000{i}" for i in range(5)]
    prices = {s: 10.0 for s in names}
    weights = {s: 0.2 for s in names}
    data = {s: daily_frame(flat(10.0)) for s in names}
    res = run_bt({REBAL[0]: weights}, data)
    exec1 = REBAL[0] + pd.offsets.BDay(1)
    bt = res["trades"]
    bt = bt[bt["date"] == exec1].set_index("symbol")["shares"].to_dict()
    assert len(bt) == 5

    class Flat:
        shares = 0
        available_shares = 0

    lv = {o.symbol: o.quantity for o in
          make_orders(weights, {}, float(CAP), prices, fee_rate_buy=BUY_FEE)}
    assert bt == lv, (bt, lv)
    assert sum(q * 10.0 for q in lv.values()) > 0.97 * CAP
