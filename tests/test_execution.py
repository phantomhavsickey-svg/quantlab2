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
from utils.position_policy import PolicyConfig
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


def run_bt(weights_by_date, data, freq="monthly", top_k=None, policy=None):
    sig = signals_from(weights_by_date)
    if top_k is None:
        first = next(iter(weights_by_date.values()))
        top_k = len(first)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency=freq,
                         max_positions=top_k, cost_model=TransactionCostModel(),
                         policy=policy)
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


# ==================== 分数带位策略 × 回测引擎 ====================

def cum_shares(trades, sym):
    """{成交日: 该票当日收盘时的累计股数}。"""
    out, t = {}, 0
    rows = trades[trades["symbol"] == sym].sort_values("date")
    for _, r in rows.iterrows():
        t += r["shares"] if r["side"] == "buy" else -r["shares"]
        out[r["date"]] = t
    return out


def test_policy_ramps_to_the_cap_then_stops_trading():
    """建仓 5% → 补两档到 15% → 顶到 16% 上限 → 分数不变就彻底不动。

    这里钉的是"同一档只触发一次":如果缩量/挡单被判成已成交,参考分数会假
    推进、仓位停在半档;如果未成交不推进状态,第 4 轮还会再买一次。
    """
    a = "600001"
    data = {a: daily_frame(flat(10.0))}
    scores = {d: {a: 0.02 if i == 0 else 0.05} for i, d in enumerate(REBAL)}
    res = run_bt(scores, data, policy=PolicyConfig(min_trade_value=5000))
    cum = cum_shares(res["trades"], a)
    assert list(cum) == EXEC_DAYS[:3], cum            # 最后一次评估没有成交
    assert cum[EXEC_DAYS[0]] == 5000                  # 建仓线 → 5%
    assert 14900 <= cum[EXEC_DAYS[1]] <= 15000        # 两档补仓 → 15%
    assert 15900 <= cum[EXEC_DAYS[2]] <= 16000        # 压到单票上限 16%
    st = res["policy_states"][a]
    assert (st.entry_score, st.adds) == (0.02, 2)
    # 第 3 轮只走掉一档的量(15%→16% 被单票上限截断),但档位按请求的 2 档一起
    # 结清:仓位已经贴顶,把没吃到上限的那一档留着记账只会让它每轮重复挂单。
    assert st.ref_score == pytest.approx(0.04)
    assert res["execution"]["policy_actions"] == {"entry": 1, "add": 2}


def test_policy_blocked_entry_retries_every_round():
    """涨停买不进 → 状态不落地、不留"已建仓"的假记录,下一轮继续重试。"""
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)),
            b: daily_frame(flat(10.0), chgs=flag_on_exec_days(10.0))}
    scores = {d: {a: 0.02, b: 0.02} for d in REBAL}
    res = run_bt(scores, data, policy=PolicyConfig())
    tr = res["trades"]
    assert (tr["symbol"] == b).sum() == 0
    assert b not in res["policy_states"] and a in res["policy_states"]
    assert res["execution"]["blocked_buy"]["涨停 +10.00%"] == len(EXEC_DAYS)


def test_policy_entry_budget_is_fee_aware_and_capped():
    """20 个高分候选:含费预算只放得下 11 只,总仓位不越 95%,之后不再换手。"""
    names = [f"6000{i:02d}" for i in range(20)]
    data = {s: daily_frame(flat(10.0)) for s in names}
    scores = {d: {s: 0.05 for s in names} for d in REBAL}
    res = run_bt(scores, data, policy=PolicyConfig())
    ex = res["execution"]
    assert ex["policy_actions"] == {"entry": 11}       # 0.88 含费,第 12 挤不进
    assert ex["policy_final_names"] == 11
    m = res["marks"]
    assert (m["market_value"] / m["total_value"]).max() < 0.96
    assert 0.85 < ex["policy_mean_gross_weight"] < 0.90
    # 第 2 轮起分数没变 → 一分钱都不该再动(补仓预留额度没被吃掉也不会乱补)
    assert set(res["trades"]["date"]) == {EXEC_DAYS[0]}


def test_policy_exit_below_the_sell_line_clears_the_state():
    a = "600001"
    data = {a: daily_frame(flat(10.0))}
    scores = {REBAL[0]: {a: 0.03}, REBAL[1]: {a: -0.01},
              REBAL[2]: {a: -0.01}}
    res = run_bt(scores, data, policy=PolicyConfig())
    tr = res["trades"]
    assert cum_shares(tr, a)[EXEC_DAYS[1]] == 0        # 跌破清仓线 → 全清
    assert res["policy_states"] == {}                  # 真止损,不锁价格
    assert res["execution"]["policy_actions"] == {"entry": 1, "exit": 1}
    assert float(res["marks"]["market_value"].iloc[-1]) == pytest.approx(0.0)


def test_live_and_backtest_issue_the_same_policy_orders():
    """两端一致(策略版):同一份分数/持仓/现金/价格 → 同一张指令单。

    回测引擎消费 position_policy.plan 的 Plan,实盘消费 plan_orders 包住的同一个
    Plan。这条测试钉的就是这个接缝不会分叉。
    """
    from live.orders import plan_orders
    from utils.position_policy import NameState, PolicyConfig
    from utils.position_policy import plan as policy_plan

    pc = PolicyConfig()
    scores = {"600001": 0.05, "600002": 0.02}     # 补两档 / 刚过建仓线
    prices = {"600001": 10.0, "600002": 10.0}
    states = {"600001": NameState(entry_score=0.03, ref_score=0.03, step=0.01)}
    held = {"600001": 5000}

    class P:
        def __init__(self, shares):
            self.shares, self.available_shares = shares, shares
            self.market_price = 10.0

    live_orders, pol = plan_orders(scores, {"600001": P(5000)}, 950_000.0,
                                   prices, dict(states), pc, asof=DATES[0])
    bt = policy_plan(scores, held, prices, dict(states), 1_000_000.0, pc,
                     asof=DATES[0])
    assert {o.symbol: o.quantity for o in live_orders
            if o.side == "buy"} == bt.plan.buys == {"600001": 10000,
                                                     "600002": 5000}
    assert not [o for o in live_orders if o.side == "sell"] and not bt.plan.sells
    assert set(pol.intents) == set(bt.intents) == {"600001", "600002"}


def test_policy_state_commits_only_after_a_real_fill(tmp_path):
    """跌停卖不掉 → 状态不推进;次日成交后按**实际成交价**记减仓价。"""
    from live.orders import plan_orders
    from live.simulate_broker import Position, SimulateBroker
    from utils.position_policy import NameState, PolicyConfig, apply_fills

    pc = PolicyConfig()
    states = {"600001": NameState(entry_score=0.03, ref_score=0.03, step=0.01)}
    br = SimulateBroker(initial_cash=850_000.0, lot_size=100, t_plus_1=True,
                        state_path=str(tmp_path / "simulate_state.json"))
    br.positions["600001"] = Position(symbol="600001", shares=15000,
                                      available_shares=15000, avg_cost=10.0,
                                      market_price=10.0)
    d1, d2 = DATES[0], DATES[1]
    orders, pol = plan_orders({"600001": 0.01}, br.positions_dict(),
                              br.get_cash(), {"600001": 10.0}, states, pc,
                              asof=d1)
    assert [(o.side, o.quantity) for o in orders] == [("sell", 10000)]
    before = {"600001": 15000}
    for o in orders:
        br.place_order(o)

    def bar(o, down):
        return {"600001": {"open": o, "high": o + 0.2, "low": o - 0.2,
                           "close": o, "volume": 1e6, "at_limit_up": False,
                           "at_limit_down": down}}

    assert br.process_daily(d1, bar(9.0, True)) == []       # 跌停卖不掉
    rep = apply_fills(states, pol.intents, before,
                      {s: p.shares for s, p in br.positions.items()}, {},
                      asof=d1)
    assert "未足量成交" in rep["600001"]
    assert (states["600001"].ref_score, states["600001"].trim_price) == (0.03, None)

    filled = br.process_daily(d2, bar(9.6, False))
    assert [(f.filled_quantity, f.filled_price) for f in filled] == [(10000, 9.6)]
    apply_fills(states, pol.intents, before,
                {s: p.shares for s, p in br.positions.items()},
                {"600001": 9.6}, asof=d2)
    st = states["600001"]
    assert st.ref_score == pytest.approx(0.01)               # 两档一起结清
    assert (st.trim_price, st.trim_date) == (9.6, str(d2.date()))
