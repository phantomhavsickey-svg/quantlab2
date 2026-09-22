"""分数带位策略(建仓 / 补仓 / 减仓 / 清仓)的行为测试。

输入全是合成的,不需要 torch、不需要预测缓存。数值都按"总资产 100 万、
统一价 10 元"手算:5% = 5,000 股,8% = 8,000 股,16% = 16,000 股。
"""

import numpy as np
import pandas as pd
import pytest

from utils.position_policy import (NameState, PolicyConfig, apply_fills,
                                   decide, enforce_min_trade_value, load_states,
                                   plan, save_states)
from utils.sizing import Plan, rebalance_plan

CAP = 1_000_000.0
P = 10.0
D0 = pd.Timestamp("2024-01-31")
A, B, C, D = "600001", "600002", "600003", "600004"


def cfg(**kw) -> PolicyConfig:
    return PolicyConfig(**kw)


def st(entry=0.03, ref=None, step=None, trim_price=None, trim_date=None,
       adds=0, trims=0) -> NameState:
    return NameState(entry_score=entry,
                     ref_score=entry if ref is None else ref,
                     step=cfg().step_of(entry) if step is None else step,
                     trim_price=trim_price,
                     trim_date=str(trim_date)[:10] if trim_date else None,
                     adds=adds, trims=trims)


def run(scores: dict, held: dict | None = None, prices: dict | None = None,
        states: dict | None = None, tv: float = CAP, asof=D0, **kw):
    """跑一次策略评估,返回 PolicyPlan。缺省所有股票都是 10 元。"""
    names = set(scores) | set(held or {})
    px = {s: P for s in names} if prices is None else prices
    return plan(scores, held or {}, px, {} if states is None else states,
                tv, cfg(**kw), lot_size=100, asof=asof)


# ==================== 建仓 ====================

def test_entry_weight_interpolates_from_buy_line_to_strong_line():
    pp = run({A: 0.020, B: 0.035, C: 0.050, D: 0.080})
    assert pp.weights == {A: 0.05, B: pytest.approx(0.065), C: 0.08, D: 0.08}
    assert pp.plan.buys == {A: 5000, B: 6500, C: 8000, D: 8000}
    assert not pp.plan.sells
    assert pp.actions["entry"] == 4


def test_score_below_the_buy_line_never_enters():
    pp = run({A: 0.0199})
    assert not pp.plan.buys and not pp.weights


@pytest.mark.parametrize("tv,expected", [
    (100_000.0, "min"),      # 5% 只有 5,000 元 < 1 万 → 建不了仓
    (200_000.0, "ok"),       # 5% = 10,000 元 → 刚好卡在线上,允许
])
def test_minimum_trade_value_blocks_entries_in_a_small_account(tv, expected):
    pp = run({A: 0.02}, tv=tv)                             # 刚好买入线 → 5%
    if expected == "min":
        assert not pp.plan.buys
        assert "最小交易额" in pp.notes[A]
    else:
        assert pp.plan.buys[A] == int(tv * 0.05 / P)


def test_entry_state_freezes_the_step_at_the_entry_score():
    states: dict[str, NameState] = {}
    pp = run({A: 0.045}, states=states)
    apply_fills(states, pp.intents, {}, {A: pp.plan.buys[A]}, {A: P}, asof=D0)
    assert states[A].entry_score == 0.045
    assert states[A].ref_score == 0.045
    assert states[A].step == pytest.approx(0.045 - 0.02)   # Δ = 建仓分数 − 买入线


# ==================== 补仓 ====================

def test_add_only_fires_after_a_full_step_of_improvement():
    held, states = {A: 5000}, {A: st(entry=0.03)}          # Δ = 0.01
    assert run(held=held, scores={A: 0.039}, states=states).plan.buys == {}
    assert A in run(held=held, scores={A: 0.039},
                    states=states).keep                    # 带内不动
    pp = run(held=held, scores={A: 0.040}, states=states)  # 刚好一档
    assert pp.plan.buys == {A: 5000}
    assert pp.intents[A].state.ref_score == pytest.approx(0.04)


def test_add_is_capped_by_max_steps_per_eval_and_leftover_carries_forward():
    held, states = {A: 5000}, {A: st(entry=0.03)}
    pp = run(held=held, scores={A: 0.065}, states=states)   # 涨 0.035 = 3 档
    assert pp.intents[A].state.ref_score == pytest.approx(0.05)   # 只走 2 档
    assert pp.plan.buys == {A: 10000}                      # 5% + 2×5% = 15%
    pp2 = run(held={A: 15000}, scores={A: 0.065},
              states={A: pp.intents[A].state})              # 剩下 0.015 还在
    assert pp2.plan.buys == {A: 1000}                      # 15% → 16% 封顶


def test_position_never_exceeds_the_16pct_cap():
    pp = run(held={A: 15000}, scores={A: 0.09},
             states={A: st(entry=0.03)})
    assert pp.weights[A] == pytest.approx(0.16)
    assert pp.plan.buys == {A: 1000}


def test_add_blocked_by_budget_is_not_committed_and_retries_next_round():
    # 12 只已持满 60%,再补一档要 5% → 剩余额度 0.95−0.60 = 0.35,够;
    # 换成总仓位几乎打满的情形:18 只 × 5% = 0.90,再加一档就破 0.95
    held = {f"6000{i:02d}": 5000 for i in range(10, 28)}   # 18 只
    states = {s: st(entry=0.03) for s in held}
    scores = {s: 0.045 for s in held}                      # 每只都想补 5%
    pp = run(scores=scores, held=held, states=states, max_names=18,
             add_reserve_weight=0.0)
    fired = [s for s, i in pp.intents.items() if i.kind == "add"]
    assert len(fired) == 1                                 # 只剩 0.05 的额度
    assert pp.gross_weight <= 0.95 + 1e-9
    assert any("最小交易额" in n or "总仓位" in n for n in pp.notes.values())


# ==================== 减仓 / 清仓 ====================

def test_reduce_band_trims_by_one_step():
    pp = run(held={A: 16000}, scores={A: 0.019},
             states={A: st(entry=0.05, step=0.03)})         # 跌 0.031 → 一档
    assert pp.weights[A] == pytest.approx(0.11)
    assert pp.plan.sells == {A: 5000}
    assert pp.intents[A].kind == "trim"


def test_reduce_band_clears_instead_of_leaving_dust():
    pp = run(held={A: 5000}, scores={A: 0.019}, states={A: st(entry=0.03)})
    assert pp.weights[A] == 0.0
    assert pp.plan.sells == {A: 5000}
    assert pp.intents[A].kind == "exit_by_trim"
    assert "min_hold_weight" in pp.notes[A]


def test_score_below_sell_line_exits_even_below_the_minimum_trade_value():
    pp = run(held={A: 500}, scores={A: -0.01}, states={A: st()})
    assert pp.plan.sells == {A: 500}                       # 5,000 元,碎仓也放行
    assert pp.intents[A].kind == "exit"


def test_price_drift_over_the_cap_is_trimmed_back_but_not_below_the_floor():
    # 16,000 股 @12 元 = 19.2% → 压回 16% 要卖 32,000 元,应当执行
    pp = run(held={A: 16000}, scores={A: 0.03}, states={A: st()},
             prices={A: 12.0}, tv=1_000_000.0 + 16000 * 12.0 - 16000 * 10.0)
    assert pp.intents[A].kind == "cap"
    assert pp.weights[A] == pytest.approx(0.16)
    # 16,500 股 @10.1 元 ≈ 16.66% → 压回只需卖 6,600 元 < 1 万,不动
    tv2 = 1_000_000.0 + 16500 * (10.1 - 10.0)
    pp2 = run(held={A: 16500}, scores={A: 0.03}, states={A: st()},
              prices={A: 10.1}, tv=tv2)
    assert not pp2.plan.sells and A in pp2.keep
    assert "最小交易额" in pp2.notes[A]


# ==================== 减仓价禁止补仓 ====================

def test_add_is_forbidden_above_the_price_the_trim_filled_at():
    states = {A: st(entry=0.03, trim_price=9.5, trim_date=D0)}
    blocked = run(held={A: 5000}, scores={A: 0.06}, states=states, prices={A: 10.0})
    assert not blocked.plan.buys and A in blocked.keep
    assert "禁止补仓" in blocked.notes[A]
    allowed = run(held={A: 5000}, scores={A: 0.06},
                  states={A: st(entry=0.03, trim_price=10.5, trim_date=D0)},
                  prices={A: 10.0})
    assert allowed.plan.buys[A] == 10000                   # 涨了 3 档,一轮最多两档


def test_trim_price_is_recorded_from_the_actual_fill_and_survives_a_full_exit():
    states = {A: st(entry=0.03)}
    pp = run(held={A: 5000}, scores={A: 0.019}, states=states)
    res = apply_fills(states, pp.intents, {A: 5000}, {}, {A: 8.88}, asof=D0)
    assert states[A].trim_price == 8.88                    # 清仓了仍留着约束
    assert states[A].trim_date == str(D0.date())
    # 涨回来想在 9 元重建 → 挡掉;在 8.8 元 → 放行
    hi = run(held={}, scores={A: 0.05}, states=states, prices={A: 9.0})
    assert not hi.plan.buys and "禁止重建" in hi.notes[A]
    lo = run(held={}, scores={A: 0.05}, states={A: states[A]}, prices={A: 8.8})
    assert lo.plan.buys[A] == int(CAP * 0.08 / 8.8 // 100 * 100)
    assert res[A]


def test_trim_price_constraint_expires():
    old = D0 - pd.Timedelta(days=90)
    states = {A: st(entry=0.03, trim_price=9.5, trim_date=old)}
    pp = run(held={A: 5000}, scores={A: 0.06}, states=states,
             prices={A: 10.0}, asof=D0)
    assert pp.plan.buys[A] == 10000                        # 5% → 15%(两档)
    assert states[A].trim_price is None
    assert "有效期" in pp.notes[A]


def test_a_stop_out_below_the_sell_line_leaves_no_price_constraint():
    states = {A: st(entry=0.03)}
    pp = run(held={A: 5000}, scores={A: -0.02}, states=states)
    apply_fills(states, pp.intents, {A: 5000}, {}, {A: 9.0}, asof=D0)
    assert states == {}                                    # 真止损,不锁价格


# ==================== 状态提交 ====================

def test_state_does_not_advance_unless_the_order_filled_in_full():
    states = {A: st(entry=0.03)}
    pp = run(held={A: 5000}, scores={A: 0.045}, states=states)
    want = pp.plan.buys[A]
    same = apply_fills(dict(states), pp.intents, {A: 5000}, {A: 5000},
                       {A: P}, asof=D0)                    # 涨停没成交
    assert "未足量成交" in same[A]
    part = apply_fills(dict(states), pp.intents, {A: 5000},
                        {A: 5000 + want - 100}, {A: P}, asof=D0)
    assert "未足量成交" in part[A]
    full = apply_fills(states, pp.intents, {A: 5000}, {A: 5000 + want},
                       {A: P}, asof=D0)
    assert "补仓提交" in full[A]
    assert states[A].ref_score == pytest.approx(0.04)


def test_blocked_order_fires_again_next_round():
    """未成交不推进参考分数 → 下一轮同一档条件仍然成立,自动重试。"""
    states = {A: st(entry=0.03)}
    pp = run(held={A: 5000}, scores={A: 0.045}, states=states)
    apply_fills(states, pp.intents, {A: 5000}, {A: 5000}, {A: P}, asof=D0)
    again = run(held={A: 5000}, scores={A: 0.045}, states=states)
    assert again.plan.buys == {A: 5000}


# ==================== 不动的情形 ====================

def test_unpriced_or_unscored_holdings_are_left_alone():
    pp = run(held={A: 5000, B: 5000, C: 5000},
             scores={A: 0.03, B: 0.03, C: np.nan},
             prices={A: P, C: P}, states={A: st(), B: st(), C: st()})
    assert not pp.plan.sells and not pp.plan.buys
    assert pp.keep == frozenset({A, B, C})    # A/C 带内不动,B 缺价不动
    assert pp.notes[B] == "缺成交价"
    assert pp.notes[C] == "当日无预测分数"
    assert B not in pp.weights                 # 缺席 = 本轮不碰,不会被误清仓


def test_held_name_without_state_gets_a_baseline_and_waits_one_round():
    states: dict[str, NameState] = {}
    pp = run(held={A: 5000}, scores={A: 0.09}, states=states)
    assert not pp.plan.buys and "基线" in pp.notes[A]
    assert states[A].entry_score == 0.09
    # 基线建好后,再涨一档才补
    pp2 = run(held={A: 5000}, scores={A: 0.16}, states={A: states[A]})
    assert pp2.plan.buys == {A: 5000}


# ==================== 组合预算 ====================

def test_entries_are_queued_by_score_and_stopped_at_max_names():
    scores = {f"6000{i:02d}": 0.02 + i * 0.001 for i in range(20)}
    pp = run(scores)
    assert len(pp.plan.buys) == 12
    assert sum(1 for n in pp.notes.values() if "max_names" in n) == 8


def test_add_reserve_weight_holds_cash_back_for_pyramiding():
    """补仓预留额度是真的:新仓吃到 entry_cap 就停,不会把现金全花光。"""
    scores = {f"6000{i:02d}": 0.05 for i in range(20)}     # 每个都给到 8%
    pp = run(scores, max_names=12, add_reserve_weight=0.30)
    assert len(pp.plan.buys) == 8                           # 0.65 / 0.08 = 8 只
    assert pp.gross_weight == pytest.approx(0.64)
    assert sum(1 for n in pp.notes.values()
               if "add_reserve_weight" in n) == 20 - 8


def test_forced_exits_free_budget_for_same_round_adds():
    held = {A: 5000, B: 5000, C: 5000}
    states = {k: st(entry=0.03) for k in held}
    scores = {A: -0.02, B: 0.05, C: 0.045}                 # A 清仓腾出 5%
    pp = run(scores=scores, held=held, states=states,
             max_total_pct=0.20, add_reserve_weight=0.0, max_names=3,
             base_weight=0.05, strong_score=0.06)
    assert pp.plan.sells[A] == 5000
    assert pp.gross_weight <= 0.20 + 1e-9
    assert pp.plan.buys                                    # B 还能补


# ==================== 与 sizing 的接缝 ====================

def test_absolute_weights_are_not_renormalized():
    """normalize=False 时 5% 就是 5%,不会因为别的名字缺席而被放大。"""
    rp = rebalance_plan({A: 0.05, B: 0.05}, {}, {A: P, B: P}, CAP,
                        normalize=False)
    assert rp.buys == {A: 5000, B: 5000}                   # 不是各买 50%
    rp2 = rebalance_plan({A: 0.05}, {}, {A: P, B: P}, CAP, normalize=False)
    assert rp2.buys == {A: 5000}


def test_keep_locks_shares_so_price_drift_alone_never_trades():
    rp = rebalance_plan({A: 5000 * P / CAP}, {A: 5000}, {A: P}, CAP,
                        normalize=False, keep={A})
    assert not rp.sells and not rp.buys
    assert rp.target_shares == {A: 5000}


def test_legacy_renormalizing_behavior_is_unchanged():
    """旧口径(等权按比例分配)必须一个数都不变,否则回测基准会被悄悄改掉。"""
    weights = {f"60000{i}": 0.1 for i in range(10)}
    prices = {f"60000{i}": 10.0 for i in range(10)}
    rp = rebalance_plan(weights, {}, prices, CAP)
    assert sum(rp.buys.values()) * 10.0 == pytest.approx(CAP, rel=2e-3)


def test_enforce_min_trade_value_only_exempts_full_liquidations():
    rp = Plan(buys={A: 500, B: 1500}, sells={A: 500, B: 1500})
    enforce_min_trade_value(rp, {A: 500, B: 3000}, {A: P, B: P}, 10_000.0)
    assert rp.buys == {B: 1500}                            # 5,000 元的买单砍掉
    assert rp.sells == {A: 500, B: 1500}                   # A 是清仓,放行
    assert rp.buy_value == 15_000.0 and rp.sell_value == 20_000.0


# ==================== 配置与持久化 ====================

@pytest.mark.parametrize("kw", [
    {"sell_score": 0.03, "buy_score": 0.02},                # 带位反了
    {"strong_score": 0.02, "buy_score": 0.02},              # 插值除零
    {"min_hold_weight": 0.06, "base_weight": 0.05},         # 减仓形同虚设
    {"max_names": 40, "base_weight": 0.05},                 # 名额超过预算装得下的
    {"max_entry_weight": 0.20},                             # 越过单票上限
    {"add_reserve_weight": 0.96},                           # 预留超过总仓位
])
def test_contradictory_config_raises_at_startup(kw):
    with pytest.raises(ValueError, match="position_policy 参数矛盾"):
        cfg(**kw)


def test_unknown_or_typo_config_keys_are_rejected():
    with pytest.raises(ValueError, match="未知参数"):
        PolicyConfig.from_dict({"buy_thr": 0.02})
    c = PolicyConfig.from_dict({"buy_score": "0.03", "max_names": "10"},
                              max_total_pct=0.9)
    assert c.buy_score == 0.03 and c.max_names == 10
    assert c.max_total_pct == 0.9                          # 与实盘风控同源


def test_states_round_trip_through_disk(tmp_path):
    path = str(tmp_path / "policy_state.json")
    states = {A: st(entry=0.03, trim_price=9.5, trim_date=D0, adds=2, trims=1)}
    save_states(path, states)
    back = load_states(path)
    assert back[A].to_dict() == states[A].to_dict()
    assert load_states(str(tmp_path / "missing.json")) == {}
    open(path, "w").write("{not json")
    assert load_states(path) == {}
