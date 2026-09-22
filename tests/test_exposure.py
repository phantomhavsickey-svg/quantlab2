"""组合级暴露层（目标波动缩放 B + 运行时 IC 门控 A）的行为测试。

输入全是合成的：3 只股票 × 120 个交易日，分数直接由"未来 N 日收益"（或其相反数）
造出来，所以 RankIC 恰好是 ±1，门控何时该关上可以手算。
"""

import math

import numpy as np
import pandas as pd
import pytest

from utils.exposure import (ExposureOverlay, OverlayConfig, Overlay,
                            close_panel, overlay_args, rank_ic_series,
                            vol_scale)
from utils.position_policy import NameState, PolicyConfig, apply_fills, plan

CAP = 1_000_000.0
P = 10.0
A, B, C, D = "600001", "600002", "600003", "600004"
E = "600007"                         # held6() 占用了 600001~600006
DATES = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=120))


def oc(**kw) -> OverlayConfig:
    return OverlayConfig(enabled=True, **kw)


def panel(calm=0.002, storm=0.05, split=10 ** 9):
    """3 只股票的收盘宽表：第 split 行之后日振幅从 calm 换成 storm。"""
    t = np.arange(len(DATES))
    cols = {}
    for k in range(3):
        amp = np.where(t >= split, storm, calm)
        r = amp * np.cos(np.pi * t / 2.0 + k * 0.9)
        cols[[A, B, C][k]] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(cols, index=DATES)


def perfect_scores(closes, horizon, bad_from=None):
    """分数 = 未来 horizon 日收益（序完全一致 → RankIC = +1）；取负则 -1。"""
    sc = (closes.shift(-horizon) / closes - 1.0).copy()
    if bad_from is not None:
        rows = closes.index >= bad_from
        sc.loc[rows] = -sc.loc[rows]
    return sc


# ==================== RankIC ====================

def test_rank_ic_is_one_for_monotone_scores_and_minus_one_reversed():
    closes = panel()
    good = rank_ic_series(perfect_scores(closes, 5), closes, 5).dropna()
    bad = rank_ic_series(perfect_scores(closes, 5, bad_from=DATES[10]),
                         closes, 5).dropna()
    assert good.round(6).eq(1.0).all()
    flipped = bad[bad.index >= DATES[10]]
    assert flipped.round(6).eq(-1.0).all() and len(flipped) == 105


def test_last_horizon_rows_have_no_ic_observation_yet():
    """末尾 horizon 行的未来收益还没走出来 —— 门控正是靠这一点挡前视。"""
    closes = panel()
    ic = rank_ic_series(perfect_scores(closes, 20), closes, 20)
    assert ic.iloc[-20:].isna().all()
    assert ic.iloc[:-20].notna().all()


# ==================== B 目标波动 ====================

def test_vol_scale_only_delevers_and_respects_the_floor():
    cfg = oc(vol_target_ann=0.15, scale_floor=0.30)
    assert vol_scale(0.30, cfg) == pytest.approx(0.5)
    assert vol_scale(0.10, cfg) == 1.0                    # 低波动不加杠杆
    assert vol_scale(15.0, cfg) == pytest.approx(0.30)    # 地板
    assert vol_scale(None, cfg) == 1.0                    # 样本不足 → 不动
    assert vol_scale(0.30, oc(vol_target_ann=0.0)) == 1.0  # B 关掉


def test_high_vol_regime_shrinks_the_cap_by_the_hand_computed_ratio():
    closes = panel(split=90)
    ov = ExposureOverlay(perfect_scores(closes, 20), closes,
                         oc(vol_target_ann=0.15, vol_lookback_days=20,
                            scale_floor=0.05, ic_window_days=0))
    a, b = ov.at(DATES[60]), ov.at(DATES[119])
    hand = float(closes.pct_change().mean(axis=1)
                 .iloc[100:120].std(ddof=1)) * math.sqrt(252)
    assert a.cap_mult == pytest.approx(min(1.0, 0.15 / a.realized_vol))
    assert b.realized_vol == pytest.approx(hand)
    assert b.cap_mult == pytest.approx(min(1.0, 0.15 / hand))
    assert b.cap_mult < 1.0 < a.cap_mult + 1e-12


# ==================== A IC 门控 ====================

def test_gate_only_reacts_to_ic_obs_realised_by_the_eval_date():
    """模型从第 60 天起变成反向指标：窗口全坏要到 60+30+20 天才成立。"""
    closes = panel()
    ov = ExposureOverlay(perfect_scores(closes, 20, bad_from=DATES[60]), closes,
                         oc(vol_target_ann=0.0, ic_window_days=30,
                            ic_horizon_days=20, ic_min_obs=3, ic_cap_mult=0.5))
    assert ov.at(DATES[79]).rank_ic == pytest.approx(1.0)   # 窗口里还没有坏观测
    assert ov.at(DATES[79]).allow_entry
    assert not ov.at(DATES[82]).gated                       # 刚混进 3 个坏的
    w = ov.at(DATES[110])
    assert w.rank_ic == pytest.approx(-1.0) and w.gated
    assert w.n_ic_obs == 30 and w.cap_mult == pytest.approx(0.5)


def test_insufficient_observations_fail_open():
    closes = panel()
    ov = ExposureOverlay(perfect_scores(closes, 20), closes,
                         oc(vol_target_ann=0.0, ic_window_days=60,
                            ic_min_obs=5, ic_cap_mult=0.5))
    early = ov.at(DATES[23])              # 已兑现观测只有 4 个
    assert early.rank_ic is None and early.allow_entry
    assert early.cap_mult == 1.0


def test_query_matches_a_truncated_history():
    """asof 之后追加的数据不得改变 asof 的结论 —— 逐日核对前视。"""
    closes = panel(split=70)
    sc = perfect_scores(closes, 20, bad_from=DATES[50])
    cfg = oc(vol_target_ann=0.15, ic_window_days=40, ic_min_obs=3)
    full = ExposureOverlay(sc, closes, cfg)
    for d in (DATES[45], DATES[70], DATES[95], DATES[119]):
        cut = int(DATES.get_loc(d)) + 1
        past = ExposureOverlay(sc.iloc[:cut], closes.iloc[:cut], cfg)
        a, b = full.at(d), past.at(d)
        assert (a.cap_mult, a.allow_entry, a.n_ic_obs) == \
               (b.cap_mult, b.allow_entry, b.n_ic_obs)
        assert (a.realized_vol or 0.0) == pytest.approx(b.realized_vol or 0.0)


def test_two_channels_compound():
    ov = Overlay(cap_mult=0.8, allow_entry=False, realized_vol=0.2,
                 rank_ic=-0.01, n_ic_obs=30)
    args = overlay_args(ov, 0.95)
    assert args["entry_allowed"] is False
    assert args["max_total_pct_override"] == pytest.approx(0.76)
    assert overlay_args(None, 0.95) == {}


def test_close_panel_matches_the_signal_shape():
    raw = {s: pd.DataFrame({"日期": DATES, "收盘": panel()[s].to_numpy()})
           for s in (A, B)}
    cp = close_panel(raw)
    assert list(cp.columns) == [A, B] and cp.index.equals(DATES)


# ==================== 与带位状态机的接缝 ====================

def pol() -> PolicyConfig:
    return PolicyConfig()


def held6(each=0.08):
    names = [f"60000{i}" for i in range(1, 7)]
    sh = {s: int(each * CAP / P) for s in names}
    st = {s: NameState(entry_score=0.04, ref_score=0.04, step=0.01)
          for s in names}
    sc = {s: 0.04 for s in names}          # 全在带内：分数自己不会减仓
    return sh, st, sc


def test_de_gross_actually_sells_even_when_scores_say_hold():
    """这就是这套东西存在的理由：带内不动的名字，降杠杆必须能卖出去。"""
    sh, st, sc = held6()
    pp = plan(sc, sh, {s: P for s in sh}, dict(st), CAP, pol(),
              lot_size=100, asof=DATES[0], max_total_pct_override=0.24)
    assert pp.actions["de_gross"] == 6
    assert sum(pp.weights.values()) == pytest.approx(0.24)
    assert pp.plan.sells == {s: 4000 for s in sh}    # 8%→4%，10 元价 4000 股
    assert not pp.plan.buys


def test_de_gross_respects_min_trade_value():
    sh, st, sc = held6()
    # 48% → 47.5%：每票只卖 833 元 < 10000 最小笔额 → 整轮不动
    pp = plan(sc, sh, {s: P for s in sh}, dict(st), CAP, pol(),
              lot_size=100, asof=DATES[0], max_total_pct_override=0.475)
    assert not pp.plan.sells and not pp.plan.buys
    assert sum(pp.weights.values()) == pytest.approx(0.48)


def test_de_gross_leaves_the_score_ladder_alone():
    """降杠杆不改参考分数、不留减仓价 —— 它是暴露动作，不是分数动作。"""
    sh, st, sc = held6()
    states = dict(st)
    pp = plan(sc, sh, {s: P for s in sh}, states, CAP, pol(),
              lot_size=100, asof=DATES[0], max_total_pct_override=0.24)
    after = {s: int(sh[s] - 4000) for s in sh}
    msgs = apply_fills(states, pp.intents, sh, after, {s: P for s in sh},
                       asof=DATES[0])
    assert len(msgs) == 6
    for s in sh:
        assert states[s].ref_score == pytest.approx(0.04)
        assert states[s].trim_price is None and states[s].trims == 0


def test_entry_gate_blocks_new_names_but_not_stop_losses():
    sh, st, sc = held6()
    pp = plan({**sc, E: 0.09}, sh, {**{s: P for s in sh}, E: P}, dict(st),
              CAP, pol(), lot_size=100, asof=DATES[0], entry_allowed=False)
    assert not pp.plan.buys and E not in pp.weights
    assert "不新增仓位" in pp.notes[E]
    # 跌破清仓线的票照卖 —— 门控禁的是新增风险，不是保留风险
    pp2 = plan({**sc, B: -0.1}, sh, {s: P for s in sh}, dict(st), CAP, pol(),
               lot_size=100, asof=DATES[0], entry_allowed=False)
    assert pp2.plan.sells.get(B) == 8000 and pp2.actions["exit"] == 1


def test_de_gross_round_spends_nothing_new():
    """压上限的那一轮不许把回笼的钱立刻花出去，否则等于没降杠杆。"""
    sh, st, sc = held6()
    pp = plan({**sc, E: 0.09}, sh, {**{s: P for s in sh}, E: P}, dict(st),
              CAP, pol(), lot_size=100, asof=DATES[0],
              max_total_pct_override=0.24)
    assert pp.actions["de_gross"] == 6 and not pp.plan.buys


def test_no_override_reproduces_the_pure_score_band():
    sh, st, sc = held6()
    a = plan(sc, sh, {s: P for s in sh}, dict(st), CAP, pol(),
             lot_size=100, asof=DATES[0])
    b = plan(sc, sh, {s: P for s in sh}, dict(st), CAP, pol(),
             lot_size=100, asof=DATES[0], max_total_pct_override=0.95)
    assert (a.weights, a.plan.buys, a.actions) == (b.weights, b.plan.buys,
                                                   b.actions)
    assert a.actions == {}


def test_override_can_only_lower_the_cap():
    """传进来一个更大的上限也必须被 min() 挡掉：不因低波动自动加杠杆。"""
    sc = {A: 0.05, B: 0.05, C: 0.05, D: 0.05}
    px = {s: P for s in sc}
    big = plan(sc, {}, px, {}, CAP, pol(), lot_size=100, asof=DATES[0],
               max_total_pct_override=1.0)
    ref = plan(sc, {}, px, {}, CAP, pol(), lot_size=100, asof=DATES[0])
    assert big.weights == ref.weights
    assert sum(big.weights.values()) <= 0.95 + 1e-9
