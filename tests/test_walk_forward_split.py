"""Walk-Forward 折内切分的正确性断言 —— P0-1 / P0-6 的回归测试。

2026-08 那次基线的验证集是训练集的子集(valid ⊂ train),导致:
早停从未触发、best_rank_ic 无意义、8 折 best_IC 全在 0.89~0.95;
并且训练/验证样本的 20 日前向标签窗口直接伸进 OOS 窗口。
这两件事都由本文件的断言永久挡住。
"""

import types

import numpy as np
import pandas as pd
import pytest

from models.trainer import TransformerTrainer
from models.sequence_data import walk_forward_windows, FoldWindow

HORIZON = 20
SEQ_LEN = 20


def make_store(n_days=1400, start="2021-01-04"):
    """只需 global_dates / horizon / seq_len 的鸭子类型 store。"""
    dates = pd.bdate_range(start, periods=n_days).to_numpy()
    return types.SimpleNamespace(global_dates=dates,
                                 horizon=HORIZON, seq_len=SEQ_LEN)


def make_samples(store, per_day=50):
    """造 (sym, t) 样本下标与日期数组,和 sample_index() 同形状。"""
    all_idx, all_dates = [], []
    for t in range(SEQ_LEN, len(store.global_dates)):
        for s in range(per_day):
            all_idx.append((s, t))
            all_dates.append(store.global_dates[t])
    return all_idx, np.asarray(all_dates)


def make_trainer(val_months=3, embargo_days=0):
    cfg = {"model": {"walk_forward": {"min_train_months": 24,
                                      "retrain_months": 6,
                                      "val_months": val_months,
                                      "embargo_days": embargo_days}}}
    return types.SimpleNamespace(cfg=cfg)


def split(store, w, **kw):
    trainer = make_trainer(**kw)
    all_idx, all_dates = make_samples(store)
    tr, va, info = TransformerTrainer._fold_split(trainer, store, all_idx,
                                                  all_dates, w)
    return tr, va, info, all_idx, all_dates


FOLD = FoldWindow(fold=1, ws=pd.Timestamp("2023-01-04"),
                  we=pd.Timestamp("2023-07-04"), n_days=126)


def test_train_and_valid_are_disjoint():
    """P0-6:验证集绝不能是训练集的子集。"""
    store = make_store()
    tr, va, _, all_idx, _ = split(store, FOLD)
    a = {all_idx[i] for i in np.nonzero(tr)[0]}
    b = {all_idx[i] for i in np.nonzero(va)[0]}
    assert a.isdisjoint(b), f"train ∩ valid = {len(a & b)} 条"


def test_three_ranges_do_not_overlap():
    store = make_store()
    tr, va, info, all_idx, all_dates = split(store, FOLD)
    ws, we = np.datetime64(FOLD.ws), np.datetime64(FOLD.we)
    oos = (all_dates >= ws) & (all_dates < we)
    for name, a in (("train", tr), ("valid", va)):
        assert not (a & oos).any(), f"{name} 与 OOS 重叠"
    assert not (tr & va).any()
    # train/valid/OOS 之外还留着 purge 缺口 [purge_start, ws) —— 它必须真实存在,
    # 四段合起来才正好覆盖到 we 为止的所有样本
    gap = (all_dates >= info["purge_start"].to_datetime64()) & (all_dates < ws)
    assert not (gap & (tr | va | oos)).any(), "purge 缺口被用尽了"
    assert (tr | va | oos | gap).tolist() == (all_dates < we).tolist()
    assert int(np.unique(all_dates[gap]).size) == HORIZON
    assert info["n_oos_days"] == int(np.unique(all_dates[oos]).size)


def test_labels_cannot_reach_into_oos():
    """P0-1:任何训练/验证样本的 [t, t+horizon] 标签窗口都在 OOS 之前。"""
    store = make_store()
    tr, va, info, _, all_dates = split(store, FOLD)
    ws_i = int(np.searchsorted(store.global_dates, np.datetime64(FOLD.ws)))
    for mask in (tr, va):
        last_i = int(np.searchsorted(store.global_dates, all_dates[mask].max()))
        assert last_i + HORIZON < ws_i, "标签窗口越过 OOS 起点"
    assert info["purge_days"] == HORIZON


def test_embargo_widens_the_gap():
    store = make_store()
    _, _, base, _, _ = split(store, FOLD)
    _, _, emb, _, _ = split(store, FOLD, embargo_days=5)
    assert emb["purge_days"] == base["purge_days"] + 5
    assert emb["purge_start"] < base["purge_start"]


def test_validation_is_immediately_before_the_purge_gap():
    store = make_store()
    tr, va, info, _, all_dates = split(store, FOLD)
    assert all_dates[va].max() < info["purge_start"].to_datetime64()
    assert all_dates[tr].max() < info["val_start"].to_datetime64()


def test_empty_validation_window_raises_instead_of_silently_reusing():
    """val_months 太小 → 验证段为空时必须报错,不能退回旧行为。"""
    store = make_store()
    with pytest.raises(RuntimeError, match="验证段为空"):
        split(store, FOLD, val_months=0)


def test_tail_fold_too_short_is_dropped():
    """P0-7:一个只有 8 个交易日的尾部折曾把头条 IC 从 0.056 抬到 0.114。"""
    store = make_store(n_days=1400)
    dates = store.global_dates
    w0 = walk_forward_windows(dates, 24, 6)
    # 手工截断数据到某个折中途,复现尾部短折
    truncated = dates[:len(dates) - 120]
    w1 = walk_forward_windows(truncated, 24, 6, min_oos_days=20)
    for w in w1:
        n = int(((pd.DatetimeIndex(truncated) >= w.ws)
                 & (pd.DatetimeIndex(truncated) < w.we)).sum())
        assert n >= 20, f"折 {w.fold} 只有 {n} 个交易日"
    assert len(w1) < len(w0) or w1[-1].we <= pd.Timestamp(truncated[-1]) \
        + pd.offsets.MonthEnd(0)
    assert w1[0].ws == w0[0].ws


def test_fold_windows_tile_without_gaps():
    store = make_store()
    ws = walk_forward_windows(store.global_dates, 24, 6)
    for a, b in zip(ws, ws[1:]):
        assert a.we == b.ws or b.ws > a.we, "相邻折的 OOS 窗口重叠"
        assert b.ws >= a.we
    assert ws[0].fold == 1
    assert [w.fold for w in ws] == list(range(1, len(ws) + 1))
