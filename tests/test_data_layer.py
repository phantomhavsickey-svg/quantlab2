"""数据层重构的等价性断言 —— 性能改动不许动任何数字。

本文件唯一职责:`Samples` 列式索引 + 展平标签数组 + 日线列裁剪,必须和
"最笨的逐样本写法"给出**逐元素完全相同**的结果。所以参考实现刻意不复用
被测代码:标签用纯 Python 双层循环从原始 daily_dict 重算,缩尾按日分组
手工 clip,样本枚举把 sequence_data 头部那 4 条规则直译成循环,全程不碰
`label_offsets` / 向量化 fancy index 这些真正可能被改错的东西。

范围说明:参考实现锚定的是**当前实现的口径**(标签在因子日期序列上平移
horizon 个位置,而不是在完整日线日历上平移)。停牌股的这两种读法结果不同,
本测试不裁决哪种更好 —— 它只保证本轮改动没有把口径挪走。

比对口径见 `assert_winsor_equal`:未缩尾的行要求逐位相等(错位一个样本就
露馅),只有被裁到边界的行放行 1 ULP —— 因为 pandas 的 Series.quantile 与
np.nanquantile 的插值本身就有这个量级差异,而缩尾那段代码本轮没动。
"""

import numpy as np
import pandas as pd
import pytest

from data.loader import DATE_COL, CLOSE_COL, load_daily_dict
from models.sequence_data import SequenceStore

HORIZON = 20
SEQ_LEN = 10
MIN_VALID = 15
N_DAYS = 90
N_SYM = 12
GAP_SYM = 3        # 这只股票中段"停牌"
GAP_START = 40     # 停牌起始日线位置
GAP_LEN = 25       # 停牌长度 > horizon,用于考验 min_valid_label_days
FACTORS = ["mom", "vol", "turn"]
WINSOR = (0.01, 0.99)
ALL_DATES = pd.bdate_range("2023-01-02", periods=N_DAYS)
FIRST_AFTER_GAP = ALL_DATES[GAP_START + GAP_LEN]   # 停牌后第一个交易日

CFG = {"model": {"horizon": HORIZON, "seq_len": SEQ_LEN,
                 "training": {"label_winsor": list(WINSOR)}},
       "data": {"min_valid_label_days": MIN_VALID}}


def make_panel_and_daily():
    """合成因子面板 + 日线字典。

    停牌在两个数据源里同时体现:日线收盘置 NaN,因子面板直接没有这些行。
    这样同时覆盖"尾部 horizon 行标签为 NaN"和"长停牌被 min_valid 过滤"
    两条路径,并且让因子日期成为日线日期的真子集 —— 位置口径与日历口径
    在停牌股上不等价,任何把两者混用的改动都会被 reference_raw 抓到。
    """
    rng = np.random.default_rng(7)
    rows, daily = [], {}
    for s in range(N_SYM):
        sym = f"{600000 + s:06d}"
        close = 10.0 * (1 + rng.normal(0, 0.02, N_DAYS)).cumprod()
        suspended = np.zeros(N_DAYS, dtype=bool)
        if s == GAP_SYM:
            suspended[GAP_START:GAP_START + GAP_LEN] = True
            close = np.where(suspended, np.nan, close)
        df = pd.DataFrame({
            DATE_COL: ALL_DATES, CLOSE_COL: close, "开盘": close * 0.99,
            "最高": close * 1.01, "最低": close * 0.98,
            "换手率": rng.uniform(0.5, 5.0, N_DAYS),
        })
        daily[sym] = df
        kept = df[~suspended].copy()
        for col in FACTORS:
            kept[col] = rng.normal(0, 1, len(kept))
        rows.append(kept.rename(columns={DATE_COL: "date"})
                    .assign(symbol=sym)[["date", "symbol", *FACTORS]])
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True), daily


@pytest.fixture(scope="module")
def sources():
    panel, daily = make_panel_and_daily()
    return panel, daily, SequenceStore(panel, daily, CFG)


# ==================== 参考实现 ====================

def reference_raw(daily, store):
    """逐股票纯循环重算 (未缩尾标签, 标签窗口有效交易日),返回 {symbol: 数组}。

    规则 3(标签):在因子日期序列上平移 horizon 个位置,
        label = close[j+h]/close[j] - 1
    任一端非有限则 NaN;j+h 越界也是 NaN。
    规则 4(有效天数):在**完整日线日历**上数 [i, i+horizon] 内的有限收盘,
    日线末端越界的行按有效计(实现的 cum_nan 末端 fill_value 就是这个口径,
    这些行标签本来就是 NaN,由规则 3 过滤掉)。
    """
    lab_out, val_out = {}, {}
    for sym in store.symbols:
        fdates = store.dates_by_symbol[sym]
        T = len(fdates)
        lab = np.full(T, np.nan, dtype=np.float32)
        val = np.zeros(T, dtype=np.int16)
        df = daily.get(sym)
        if df is not None and not df.empty:
            dcol = df[DATE_COL].to_numpy(dtype="datetime64[ns]")
            ccol = df[CLOSE_COL].to_numpy(dtype=np.float64)
            pos = {d: i for i, d in enumerate(dcol)}
            n = len(dcol)
            aligned = np.array([ccol[pos[d]] if d in pos else np.nan
                                for d in fdates], dtype=np.float64)
            for j in range(T):
                if j + HORIZON < T:
                    a, b = aligned[j], aligned[j + HORIZON]
                    if np.isfinite(a) and np.isfinite(b):
                        lab[j] = np.float32(b / a - 1.0)
                i = pos.get(fdates[j])
                if i is None:
                    continue
                cnt = 0
                for k in range(i, i + HORIZON + 1):
                    if k >= n or np.isfinite(ccol[k]):
                        cnt += 1
                val[j] = cnt
        lab_out[sym] = lab
        val_out[sym] = val
    return lab_out, val_out


def reference_winsorized(store, raw_lab):
    """按日截面缩尾(只用当日横截面),拼平成 (总行数,) float32。

    拼接顺序 = store.symbols 顺序,每个 symbol 内日期升序 —— 这正是
    label_offsets 声称的布局,所以本函数同时独立校验了 offsets。

    Returns:
        (缩尾后, 缩尾前, 被裁的行数) —— 三个数组同序同长
    """
    flat_raw = np.concatenate([raw_lab[s] for s in store.symbols])
    flat_dates = np.concatenate([store.dates_by_symbol[s]
                                 for s in store.symbols])

    groups = {}
    for i, d in enumerate(flat_dates):
        groups.setdefault(d, []).append(i)

    out = flat_raw.copy()
    n_clipped = 0
    for idxs in groups.values():
        v = flat_raw[idxs]
        if not np.isfinite(v).any():
            continue
        lo, hi = np.nanquantile(v, WINSOR[0]), np.nanquantile(v, WINSOR[1])
        for i in idxs:
            x = flat_raw[i]
            if not np.isfinite(x):
                continue
            y = lo if x < lo else (hi if x > hi else x)
            if y != x:
                n_clipped += 1
                out[i] = y
    return out, flat_raw, n_clipped


def reference_sample_rows(store, flat_lab, flat_raw):
    """把 4 条规则直译成循环,返回 [(sym_pos, t, date, 缩尾后, 缩尾前), ...]。"""
    offsets, acc = {}, 0
    for sym in store.symbols:
        offsets[sym] = acc
        acc += len(store.dates_by_symbol[sym])

    rows = []
    for si, sym in enumerate(store.symbols):
        fdates = store.dates_by_symbol[sym]
        if len(fdates) <= SEQ_LEN:
            continue
        for t in range(SEQ_LEN - 1, len(fdates)):
            if store.valid_flat[offsets[sym] + t] < MIN_VALID:
                continue
            lab = flat_lab[offsets[sym] + t]
            if not np.isfinite(lab):
                continue
            rows.append((si, t, fdates[t], lab, flat_raw[offsets[sym] + t]))
    return rows


def assert_winsor_equal(actual, ref, raw, where="label"):
    """缩尾结果的逐元素比对:未动的行逐位相等,被裁的行放行 1 ULP。

    `raw` 是参考实现的缩尾前值。三条断言各司其职:
      1. 有限性模式一致(哪些行有标签)
      2. **"被裁过"的行集合一致** —— 偏移/顺序错一位就会露馅
      3. 未裁的行逐位相等,全部行在 1 ULP 内
    """
    fin_ref = np.isfinite(ref)
    np.testing.assert_array_equal(np.isfinite(actual), fin_ref,
                                  err_msg=f"{where}: NaN 模式不一致")
    moved_ref = fin_ref & (ref != raw)
    moved_act = fin_ref & (actual != raw)
    np.testing.assert_array_equal(moved_act, moved_ref,
                                  err_msg=f"{where}: 被缩尾的行集合不一致")
    np.testing.assert_array_equal(actual[fin_ref & ~moved_ref],
                                  ref[fin_ref & ~moved_ref],
                                  err_msg=f"{where}: 未缩尾的行不逐位相等")
    np.testing.assert_allclose(actual[fin_ref], ref[fin_ref], rtol=1e-6)


# ==================== 标签层 ====================

def test_label_flat_matches_reference(sources):
    _, daily, store = sources
    raw_lab, _ = reference_raw(daily, store)
    ref, raw, n_clipped = reference_winsorized(store, raw_lab)

    assert ref.shape == store.label_flat.shape
    n_moved = int((ref != raw)[np.isfinite(ref)].sum())
    assert n_clipped == n_moved
    # 截面只有 12 只股票,线性插值的 1%/99% 分位必然落在首尾两点之间,
    # 所以每个交易日大约裁掉 2 行(尾部无标签的日子更少)。
    assert 0 < n_moved <= 2 * len(store.global_dates)
    assert_winsor_equal(store.label_flat, ref, raw)


def test_valid_flat_matches_reference(sources):
    """valid 的末端越界口径很微妙,单独钉住,防止将来被"顺手修正"改掉。"""
    _, daily, store = sources
    _, raw_val = reference_raw(daily, store)
    ref = np.concatenate([raw_val[s] for s in store.symbols])
    np.testing.assert_array_equal(store.valid_flat, ref)

    v = raw_val[store.symbols[GAP_SYM]]
    assert v.min() == 1, "停牌起点附近应有窗口几乎全是缺口的行"
    assert (v < MIN_VALID).any() and (v >= MIN_VALID).any()
    # 非停牌股票的 valid 只受日线末端影响
    v_ok = raw_val[store.symbols[0]]
    assert (v_ok[:-HORIZON] == HORIZON + 1).all()


def test_symbol_without_daily_has_no_samples(sources):
    """面板里有、日线字典里缺的股票:标签全 NaN、一条样本都没有,但不报错。"""
    panel, daily, store = sources
    gone = store.symbols[-1]
    stripped = dict(daily)
    stripped.pop(gone)
    s2 = SequenceStore(panel, stripped, CFG)
    assert gone in s2.symbols

    si = s2.symbols.index(gone)
    lo, hi = s2.label_offsets[si], s2.label_offsets[si + 1]
    assert np.isnan(s2.label_flat[lo:hi]).all()
    assert (s2.valid_flat[lo:hi] == 0).all()

    base = store.sample_index()
    kept = base.select(base.sym_pos != si)
    now = s2.sample_index()
    assert (now.sym_pos != si).all()
    assert len(kept) < len(base)
    np.testing.assert_array_equal(now.keys, kept.keys)
    np.testing.assert_array_equal(np.isfinite(now.labels(s2)),
                                  np.ones(len(now), bool))
    # 标签数值这里会小幅变动:少了这只股票,每日截面的 1%/99% 分位数
    # 本身就变了。属于口径的正当结果,不是本轮改动的事,故不比。


# ==================== 样本索引层 ====================

def test_sample_index_matches_reference_enumeration(sources):
    _, daily, store = sources
    raw_lab, _ = reference_raw(daily, store)
    flat, raw, _ = reference_winsorized(store, raw_lab)
    ref = reference_sample_rows(store, flat, raw)
    assert len(ref) > 500, f"合成样本太少({len(ref)}),测试失效"

    samples = store.sample_index()
    assert len(samples) == len(ref)
    np.testing.assert_array_equal(samples.sym_pos,
                                  np.array([r[0] for r in ref], np.int32))
    np.testing.assert_array_equal(samples.t,
                                  np.array([r[1] for r in ref], np.int32))
    np.testing.assert_array_equal(samples.dates,
                                  np.array([r[2] for r in ref],
                                           dtype="datetime64[ns]"))
    assert_winsor_equal(
        samples.labels(store),
        np.array([r[3] for r in ref], np.float32),
        np.array([r[4] for r in ref], np.float32), where="样本标签")


def test_offset_lookup_and_scalar_paths_agree(sources):
    """label_offsets 布局、Samples.labels、get_label 三者互证(全等)。"""
    _, _, store = sources
    samples = store.sample_index()
    via_offsets = store.label_flat[store.label_offsets[samples.sym_pos]
                                   + samples.t]
    np.testing.assert_array_equal(samples.labels(store), via_offsets)
    assert samples.labels(store).dtype == np.float32

    for i in np.linspace(0, len(samples) - 1, 200).astype(int):
        si, t = samples[i]
        assert store.get_label(si, t) == float(via_offsets[i])

    assert store.label_offsets[0] == 0
    assert store.label_offsets[-1] == len(store.label_flat)
    assert np.all(np.diff(store.label_offsets) > 0)
    assert len(samples.keys) == len(set(samples.keys.tolist()))


def test_select_and_date_range_agree(sources):
    """select(掩码) 必须等价于把同样日期条件传给 sample_index。"""
    _, _, store = sources
    samples = store.sample_index()
    uniq = np.unique(samples.dates)
    assert len(uniq) > 30
    lo, hi = uniq[len(uniq) // 3], uniq[2 * len(uniq) // 3]
    mask = (samples.dates >= lo) & (samples.dates <= hi)
    sub = samples.select(mask)
    bounded = store.sample_index(min_date=pd.Timestamp(lo),
                                 max_date=pd.Timestamp(hi))
    assert 0 < len(sub) < len(samples)
    np.testing.assert_array_equal(sub.keys, bounded.keys)
    np.testing.assert_array_equal(sub.dates, bounded.dates)
    np.testing.assert_array_equal(sub.labels(store), bounded.labels(store))
    assert set(sub.keys.tolist()) <= set(samples.keys.tolist())
    np.testing.assert_array_equal(sub.labels(store),
                                  np.array([store.get_label(*sub[i])
                                            for i in range(len(sub))],
                                           dtype=np.float32))
    np.testing.assert_array_equal(store.symbols_of(sub),
                                  np.take(store.symbols, sub.sym_pos))


def test_inference_index_superset_and_tail(sources):
    """推理索引不要求标签:日线尾部 horizon 个位置必须出现。"""
    _, _, store = sources
    train = store.sample_index()
    infer = store.inference_index()
    assert set(train.keys.tolist()) <= set(infer.keys.tolist())
    assert len(infer) > len(train)

    last = store.global_dates.max()
    assert (infer.dates == last).any()
    assert (train.dates == last).sum() == 0
    # 推理集里存在无标签的行(尾部),而训练集一行都不能有
    assert (~np.isfinite(infer.labels(store))).sum() >= HORIZON
    assert np.isfinite(train.labels(store)).all()
    assert (infer.t >= store.seq_len - 1).all()


# ==================== 窗口层(规则 2) ====================

def test_windows_are_position_sliced(sources):
    """窗口 = 最近 seq_len 个有因子值的交易日;停牌缺口不做日历对齐。"""
    _, _, store = sources
    sym = store.symbols[GAP_SYM]
    fdates = store.dates_by_symbol[sym]

    samples = store.sample_index()
    ts = samples.t[samples.sym_pos == GAP_SYM]
    assert ts.size > 0
    for t in ts[:: max(1, ts.size // 20)]:
        t = int(t)
        win = store.get_window(GAP_SYM, t)
        assert win.shape == (SEQ_LEN, store.n_features)
        assert win.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(
            win, store.factors_by_symbol[sym][t - SEQ_LEN + 1: t + 1])
        assert np.isfinite(win).all()
        d = fdates[t - SEQ_LEN + 1: t + 1]
        assert np.all(np.diff(d.astype("int64")) > 0), "窗口日期非严格递增"
        assert d.max() == fdates[t]

    # 停牌后第一个交易日的窗口横跨缺口:日历天数远大于 seq_len,
    # 说明切片按位置而非按日历对齐(规则 2)
    t = int(np.nonzero(fdates == np.datetime64(FIRST_AFTER_GAP))[0][0])
    assert t >= SEQ_LEN - 1
    d = fdates[t - SEQ_LEN + 1: t + 1]
    span = int((d[-1] - d[0]).astype("timedelta64[D]").astype(int)) + 1
    assert span > SEQ_LEN + GAP_LEN, "该样本未跨越停牌缺口"


# ==================== 日线列裁剪(E5) ====================

def test_column_projection_keeps_everything_identical(sources, tmp_path):
    """只读 日期/收盘 两列,store 的标签与样本枚举必须逐元素不变。"""
    panel, daily, store = sources
    cache = tmp_path / "daily"
    cache.mkdir()
    for sym, df in daily.items():
        df.to_parquet(cache / f"{sym}.parquet")

    symbols = sorted(daily)
    full = load_daily_dict(str(cache), symbols)
    proj = load_daily_dict(str(cache), symbols, columns=[DATE_COL, CLOSE_COL])
    assert all(list(df.columns) == [DATE_COL, CLOSE_COL]
               for df in proj.values())
    assert all(len(df.columns) > 2 for df in full.values())
    assert sorted(full) == sorted(proj) == symbols

    for other in (SequenceStore(panel, full, CFG),
                  SequenceStore(panel, proj, CFG)):
        assert other.symbols == store.symbols
        np.testing.assert_array_equal(other.label_flat, store.label_flat)
        np.testing.assert_array_equal(other.valid_flat, store.valid_flat)
        np.testing.assert_array_equal(other.sample_index().keys,
                                      store.sample_index().keys)


def test_projection_refuses_to_drop_needed_columns(sources, tmp_path):
    """列裁剪不能把 DATE_COL/CLOSE_COL 裁掉 —— 缺列必须报错而非静默。"""
    _, daily, _ = sources
    cache = tmp_path / "daily"
    cache.mkdir()
    for sym, df in daily.items():
        df.to_parquet(cache / f"{sym}.parquet")
    with pytest.raises(ValueError, match="列"):
        load_daily_dict(str(cache), sorted(daily), columns=["开盘", "最高"])
