"""数据层分阶段计时 —— 回答"到底慢在哪一步",并给出可复现的数字。

用法:
    # 真实缓存(config.yaml 里的 data.factor_panel / daily_dir)
    python scripts/profile_data.py

    # 没有缓存的机器:按生产规模合成一份,只测 IO/构建的结构性开销
    python scripts/profile_data.py --synthetic

    # 只测日线列裁剪(E5)的读写比
    python scripts/profile_data.py --stage daily

torch 缺失时自动跳过 DataLoader 阶段(取窗口吞吐量要 GPU/CPU 推理环境)。
结果同时打印表格和写 JSON(--out profile.json)。
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SequenceStore 本身不需要 torch(只有 Dataset/DataLoader 要),没装 torch 的
# 机器借用 tests/conftest 的哑模块把 import 期糊过去,好让 [3][4] 两阶段可测。
HAVE_TORCH = importlib.util.find_spec("torch") is not None
if not HAVE_TORCH:
    import tests.conftest  # noqa: F401  (import 即安装哑模块)

from data.loader import (DATE_COL, CLOSE_COL, load_daily_dict,  # noqa: E402
                         load_factor_panel)

# 生产规模:999 只 × ~1360 个交易日 × 25 因子;日线 11 列
SYN_N_SYM = 999
SYN_N_DAYS = 1360
SYN_N_FACT = 25
DAILY_COLS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
              "振幅", "涨跌幅", "涨跌额", "换手率"]


def timer():
    t0 = time.perf_counter()
    return lambda: round(time.perf_counter() - t0, 3)


def stage(fn, out, key, *a, **kw):
    el = timer()
    res = fn(*a, **kw)
    out[key] = el()
    print(f"  {key:<26} {out[key]:>8.3f} s")
    return res


def make_synthetic(dirpath, n_sym, n_days, n_fact, seed=7):
    """按生产规模合成因子面板 + 日线缓存(parquet,列名与 Quantlab 一致)。"""
    daily_dir = os.path.join(dirpath, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    rng = np.random.default_rng(seed)
    rows = []
    cols = [f"f{i:02d}" for i in range(n_fact)]
    t0 = time.perf_counter()
    for s in range(n_sym):
        sym = f"{600000 + s:06d}"
        close = 8.0 * (1 + rng.normal(0, 0.02, n_days)).cumprod()
        df = pd.DataFrame({DATE_COL: dates, CLOSE_COL: close})
        for c in DAILY_COLS[1:]:
            df[c] = rng.uniform(0, 1e5, n_days)
        df.to_parquet(os.path.join(daily_dir, f"{sym}.parquet"))
        sus = np.zeros(n_days, bool)
        if s % 97 == 0:                       # 少量停牌股
            sus[n_days // 3:n_days // 3 + 30] = True
        kept = np.nonzero(~sus)[0]
        f = pd.DataFrame(rng.normal(0, 1, (kept.size, n_fact)), columns=cols)
        f.insert(0, "date", dates[kept])
        f.insert(1, "symbol", sym)
        rows.append(f)
    panel = pd.concat(rows, ignore_index=True)
    panel_path = os.path.join(dirpath, "factor_panel.parquet")
    panel.to_parquet(panel_path)
    print(f"  合成 {len(panel):,} 行面板 + {n_sym} 个日线文件,用时 "
          f"{time.perf_counter() - t0:.1f} s → {dirpath}")
    return panel_path, daily_dir


def dir_bytes(path):
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            tot += os.path.getsize(os.path.join(root, fn))
    return tot


def profile(panel_path, daily_dir, cfg, out, rm_dir):
    symbols = None
    if os.path.exists(daily_dir):
        symbols = sorted(f[:-8] for f in os.listdir(daily_dir)
                         if f.endswith(".parquet"))
    n_sym = len(symbols) if symbols else 0
    out["cache"] = {"panel_MB": round(os.path.getsize(panel_path) / 2**20, 1)
                    if os.path.exists(panel_path) else None,
                    "daily_MB": round(dir_bytes(daily_dir) / 2**20, 1),
                    "n_daily_files": n_sym}
    print(f"[缓存] 面板 {out['cache']['panel_MB']} MB | 日线 "
          f"{out['cache']['daily_MB']} MB / {n_sym} 个文件")

    print("[1] 因子面板读取")
    panel = stage(load_factor_panel, out, "load_factor_panel_s", panel_path)
    if symbols is None:
        symbols = sorted(panel["symbol"].unique())
    print(f"    {len(panel):,} 行 × {panel.shape[1]} 列")

    print("[2] 日线读取:全部列 vs 只要 日期/收盘(E5)")
    out["daily_all_s"] = round(_time_read(daily_dir, symbols, None), 3)
    print(f"  {'daily_all_columns':<26} {out['daily_all_s']:>8.3f} s")
    out["daily_projected_s"] = round(
        _time_read(daily_dir, symbols, [DATE_COL, CLOSE_COL]), 3)
    print(f"  {'daily_projected':<26} {out['daily_projected_s']:>8.3f} s")
    out["daily_projection_speedup"] = round(
        out["daily_all_s"] / max(out["daily_projected_s"], 1e-9), 2)
    print(f"    → 列裁剪加速 {out['daily_projection_speedup']}×")
    daily = load_daily_dict(daily_dir, symbols,
                            columns=[DATE_COL, CLOSE_COL])

    print("[3] SequenceStore 构建")
    from models.sequence_data import SequenceStore, Samples  # noqa: F401
    store = stage(SequenceStore, out, "store_build_s", panel, daily, cfg)
    print(f"    因子矩阵 {(len(store.symbols) * store.global_dates.size
                           * store.n_features * 4) / 2**20:.0f} MB 量级")

    print("[4] 样本索引 + 向量化标签")
    samples = stage(store.sample_index, out, "sample_index_s")
    out["n_samples"] = len(samples)
    out["index_MB"] = round((samples.sym_pos.nbytes + samples.t.nbytes
                             + samples.dates.nbytes) / 2**20, 1)
    t = timer()
    lab = samples.labels(store)
    out["labels_vectorised_s"] = t()
    # 同样工作的"笨写法":每样本一次 Python 调用,作为对照
    t = timer()
    for i in range(0, len(samples), 100):
        store.get_label(*samples[i])
    out["get_label_loop_s_per_1pct"] = t()
    print(f"    {len(samples):,} 样本 / {out['index_MB']} MB | 标签整段取 "
          f"{out['labels_vectorised_s']:.3f} s | 逐样本取 1% 要 "
          f"{out['get_label_loop_s_per_1pct']:.3f} s")

    print("[5] DataLoader 取窗口吞吐")
    if HAVE_TORCH:
        from models.sequence_data import make_loader
        sub = samples.select(np.arange(0, len(samples),
                                       max(1, len(samples) // 200_000)))
        t = timer()
        loader = make_loader(store, sub, batch_size=2048, shuffle=True)
        n = 0
        for x, y, d in loader:
            n += x.shape[0]
        out["loader_200k_samples_s"] = t()
        out["loader_samples"] = n
        print(f"    {n:,} 样本 / batch 2048 / workers 0 → "
              f"{out['loader_200k_samples_s']:.2f} s "
              f"({n / max(out['loader_200k_samples_s'], 1e-9):,.0f} 样本/s)")
    else:
        out["loader_200k_samples_s"] = None
        print("    torch 未安装,跳过(取窗口吞吐要在有 torch 的机器上测)")

    if rm_dir:
        shutil.rmtree(rm_dir, ignore_errors=True)
    return out


def _time_read(daily_dir, symbols, columns):
    t = timer()
    load_daily_dict(daily_dir, symbols, columns=columns)
    return t()


CFG = {"model": {"horizon": 20, "seq_len": 20,
                 "training": {"label_winsor": [0.01, 0.99]}},
       "data": {"min_valid_label_days": 15}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="忽略 config 里的路径,按生产规模合成一份缓存再测")
    ap.add_argument("--n-sym", type=int, default=SYN_N_SYM)
    ap.add_argument("--n-days", type=int, default=SYN_N_DAYS)
    ap.add_argument("--stage", choices=["all", "daily"], default="all")
    ap.add_argument("--out", default="data/profile.json")
    ap.add_argument("--keep", action="store_true",
                    help="--synthetic 时保留生成的缓存目录")
    a = ap.parse_args()

    tmp = None
    if a.synthetic:
        tmp = tempfile.mkdtemp(prefix="q2profile")
        panel_path, daily_dir = make_synthetic(tmp, a.n_sym, a.n_days,
                                              SYN_N_FACT)
    else:
        import yaml
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        panel_path = cfg["data"]["factor_panel"]
        daily_dir = cfg["data"]["daily_dir"]
        if not os.path.exists(panel_path) or not os.path.isdir(daily_dir):
            sys.exit(f"缓存不存在:\n  {panel_path}\n  {daily_dir}\n"
                     f"这台机器没有数据缓存,加 --synthetic 测结构性开销。")

    out = {"mode": "synthetic" if a.synthetic else "real",
           "pandas": pd.__version__, "numpy": np.__version__}
    if a.stage == "daily":
        symbols = sorted(f[:-8] for f in os.listdir(daily_dir)
                         if f.endswith(".parquet"))
        out["daily_all_s"] = _time_read(daily_dir, symbols, None)
        out["daily_projected_s"] = _time_read(
            daily_dir, symbols, [DATE_COL, CLOSE_COL])
        out["n_daily_files"] = len(symbols)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        profile(panel_path, daily_dir, CFG, out,
                None if a.keep else tmp)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
