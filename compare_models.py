#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型对比 — 同一时间窗下对比 Transformer (quantlab2) 与 LightGBM (quantlab)
的样本外 Rank IC。

用法:
    python compare_models.py [--lgb D:/quantlab/data/cache/predictions.parquet]
                             [--tf data/cache/predictions.parquet]

说明:
    - 两个预测文件 schema 一致: date, symbol, prediction, forward_return
    - 按 quantlab2 的 Walk-Forward 折窗口(每 6 个月)分别计算日截面
      Spearman Rank IC,并输出全样本汇总
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from models.trainer import compute_rank_ic
from utils.ic_stats import (summarize_ic, paired_hac_t,
                            moving_block_bootstrap_ci)


def load_preds(path: str, name: str) -> pd.DataFrame:
    if not Path(path).exists():
        print(f"[{name}] 预测文件不存在: {path}")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df


def fold_windows(start: str, months: int = 6):
    """生成每 6 个月的对比窗口(与 config 的 walk_forward 一致)。"""
    ws = pd.Timestamp(start)
    end = pd.Timestamp("2026-08-15")
    windows = []
    while ws < end:
        windows.append((ws, ws + pd.DateOffset(months=months)))
        ws = ws + pd.DateOffset(months=months)
    return windows


def main():
    parser = argparse.ArgumentParser(description="Transformer vs LightGBM OOS IC 对比")
    parser.add_argument("--lgb", default="D:/quantlab/data/cache/predictions.parquet")
    parser.add_argument("--tf", default="data/cache/predictions.parquet")
    parser.add_argument("--start", default="2023-01-04",
                        help="对比窗口起点(quantlab2 Walk-Forward 首个 OOS 起点)")
    parser.add_argument("--horizon", type=int, default=20,
                        help="前向标签天数;决定 HAC 带宽(horizon-1)与 bootstrap 块长")
    parser.add_argument("--bootstrap", type=int, default=20000,
                        help="moving-block bootstrap 重抽样次数")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="显著性水平")
    args = parser.parse_args()

    lgb = load_preds(args.lgb, "LightGBM")
    tf = load_preds(args.tf, "Transformer")

    header = (f"{'窗口':<24s} {'LightGBM IC':>12s} {'Transformer IC':>15s} "
              f"{'IC差(TF-LGB)':>13s}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    all_ic = {}
    for i, (ws, we) in enumerate(fold_windows(args.start), start=1):
        row = f"Fold {i} [{ws:%Y-%m}, {we:%Y-%m})"
        vals = []
        for df in (lgb, tf):
            if df is None:
                vals.append(np.nan)
                continue
            w = df[(df["date"] >= ws) & (df["date"] < we)]
            if len(w) == 0:
                vals.append(np.nan)
                continue
            di = w["date"].values.astype("datetime64[D]").astype(np.int64)
            ic = compute_rank_ic(w["prediction"].to_numpy(),
                                 w["forward_return"].to_numpy(), di)
            vals.append(ic["mean_ic"])
        diff = vals[1] - vals[0] if np.isfinite(vals[0]) and np.isfinite(vals[1]) else np.nan
        print(f"{row:<24s} {vals[0]:>12.4f} {vals[1]:>15.4f} {diff:>13.4f}")
        all_ic[f"fold_{i}"] = {"lgb": vals[0], "tf": vals[1]}

    # 全样本汇总(按日合并,保留逐日 IC 序列用于统计检验)
    print("-" * len(header))
    sums, daily = [], {}
    for name, df in (("LightGBM", lgb), ("Transformer", tf)):
        if df is None:
            sums.append(np.nan)
            continue
        di = df["date"].values.astype("datetime64[D]").astype(np.int64)
        ic = compute_rank_ic(df["prediction"].to_numpy(),
                             df["forward_return"].to_numpy(), di)
        sums.append(ic["mean_ic"])
        daily[name] = ic["ic_by_date"]
    n_days = max((len(s) for s in daily.values()), default=0)
    print(f"{'全样本汇总(按日)':<22s} {sums[0]:>12.4f} {sums[1]:>15.4f} "
          f"{sums[1]-sums[0] if np.isfinite(sums[0]) and np.isfinite(sums[1]) else np.nan:>13.4f}"
          f"   [{n_days} 个交易日]")
    print("=" * len(header))

    if len(daily) < 2:
        print("结论: 缺少一侧预测文件,无法对比")
        return

    # --- 两个模型各自的水平是否显著(朴素 t 在 20 日重叠标签下非法) ---
    print(f"\n逐日 Rank IC 的显著性(HAC 带宽 = horizon-1 = {args.horizon - 1}):")
    for name in ("LightGBM", "Transformer"):
        s = summarize_ic(daily[name], horizon=args.horizon,
                         n_boot=args.bootstrap)
        print(f"  {name:<12s} mean={s['mean_ic']:+.4f}  n={s['n_days']}  "
              f"朴素 t={s['naive_t']:.2f}  VR={s['variance_ratio']:.2f}  "
              f"N_eff={s['n_eff']:.0f}  NW t={s['nw_t']:.2f}  p={s['nw_p']:.4f}  "
              f"95%CI[{s['boot_ci_low']:+.4f}, {s['boot_ci_high']:+.4f}]")

    # --- 差值检验:唯一能支撑"谁更好"的说法 ---
    r = paired_hac_t(daily["Transformer"], daily["LightGBM"],
                     args.horizon - 1)
    both = pd.concat([daily["Transformer"].rename("tf"),
                      daily["LightGBM"].rename("lgb")], axis=1).dropna()
    d = (both["tf"] - both["lgb"]).to_numpy(float)
    _, lo, hi = moving_block_bootstrap_ci(
        d, block=args.horizon, n_boot=args.bootstrap)
    print(f"\n配对差值(TF − LGB),n={r['n_days']} 个交易日:")
    print(f"  Δmean = {r['diff']:+.4f}   NW t = {r['t']:.2f}   p = {r['p']:.4f}")
    print(f"  moving-block bootstrap( L={args.horizon}) 95% CI "
          f"= [{lo:+.4f}, {hi:+.4f}]")

    sig = np.isfinite(r["p"]) and r["p"] < args.alpha
    same = (np.isfinite(lo) and lo <= 0 <= hi) or not sig
    if same:
        print(f"\n结论: 两者 OOS Rank IC 之差在 α={args.alpha} 下不显著"
              f"(p={r['p']:.3f},CI 含 0)—— 不能说谁更强。")
    elif r["diff"] > 0:
        print(f"\n结论: Transformer 的 OOS Rank IC 显著高于 LightGBM"
              f"(Δ={r['diff']:+.4f}, p={r['p']:.4f})。")
    else:
        print(f"\n结论: LightGBM 的 OOS Rank IC 显著高于 Transformer"
              f"(Δ={r['diff']:+.4f}, p={r['p']:.4f})。")


if __name__ == "__main__":
    main()
