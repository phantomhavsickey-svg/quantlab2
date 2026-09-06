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

    # 全样本汇总
    print("-" * len(header))
    sums = []
    for name, df in (("LightGBM", lgb), ("Transformer", tf)):
        if df is None:
            sums.append(np.nan)
            continue
        di = df["date"].values.astype("datetime64[D]").astype(np.int64)
        ic = compute_rank_ic(df["prediction"].to_numpy(),
                             df["forward_return"].to_numpy(), di)
        sums.append(ic["mean_ic"])
    print(f"{'全样本汇总':<24s} {sums[0]:>12.4f} {sums[1]:>15.4f} "
          f"{sums[1]-sums[0] if np.isfinite(sums[0]) and np.isfinite(sums[1]) else np.nan:>13.4f}")
    print("=" * len(header))

    # 结论
    if np.isfinite(sums[0]) and np.isfinite(sums[1]):
        if sums[1] > sums[0] + 0.005:
            print(f"结论: Transformer OOS Rank IC 显著优于 LightGBM"
                  f" (+{sums[1]-sums[0]:.4f})")
        elif sums[1] < sums[0] - 0.005:
            print(f"结论: LightGBM OOS Rank IC 更优"
                  f" ({sums[0]:.4f} vs {sums[1]:.4f})")
        else:
            print(f"结论: 两者 OOS Rank IC 相当"
                  f" ({sums[0]:.4f} vs {sums[1]:.4f})")


if __name__ == "__main__":
    main()
