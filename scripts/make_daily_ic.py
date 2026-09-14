#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 OOS 预测缓存导出逐日 Rank IC,并给出可信的显著性检验。

为什么单独有这个脚本:
    所有"IC 是多少、显不显著"的说法都必须能在**不重训**的前提下被第三方复算。
    predictions.parquet 有 11 MB 且需要 torch 才能生成,而逐日 IC 只有 ~20 KB,
    把它提交进仓库,任何人 `git clone` 后跑一个脚本就能核对全部统计结论。

用法:
    python scripts/make_daily_ic.py                          # 用 config 里的缓存
    python scripts/make_daily_ic.py --pred path.parquet --out reports/daily_ic.csv
    python scripts/make_daily_ic.py --pred path.parquet --csv old.csv   # 只核对不重写

输出:
    reports/daily_ic.csv        date,rank_ic,n_symbols
    stdout                      mean / 朴素 t / 方差比 / N_eff / NW t / bootstrap CI
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ic_stats import summarize_ic  # noqa: E402


def daily_ic(df: pd.DataFrame, min_symbols: int = 30) -> pd.DataFrame:
    """按日截面 Spearman Rank IC。

    Args:
        df: 含 date / prediction / forward_return 的 OOS 预测
        min_symbols: 当日截面样本数下限,低于此值不计算 IC(排序无意义)

    Returns:
        DataFrame(index=date, columns=rank_ic / n_symbols)
    """
    from scipy.stats import spearmanr

    rows = []
    for d, sub in df.groupby("date", sort=True):
        n = len(sub)
        if n < min_symbols:
            rows.append((d, np.nan, n))
            continue
        p = sub["prediction"].to_numpy(dtype=np.float64)
        y = sub["forward_return"].to_numpy(dtype=np.float64)
        if p.std() < 1e-8 or y.std() < 1e-8:
            rows.append((d, np.nan, n))
            continue
        r = spearmanr(p, y).correlation
        rows.append((d, float(r) if np.isfinite(r) else np.nan, n))
    out = pd.DataFrame(rows, columns=["date", "rank_ic", "n_symbols"])
    return out.set_index("date").sort_index()


def compare_csv(new: pd.DataFrame, path: str) -> bool:
    """与已有 CSV 逐日核对(容差 1e-9);不一致则打印最大差异并返回 False。"""
    old = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    both = pd.concat([old["rank_ic"].rename("old"),
                      new["rank_ic"].rename("new")], axis=1)
    only_old = both["new"].isna() & both["old"].notna()
    only_new = both["old"].isna() & both["new"].notna()
    diff = (both["old"] - both["new"]).abs()
    bad = diff > 1e-9
    ok = not (only_old.any() or only_new.any() or bad.any())
    print(f"\n核对 {path}: 共同 {int(both[['old', 'new']].notna().all(axis=1).sum())} 天"
          f" | 仅旧 {int(only_old.sum())} | 仅新 {int(only_new.sum())}"
          f" | 数值不符 {int(bad.sum())}"
          f" | 最大差 {diff.max():.3e}")
    if not ok:
        print(f"  首个不符: {both[bad | only_old | only_new].index[0]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="导出逐日 Rank IC 与显著性检验")
    ap.add_argument("--pred", default=None,
                    help="OOS 预测 parquet(date/symbol/prediction/forward_return),"
                         "默认取 config.yaml 的 data.prediction_cache")
    ap.add_argument("--config", default=None, help="config.yaml 路径")
    ap.add_argument("--out", default="reports/daily_ic.csv")
    ap.add_argument("--csv", default=None,
                    help="只与给定 CSV 核对,不写文件")
    ap.add_argument("--horizon", type=int, default=20,
                    help="前向标签天数 → HAC 带宽与 bootstrap 块长")
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()

    pred = args.pred
    if pred is None:
        cfg_path = args.config or str(
            Path(__file__).resolve().parent.parent / "config.yaml")
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pred = cfg["data"]["prediction_cache"]
        args.horizon = int(cfg["model"].get("horizon", args.horizon))
    pred = Path(os.path.expandvars(str(pred)))
    if not pred.exists():
        sys.exit(f"预测文件不存在: {pred}\n"
                 f"  先跑 python main.py train,或用 --pred 指定路径")

    df = pd.read_parquet(pred)
    missing = {"date", "prediction", "forward_return"} - set(df.columns)
    if missing:
        sys.exit(f"{pred} 缺少列: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])

    out = daily_ic(df)
    ic = out["rank_ic"].dropna()
    if ic.empty:
        sys.exit("没有可计算的交易日(截面样本都少于 min_symbols?)")

    s = summarize_ic(ic, horizon=args.horizon, n_boot=args.bootstrap)
    print(f"预测文件        : {pred}  ({len(df):,} 条)")
    print(f"可计算交易日    : {s['n_days']} / {len(out)}"
          f"  (跳过 {len(out) - s['n_days']} 天:样本过少或无方差)")
    print(f"日均截面样本数  : {out['n_symbols'].mean():.0f}")
    print(f"mean Rank IC    : {s['mean_ic']:+.6f}")
    print(f"std             : {s['std_ic']:.6f}   ICIR(日) = "
          f"{s['mean_ic'] / s['std_ic']:.3f}")
    print(f"朴素 t          : {s['naive_t']:.3f}   ← 20 日重叠标签下非法,不要引用")
    print(f"方差比 VR       : {s['variance_ratio']:.3f}   → 标准误被低估 "
          f"{np.sqrt(s['variance_ratio']):.2f} 倍")
    print(f"有效样本 N_eff  : {s['n_eff']:.1f}  (原始 {s['n_days']})")
    print(f"Newey-West t    : {s['nw_t']:.3f}   p = {s['nw_p']:.6f}"
          f"   (带宽 = {s['nw_lag']})")
    print(f"block bootstrap : 95% CI [{s['boot_ci_low']:+.6f},"
          f" {s['boot_ci_high']:+.6f}]  (块长 {s['boot_block']}, "
          f"{args.bootstrap} 次重抽样)")

    if args.csv:
        sys.exit(0 if compare_csv(out, args.csv) else 1)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(path, index=False, float_format="%.8f")
    print(f"\n写入 {path}  ({len(out)} 行, {path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
