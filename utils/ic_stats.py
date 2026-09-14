"""
逐日 IC 的统计推断 — HAC 标准误、方差比、block bootstrap、配对检验。

为什么需要这一层:
    前向 20 日收益标签在相邻交易日之间重叠 19 天,逐日 IC 序列因此强正自相关。
    直接用 mean/std 算 t(朴素 t)会把标准误低估约 sqrt(方差比) 倍,
    使显著性虚高一个量级。这里提供的三个量都是为了让 t 可以被辩护:

    - variance_ratio: VR = 1 + 2*sum_{h=1}^{H} (1-h/(H+1)) * rho_h  (Bartlett)
      有效样本量 N_eff = N / VR
    - newey_west_mean_t: 均值的 HAC t,带宽 H 取 >= 标签重叠长度 - 1
    - moving_block_bootstrap_ci: 不假设正态,按 L 日为块重抽样
"""

import numpy as np
import pandas as pd


def _as_1d(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def acf(x: np.ndarray, n_lags: int) -> np.ndarray:
    """样本自相关 rho_1..rho_{n_lags}(去均值,分母用有偏方差)。"""
    x = _as_1d(x)
    n = len(x)
    if n < 3:
        return np.zeros(max(n_lags, 0))
    d = x - x.mean()
    denom = float(d @ d)
    if denom < 1e-20:
        return np.zeros(n_lags)
    out = []
    for h in range(1, n_lags + 1):
        out.append(float(d[:-h] @ d[h:]) / denom if n - h > 0 else 0.0)
    return np.asarray(out, dtype=np.float64)


def _nw_bartlett_weights(n_lags: int) -> np.ndarray:
    """Newey-West Bartlett 核 w_h = 1 - h/(H+1),h = 1..H。"""
    return 1.0 - np.arange(1, n_lags + 1) / (n_lags + 1.0)


def variance_ratio(x: np.ndarray, n_lags: int = 19) -> float:
    """重叠标签造成的长期相关 → 方差比 VR >= 1。

    用 Newey-West 的 Bartlett 核截尾,与 newey_west_mean_t 完全同口径,
    因此恒有 naive_t / sqrt(VR) == HAC_t。
    """
    x = _as_1d(x)
    n = len(x)
    if n < 3:
        return 1.0
    rho = acf(x, min(n_lags, n - 1))
    return float(1.0 + 2.0 * np.sum(_nw_bartlett_weights(len(rho)) * rho))


def effective_n(x: np.ndarray, n_lags: int = 19) -> tuple[int, float]:
    """(原始样本量, 有效样本量 N/VR)。VR 下限截到 1。"""
    x = _as_1d(x)
    vr = max(variance_ratio(x, n_lags), 1.0)
    return len(x), float(len(x) / vr)


def newey_west_mean_t(x: np.ndarray, n_lags: int = 19
                      ) -> tuple[float, float, float, int]:
    """均值的 Newey-West(HAC)t 检验,与 statsmodels HAC(maxlags=H) 同口径。

    H0: mean(x) = 0。长程正相关下 HAC 方差 = 长程方差 / N。

    Args:
        x: 逐日 IC(或逐日 IC 差)序列
        n_lags: HAC 带宽,应 >= 标签重叠天数(20 日标签 → >= 19)

    Returns:
        (t, 双尾 p, mean, 样本量)
    """
    from scipy.stats import norm

    x = _as_1d(x)
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, float(np.mean(x)) if n else np.nan, n)
    d = x - x.mean()
    gamma0 = float(d @ d) / n
    if gamma0 < 1e-24:          # 序列无变化 → t 无定义(否则下溢会炸出巨值)
        return np.nan, np.nan, float(x.mean()), n
    long_run = gamma0
    hmax = min(n_lags, n - 1)
    if hmax >= 1:
        w = _nw_bartlett_weights(hmax)
        gammas = np.array([float(d[:-h] @ d[h:]) / n
                           for h in range(1, hmax + 1)])
        long_run += 2.0 * float(np.sum(w * gammas))
    se = np.sqrt(max(long_run, 1e-30) / n)
    t = float(x.mean() / se)
    p = float(2.0 * norm.sf(abs(t)))
    return t, p, float(x.mean()), n


def paired_hac_t(a: np.ndarray, b: np.ndarray, n_lags: int = 19
                 ) -> dict:
    """两个模型逐日 IC 差的 HAC 检验(按日期对齐后配对)。

    Args:
        a, b: pd.Series(index=交易日), 两个模型的逐日 IC
        n_lags: HAC 带宽

    Returns:
        {n_days, mean_a, mean_b, diff, t, p}
    """
    a = a if isinstance(a, pd.Series) else pd.Series(a)
    b = b if isinstance(b, pd.Series) else pd.Series(b)
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (both["a"] - both["b"]).to_numpy()
    t, p, m, n = newey_west_mean_t(d, n_lags)
    return {"n_days": int(n), "mean_a": float(both["a"].mean()),
            "mean_b": float(both["b"].mean()), "diff": m, "t": t, "p": p}


def moving_block_bootstrap_ci(x: np.ndarray, block: int = 20,
                              n_boot: int = 20000, alpha: float = 0.05,
                              seed: int = 42) -> tuple[float, float, float]:
    """均值的 moving-block bootstrap 置信区间(整块重抽样保留自相关结构)。

    块起点在 [0, n-block] 上均匀抽取,每份重抽样取 ceil(n/block) 个块后截断
    到 n —— 与序列长度无关的块数选择,避免短序列被过度重抽。

    Returns:
        (mean, lo, hi)
    """
    x = _as_1d(x)
    n = len(x)
    if n < block:
        return (float(np.mean(x)) if n else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    stat = x[idx].mean(axis=1)
    lo, hi = np.quantile(stat, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(x)), float(lo), float(hi)


def summarize_ic(ic_by_date: pd.Series, horizon: int = 20,
                 n_boot: int = 2000) -> dict:
    """一份逐日 IC 的完整推断摘要(可直接写进报告)。"""
    x = _as_1d(ic_by_date)
    n, n_eff = effective_n(x, horizon - 1)
    naive_t = float(x.mean() / (x.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan
    lag_t, lag_p, mean, _ = newey_west_mean_t(x, horizon - 1)
    _, lo, hi = moving_block_bootstrap_ci(x, block=horizon, n_boot=n_boot)
    return {
        "n_days": n,
        "mean_ic": float(mean),
        "std_ic": float(x.std(ddof=1)) if n > 1 else np.nan,
        "naive_t": naive_t,
        "variance_ratio": variance_ratio(x, horizon - 1),
        "n_eff": n_eff,
        "nw_lag": horizon - 1,
        "nw_t": lag_t,
        "nw_p": lag_p,
        "boot_block": horizon,
        "boot_ci_low": lo,
        "boot_ci_high": hi,
    }
