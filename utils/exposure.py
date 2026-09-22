"""
组合级暴露层 —— 目标波动缩放(B) + 运行时 IC 门控(A)。

带位策略(`utils/position_policy.py`)只管**单票**的分数带和名额,组合层的
"总共敢下多少注"仍然是 config 里一个写死的 `max_total_pct`。2026-09-22 的实测
说明这正是回撤的来源:分数带全档最大回撤 36%~46%,而把 `max_total_pct` 静态压到
60% 就掉到 33% —— 分数线控不住暴露,总仓位上限才控得住。本模块把那个上限变成
随市场状态变化的量,两条通道:

    B 目标波动:scale = clip(target_ann / 近 lookback 日已实现波动, floor, 1)
      只降不升 —— 低波动时不放大杠杆(这个策略的年化波动本来就常年高于目标)。
    A 运行时 IC 门控:近 window 交易日**已兑现**的 RankIC 均值为负 → 不许开新仓,
      总仓位再乘 ic_cap_mult。直接打"模型失效还满仓"这一段。

两者都只作用于"这一轮允许用多少仓位 + 还让不让建仓",不改分数带状态机本身,
所以不需要重训,`predictions.parquet` 一字不变。

**没有未来函数**:两个量都只用 asof 当日收盘及之前的信息,而成交发生在次日开盘。
IC 尤其要留心 —— t 日的 RankIC 要用到 t+20 日的收盘价,所以 t 日这个观测在
asof = t+20 之前**不存在**:`at()` 只取位置 ≤ i − horizon 的 IC 片段。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ==================== 配置 ====================

@dataclass(frozen=True)
class OverlayConfig:
    """暴露层参数。全部来自 config.yaml 的 exposure_overlay 段。"""

    enabled: bool = False
    # --- B 目标波动 ---
    vol_target_ann: float = 0.15      # 0 = 关掉波动缩放
    vol_lookback_days: int = 20       # 已实现波动的回看交易日数
    scale_floor: float = 0.30         # 缩放的地板,避免波动尖峰把仓位打到近零
    # --- A 运行时 IC 门控 ---
    ic_window_days: int = 60          # 0 = 关掉门控
    ic_horizon_days: int = 20         # 分数对应的预测期(与标签口径一致)
    ic_min_obs: int = 5               # 窗口内不足这么多个已兑现观测就不判
    ic_cap_mult: float = 0.50         # 门控触发时总仓位再乘的系数

    def __post_init__(self):
        if self.vol_target_ann < 0:
            raise ValueError("vol_target_ann 不能为负(0 = 关闭)")
        if self.ic_window_days < 0:
            raise ValueError("ic_window_days 不能为负(0 = 关闭)")
        if not 0 < self.scale_floor <= 1:
            raise ValueError("scale_floor 必须落在 (0, 1]")
        if not 0 < self.ic_cap_mult <= 1:
            raise ValueError("ic_cap_mult 必须落在 (0, 1]")
        if self.vol_lookback_days < 2 or self.ic_horizon_days < 1:
            raise ValueError("vol_lookback_days 至少 2、ic_horizon_days 至少 1")
        if self.ic_min_obs < 1:
            raise ValueError("ic_min_obs 至少 1")

    @property
    def vol_on(self) -> bool:
        return self.vol_target_ann > 0

    @property
    def ic_on(self) -> bool:
        return self.ic_window_days > 0


def overlay_from_config(config: dict) -> OverlayConfig | None:
    """config.yaml → OverlayConfig;缺段或 enabled=false 时返回 None。"""
    d = dict(config.get("exposure_overlay") or {})
    if not d.pop("enabled", False):
        return None
    fields = set(OverlayConfig.__dataclass_fields__)
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"exposure_overlay 有未知参数 {sorted(unknown)},"
                         f" 可用参数: {sorted(fields)}")
    ints = ("vol_lookback_days", "ic_window_days", "ic_horizon_days",
            "ic_min_obs")
    kwargs = {k: (int(v) if k in ints else float(v)) for k, v in d.items()}
    return OverlayConfig(enabled=True, **kwargs)


# ==================== 计算 ====================

def close_panel(data_dict: dict) -> pd.DataFrame:
    """{symbol: 日线} → date × symbol 收盘宽表(暴露层的数据口径)。

    与 `BacktestEngine._build_price_panel` 同口径,抽出来是为了让模拟盘链路
    不必依赖引擎的私有方法 —— 两端的波动/IC 必须用同一张价格表算。
    """
    cols = {}
    for sym, df in (data_dict or {}).items():
        if df is None or len(df) == 0 or "收盘" not in df.columns:
            continue
        s = df.set_index("日期")["收盘"]
        s.index = pd.DatetimeIndex(s.index)
        cols[str(sym)] = s
    return pd.DataFrame(cols).sort_index()


def rank_ic_series(scores: pd.DataFrame, closes: pd.DataFrame,
                   horizon: int) -> pd.Series:
    """逐日截面 RankIC,对应"分数预测未来 horizon 个交易日收益"的标签口径。

    Args:
        scores: 宽表 date × symbol 的分数(缺席/不可交易为 NaN)
        closes: 宽表 date × symbol 收盘价,与 scores 同一列口径
    Returns:
        Series index=date, name='rank_ic';样本不足的日子为 NaN

    注意最后 horizon 行必然是 NaN(未来收益还没走出来),这不是 bug,
    正是 `ExposureOverlay.at()` 靠它挡前视的依据。
    """
    fwd = closes.shift(-int(horizon)) / closes - 1.0
    fwd = fwd.reindex(index=scores.index, columns=scores.columns)
    ok = scores.notna() & fwd.notna()
    a = scores.where(ok).rank(axis=1)
    b = fwd.where(ok).rank(axis=1)
    ac = a.sub(a.mean(axis=1), axis=0)
    bc = b.sub(b.mean(axis=1), axis=0)
    den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
    ic = (ac * bc).sum(axis=1) / den.replace(0.0, np.nan)
    ic = ic.where(ok.sum(axis=1) >= 3)
    ic.name = "rank_ic"
    return ic


def vol_scale(realized_ann: float | None, cfg: OverlayConfig) -> float:
    """已实现波动 → 仓位缩放系数。只降不升,带地板。"""
    if not cfg.vol_on or realized_ann is None or not math.isfinite(realized_ann) \
            or realized_ann <= 0:
        return 1.0
    return min(1.0, max(cfg.vol_target_ann / realized_ann, cfg.scale_floor))


@dataclass(frozen=True)
class Overlay:
    """一次评估的暴露层结论。"""
    cap_mult: float          # 乘在 PolicyConfig.max_total_pct 上
    allow_entry: bool        # False = 本轮不许开新仓/补仓
    realized_vol: float | None
    rank_ic: float | None    # 窗口内已兑现 RankIC 的均值
    n_ic_obs: int

    @property
    def gated(self) -> bool:
        return not self.allow_entry

    def note(self) -> str:
        v = "样本不足" if self.realized_vol is None else f"{self.realized_vol:.1%}"
        i = ("未判" if self.rank_ic is None
             else f"{self.rank_ic:+.4f}({self.n_ic_obs} 个观测)")
        return (f"全池已实现年化波动 {v} → 缩放 {self.cap_mult:.2f},"
                f" 近窗 RankIC {i},"
                f" {'门控触发:禁建仓' if self.gated else '门控未触发'}")


def overlay_args(ov: "Overlay | None", base_cap: float) -> dict:
    """暴露层结论 → `policy_plan` / `plan_orders` 的 kwargs。

    集中在一处翻译,回测引擎、模拟盘与每日批处理三条链路才不会各算各的。
    """
    if ov is None:
        return {}
    return {"max_total_pct_override": base_cap * ov.cap_mult,
            "entry_allowed": ov.allow_entry}


class ExposureOverlay:
    """把 (分数面板 + 收盘价面板) 变成一个按日期查询的暴露层。

    构造时一次算完全体 RankIC 与全池等权日收益(都是 O(天数×股票数) 的向量化
    运算),之后 `at()` 只做切片,所以回测里逐调仓日查询不额外放大耗时。
    """

    def __init__(self, scores: pd.DataFrame | pd.Series,
                 closes: pd.DataFrame, cfg: OverlayConfig):
        if not cfg.enabled:
            raise ValueError("OverlayConfig.enabled=false,不需要构造暴露层")
        self.cfg = cfg
        c = closes[~closes.index.duplicated(keep="last")].sort_index()
        self.dates = pd.DatetimeIndex(c.index)
        # 全池等权日收益:横截面均值。用它的波动是因为多头组合的波动
        # 几乎就是它的波动(实测 β≈1.1、ρ≈0.9),而组合自身净值在开局没有历史。
        self.ew_ret = c.pct_change().mean(axis=1)
        # 分数可以是 (date, symbol) MultiIndex 的长序列/带 score 列的长表,
        # 也可以直接是 date × symbol 宽表 —— 三条链路传进来的形态不一样
        wide = scores
        if isinstance(wide, pd.DataFrame) and "score" in wide.columns:
            wide = wide["score"]
        if isinstance(wide, pd.Series) and isinstance(wide.index, pd.MultiIndex):
            wide = wide.unstack("symbol")
        self.ic: pd.Series | None = None
        if cfg.ic_on:
            sw = wide if isinstance(wide, pd.DataFrame) else wide.to_frame("score")
            sw = sw[~sw.index.duplicated(keep="last")].sort_index()
            self.ic = rank_ic_series(sw, c, cfg.ic_horizon_days) \
                .reindex(self.dates)

    def _pos(self, asof) -> int:
        """asof 对应的日历下标:取 ≤ asof 的最后一个交易日。"""
        ts = pd.Timestamp(asof)
        return int(self.dates.searchsorted(ts, side="right")) - 1

    def at(self, asof) -> Overlay:
        i = self._pos(asof)
        cfg = self.cfg
        if i < 1:
            return Overlay(1.0, True, None, None, 0)

        realized = None
        if cfg.vol_on:
            lo = i - cfg.vol_lookback_days + 1
            seg = self.ew_ret.iloc[max(lo, 0):i + 1]
            if lo >= 0 and seg.notna().sum() >= max(5, cfg.vol_lookback_days // 2):
                realized = float(seg.std(ddof=1)) * math.sqrt(TRADING_DAYS)

        ic_mean, n_obs = None, 0
        if self.ic is not None:
            hi = i - cfg.ic_horizon_days          # 只有走到 asof 才"兑现"完
            lo = hi - cfg.ic_window_days + 1
            if hi >= 0:
                obs = self.ic.iloc[max(lo, 0):hi + 1].dropna()
                n_obs = int(len(obs))
                if n_obs >= cfg.ic_min_obs:
                    ic_mean = float(obs.mean())
        gated = ic_mean is not None and ic_mean < 0
        cap_mult = vol_scale(realized, cfg) * (cfg.ic_cap_mult if gated else 1.0)
        return Overlay(cap_mult, not gated, realized, ic_mean, n_obs)
