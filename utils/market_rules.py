r"""
A股交易规则 — 涨跌停幅度、停牌、可成交性、可交易掩码。

回测与实盘共用这一份判定,避免"回测假设成交、实盘被拒单"两套口径。

口径约定:
    - 用 `涨跌幅` 列(百分数)判涨跌停,不用价差:前复权改变价格绝对值但
      不改变涨跌幅
    - 容差 0.5 个百分点,与 Quantlab data/cleaner.py 的识别方式一致
    - `can_fill` 是**成交时点**的性质(开盘一字涨停买不进、跌停卖不出)
    - `build_tradable_mask` 是**信号时点**的性质(信号日没成交 = 已停牌),
      不含未来信息,用于选股阶段
"""

import pandas as pd
from loguru import logger

DATE_COL = "日期"
CLOSE_COL = "收盘"
CHG_COL = "涨跌幅"
VOL_COL = "成交量"

# 板块 → 涨跌停幅度(科创板/创业板 20%,主板 10%)
LIMIT_RULES = (("688", 0.20), ("30", 0.20), ("60", 0.10), ("00", 0.10))
DEFAULT_LIMIT_PCT = 0.10
LIMIT_TOL = 0.5  # 百分点容差


def get_limit_pct(symbol) -> float:
    """按代码前缀取涨跌停幅度;未覆盖的前缀(北交所等)按主板 10% 处理。"""
    s = str(symbol).zfill(6)
    for prefix, pct in LIMIT_RULES:
        if s.startswith(prefix):
            return pct
    return DEFAULT_LIMIT_PCT


def at_limit_up(symbol, change_pct) -> bool:
    if change_pct is None or pd.isna(change_pct):
        return False
    return float(change_pct) >= get_limit_pct(symbol) * 100 - LIMIT_TOL


def at_limit_down(symbol, change_pct) -> bool:
    if change_pct is None or pd.isna(change_pct):
        return False
    return float(change_pct) <= -(get_limit_pct(symbol) * 100 - LIMIT_TOL)


def _get(row, col):
    return row[col] if (hasattr(row, "index") and col in row.index) or \
        (isinstance(row, dict) and col in row) else None


def is_suspended(row) -> bool:
    """停牌/无成交:成交量缺失或为 0,或收盘价缺失。"""
    if row is None:
        return True
    vol = _get(row, VOL_COL)
    close = _get(row, CLOSE_COL)
    if vol is None or pd.isna(vol) or float(vol) <= 0:
        return True
    if close is None or pd.isna(close) or float(close) <= 0:
        return True
    return False


def can_fill(row, symbol, side: str) -> tuple[bool, str]:
    """成交时点可成交性检查。

    Args:
        row: 当日日线一行(Series/dict),当日无 bar 时传 None
        symbol: 股票代码,决定涨跌停幅度
        side: "buy" / "sell"

    Returns:
        (能否成交, 原因)
    """
    if row is None:
        return False, "无当日日线"
    if is_suspended(row):
        return False, "停牌/无成交"
    chg = _get(row, CHG_COL)
    if side == "buy" and at_limit_up(symbol, chg):
        return False, f"涨停 {float(chg):+.2f}%"
    if side == "sell" and at_limit_down(symbol, chg):
        return False, f"跌停 {float(chg):+.2f}%"
    return True, "ok"


def build_tradable_mask(daily: dict, dates, symbols=None) -> pd.Series:
    """信号日口径的可交易掩码(选股阶段用)。

    只回答"信号日当天这只股票有没有正常成交",不看下一个交易日,
    所以喂给 signals_from_predictions 不会引入未来函数。已连续停牌的
    股票不该占掉 Top-K 名额。

    Args:
        daily: {symbol: DataFrame(日期/成交量/收盘)}
        dates: 信号日序列
        symbols: 限定股票范围(默认 daily 里全部)

    Returns:
        全 True 的 bool Series,MultiIndex (date, symbol);
        不可交易的组合直接缺席(reindex 后按 False 处理)
    """
    want = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    cols = {}
    for sym, df in daily.items():
        if symbols is not None and sym not in symbols:
            continue
        if df is None or len(df) == 0 or DATE_COL not in df.columns:
            continue
        s = df.set_index(DATE_COL)
        s = s[~s.index.duplicated(keep="last")]
        if CLOSE_COL not in s.columns:
            continue
        ok = s[CLOSE_COL].notna() & (s[CLOSE_COL] > 0)
        if VOL_COL in s.columns:
            ok &= s[VOL_COL].fillna(0) > 0
        cols[sym] = ok
    if not cols:
        return pd.Series(dtype=bool,
                         index=pd.MultiIndex.from_arrays(
                             [[], []], names=["date", "symbol"]))

    panel = pd.DataFrame(cols).reindex(want)
    panel.index = want
    mask = panel.fillna(False).stack().astype(bool)
    mask.index.names = ["date", "symbol"]
    mask.name = "tradable"
    kept = int(mask.sum())
    logger.info(f"可交易掩码: {want.size:,} 个信号日 × {len(cols)} 只 = "
                f"{want.size * len(cols):,} 候选,通过 {kept:,} "
                f"(剔除 {want.size * len(cols) - kept:,})")
    return mask[mask]
