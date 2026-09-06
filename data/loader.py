"""
数据加载 — 复用 D:\quantlab 的缓存资产（路径来自 config 的 data 段）。

约定:
    - 因子面板: 长表 parquet,列 = 25 个因子 + date + symbol
      （已在 quantlab 处理阶段完成截面 zscore + lag 1,防未来函数）
    - 日线缓存: 每股一个 parquet,中文列名(日期/开盘/收盘/最高/最低/
      成交量/换手率/...),前复权,用于计算前向收益标签
    - 基准指数: akshare 下载 + 本地缓存,失败降级为 None(不阻断回测)
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# 日线缓存列名(中文,来自 quantlab 下载器)
DATE_COL = "日期"
CLOSE_COL = "收盘"


# ==================== 因子面板 ====================

def load_factor_panel(path: str) -> pd.DataFrame:
    """加载因子面板长表。

    断言并记录:
        - date/symbol 无缺失
        - 每个 symbol 的日期严格递增
        - 行数/股票数/因子数/日期范围

    Args:
        path: 因子面板 parquet 路径

    Returns:
        DataFrame 含 date/symbol 列 + 因子列
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"因子面板不存在: {path}\n"
            f"请先在 quantlab 运行 python main.py factors 生成因子面板,"
            f"或修改 config.yaml 的 data.factor_panel 指向其他路径")

    logger.info(f"加载因子面板: {path}")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])

    assert panel["date"].notna().all(), "因子面板 date 列含 NaN"
    assert panel["symbol"].notna().all(), "因子面板 symbol 列含 NaN"

    factor_cols = [c for c in panel.columns if c not in ["date", "symbol"]]

    # 每个 symbol 内日期必须严格递增(面板原本按 symbol+date 排序)
    for sym, grp in panel.groupby("symbol", sort=False):
        if not grp["date"].is_monotonic_increasing:
            raise ValueError(f"symbol {sym} 的日期未严格递增")

    n_symbols = panel["symbol"].nunique()
    logger.info(f"因子面板: {len(panel):,} 行 × {len(factor_cols)} 个因子, "
                f"{n_symbols} 只股票, "
                f"{panel['date'].min().date()} → {panel['date'].max().date()}")
    return panel


# ==================== 日线 ====================

def load_daily_dict(daily_dir: str,
                    symbols: list[str]) -> dict[str, pd.DataFrame]:
    """加载日线缓存 {symbol: DataFrame(中文列名)}。

    只加载 symbols 列表中的股票(999 个 parquet 是主要 IO 耗时项,
    冒烟测试可用子集)。缺失文件 warn 并跳过。

    Args:
        daily_dir: 日线缓存目录(内含 {symbol}.parquet)
        symbols: 需要加载的股票代码列表

    Returns:
        {symbol: DataFrame},日期升序、日期列为 datetime
    """
    daily_dir = Path(daily_dir)
    if not daily_dir.exists():
        raise FileNotFoundError(
            f"日线缓存目录不存在: {daily_dir}\n"
            f"请先在 quantlab 运行 python main.py download 下载日线,"
            f"或修改 config.yaml 的 data.daily_dir")

    data = {}
    missing = 0
    for sym in symbols:
        f = daily_dir / f"{sym}.parquet"
        if not f.exists():
            missing += 1
            continue
        df = pd.read_parquet(f)
        if DATE_COL not in df.columns or CLOSE_COL not in df.columns:
            raise ValueError(f"日线文件缺少 {DATE_COL}/{CLOSE_COL} 列: {f}")
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        data[sym] = df

    if missing:
        logger.warning(f"{missing} 只股票缺少日线文件(标签将为 NaN 并被过滤)")
    logger.info(f"加载日线: {len(data)} 只股票")
    return data


# ==================== 基准指数 ====================

def load_benchmark(code: str, start: str, end: str,
                   cache_path: str | None = None) -> pd.Series | None:
    """加载基准指数收盘价序列。

    优先读本地缓存;否则 akshare 下载并落盘。
    任何异常(网络/接口)返回 None,回测降级为无基准对比。

    Args:
        code: 指数代码,如 "000852"(中证1000)
        start/end: 日期范围 "YYYY-MM-DD"
        cache_path: 本地缓存 parquet 路径(列 date, close)

    Returns:
        Series (index=date, 值=收盘价) 或 None
    """
    try:
        if cache_path and os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["close"].sort_index()
            logger.info(f"基准 {code} 已从缓存加载: {len(s)} 个交易日")
            return s

        import akshare as ak
        logger.info(f"下载基准指数 {code} ...")
        df = ak.index_zh_a_hist(symbol=code, period="daily",
                                start_date=start.replace("-", ""),
                                end_date=end.replace("-", ""))
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].sort_index()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df[["date", "close"]].to_parquet(cache_path, index=False)
            logger.info(f"基准 {code} 已缓存: {cache_path}")
        return s
    except Exception as e:
        logger.warning(f"基准指数 {code} 加载失败({e}),回测无基准对比")
        return None


# ==================== 股票池等权基准 ====================

def build_universe_benchmark(daily_dict: dict[str, pd.DataFrame]
                             ) -> pd.Series:
    """用股票池日线构造等权基准(akshare 不可用时的兜底)。

    定义:每日等权持有全部有行情的股票(日频再平衡),即
    "满仓持有整个股票池"的组合收益——对 long-only 选股策略
    是比指数更直接的基准。

    Args:
        daily_dict: {symbol: 日线 DataFrame(含 '日期','收盘')}

    Returns:
        Series (index=date, 值=基准指数点位,起始=1.0)
    """
    closes = {}
    for sym, df in daily_dict.items():
        s = df.set_index(DATE_COL)[CLOSE_COL]
        s = s[~s.index.duplicated(keep="last")]
        closes[sym] = s
    panel = pd.DataFrame(closes).sort_index()
    # 日收益率 = 当日有数据股票收益率的等权平均(停牌/无数据跳过)
    daily_ret = panel.pct_change().mean(axis=1, skipna=True)
    benchmark = (1 + daily_ret.fillna(0)).cumprod()
    benchmark.name = "universe_ew"
    logger.info(f"股票池等权基准: {len(benchmark)} 个交易日, "
                f"{len(panel.columns)} 只股票")
    return benchmark
