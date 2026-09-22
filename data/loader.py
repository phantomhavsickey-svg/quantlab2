r"""
数据加载 — 读 quantlab 生成的缓存资产（路径来自 config 的 data 段）。

本仓库不自己抓个股行情:日线的下载与限速在 quantlab/data/downloader.py(并发+
令牌桶),这里只负责**读**。999 个 parquet 的读取是这条链路上唯一的 IO 大头,
所以 load_daily_dict 用线程池并行读盘(pyarrow 释放 GIL),数值与串行一致。
唯一例外的网络请求是基准指数(load_benchmark 一次请求,自带本地缓存)。

约定:
    - 因子面板: 长表 parquet,列 = 25 个因子 + date + symbol
      （已在 quantlab 处理阶段完成截面 zscore + lag 1,防未来函数）
    - 日线缓存: 每股一个 parquet,中文列名(日期/开盘/收盘/最高/最低/
      成交量/换手率/...),前复权,用于计算前向收益标签。成交量单位为"股"
      （quantlab 下载器已把东财/腾讯的"手"统一换算）
    - 基准指数: 腾讯直连优先(东财/akshare 的指数接口常被限流),akshare 兜底,
      两者都失败则降级为股票池等权基准(不阻断回测)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# 日线缓存列名(中文,来自 quantlab 下载器)
DATE_COL = "日期"
CLOSE_COL = "收盘"
VOL_COL = "成交量"
CHG_COL = "涨跌幅"

# 读盘的默认并发度:IO 型,线程数只要覆盖"同时在飞的 parquet 个数"即可
DEFAULT_LOAD_WORKERS = 8

# 撮合/盯市/可交易性判定要用的列。SequenceStore 只要 日期/收盘,但实盘引擎
# 与回测引擎还要 开盘 定价、成交量/涨跌幅 判停牌与涨跌停,不能裁到两列。
EXEC_DAILY_COLS = [DATE_COL, "开盘", "最高", "最低", CLOSE_COL,
                   VOL_COL, CHG_COL]


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
                    symbols: list[str],
                    columns: list[str] | None = None,
                    max_workers: int = DEFAULT_LOAD_WORKERS
                    ) -> dict[str, pd.DataFrame]:
    """加载日线缓存 {symbol: DataFrame(中文列名)}。

    只加载 symbols 列表中的股票(999 个 parquet 是主要 IO 耗时项,
    冒烟测试可用子集)。缺失文件 warn 并跳过。

    IO 并行:pyarrow 读 parquet 时释放 GIL,所以线程池能把"每个文件的
    打开+解压+反序列化"重叠起来。这只是把同一批文件读得更快 —— 每个文件
    只有一个线程碰它,校验/排序/返回顺序都在本函数内定死,数值与单线程
    逐字节相同,不影响任何已发布结果。

    Args:
        daily_dir: 日线缓存目录(内含 {symbol}.parquet)
        symbols: 需要加载的股票代码列表
        columns: 只读这些列。parquet 是列存,裁列就是裁 IO;
                 算标签传 [日期, 收盘] 足够,回测撮合要 开盘 就不能裁。
                 None = 全部列
        max_workers: 读盘并发线程数(1 = 退回串行)

    Returns:
        {symbol: DataFrame},日期升序、日期列为 datetime
    """
    daily_dir = Path(daily_dir)
    if not daily_dir.exists():
        raise FileNotFoundError(
            f"日线缓存目录不存在: {daily_dir}\n"
            f"请先在 quantlab 运行 python main.py download 下载日线,"
            f"或修改 config.yaml 的 data.daily_dir")

    todo = [(sym, daily_dir / f"{sym}.parquet") for sym in symbols]
    missing = [sym for sym, p in todo if not p.exists()]
    todo = [(sym, p) for sym, p in todo if p.exists()]

    def read(item):
        sym, path = item
        df = pd.read_parquet(path, columns=columns)
        if DATE_COL not in df.columns or CLOSE_COL not in df.columns:
            raise ValueError(f"日线文件缺少 {DATE_COL}/{CLOSE_COL} 列: {path}")
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        return df.sort_values(DATE_COL).reset_index(drop=True)

    t0 = time.monotonic()
    workers = max(1, min(int(max_workers), len(todo)))
    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(read, todo))     # map 保序 → 与 symbols 同序
    else:
        frames = [read(x) for x in todo]
    data = {sym: df for (sym, _), df in zip(todo, frames)}

    elapsed = time.monotonic() - t0
    if missing:
        logger.warning(f"{len(missing)} 只股票缺少日线文件(标签将为 NaN 并被过滤)")
    logger.info(f"加载日线: {len(data)} 只股票,"
                f"{'全部列' if columns is None else f'{len(columns)} 列'},"
                f" {workers} 线程, {elapsed:.1f}s")
    return data


# ==================== 基准指数 ====================

def _benchmark_from_tx(code: str, start: str, end: str) -> pd.Series:
    """腾讯直连取指数收盘价(一次请求 2000 根,够 8 年日线)。

    akshare 的指数接口是东财域名,本机 IP 常被限流(RemoteDisconnected);
    日期必须是 2026-09-22 这种带横线格式,传 20260922 接口会返回一个空 list。
    """
    import json
    import requests
    sym = code.lower() if code[:2] in ("sh", "sz") else \
        (f"sz{code}" if code.startswith("399") else f"sh{code}")
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    r = requests.get(url, params={"param": f"{sym},day,,{end},2000,"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    text = r.text
    i = text.find("={")
    node = json.JSONDecoder().raw_decode(text[i + 1:].strip())[0]["data"][sym]
    rows = node.get("day") or node.get("qfqday") or []
    df = pd.DataFrame({"date": [x[0] for x in rows],
                       "close": pd.to_numeric([x[2] for x in rows],
                                              errors="coerce")})
    df["date"] = pd.to_datetime(df["date"])
    s = df.drop_duplicates("date").set_index("date")["close"].sort_index()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if s.empty:
        raise RuntimeError(f"腾讯指数接口对 {sym} 返回空数据")
    return s


def load_benchmark(code: str, start: str, end: str,
                   cache_path: str | None = None) -> pd.Series | None:
    """加载基准指数收盘价序列。

    优先读本地缓存;否则腾讯直连,再退回 akshare(东财)。
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

        try:
            s = _benchmark_from_tx(code, start, end)
            logger.info(f"基准 {code} 腾讯直连: {len(s)} 个交易日")
            df = s.reset_index()
        except Exception as e:
            logger.warning(f"基准 {code} 腾讯直连失败({e}),改用 akshare")
            import akshare as ak
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

    注意股票池取的是**当前**成分列表,回溯早期带幸存者偏差,因此这条曲线
    系统性高于真实指数(Quantlab 2026-09-22 实测同区间等权 +55.7%、
    中证1000 指数 +12.2%)。它作为 hurdle 更严苛,不是更宽松。

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
