"""
序列数据集 — per-symbol 连续矩阵 + 位置滑窗 + 前向收益标签。

内存布局:999 × ~1360 × 25 float32 ≈ 136MB(全样本),绝不预切片成
(1.2M, seq_len, 25) 大张量(会放大到 2.5GB);__getitem__ 惰性切片。
样本索引和标签都是列式数组(见 Samples),不用 tuple 列表。

防未来函数规则(冒烟测试逐条断言):
    1. 因子面板已由 quantlab 做 lag 1:日期 t 的因子信息止于 close[t-1]
    2. 样本 t 的窗口 = 最近 seq_len 个有因子值的交易日(位置滑窗,
       停牌缺口不做日历对齐),窗口内所有日期 ≤ t
    3. 标签 = close[t+horizon]/close[t]-1(在日线空间计算,与 quantlab
       口径一致),只作为 y 绝不进入特征;标签按日截面缩尾只使用
       当日横截面(与未来无关)
    4. 长期停牌(标签窗口有效交易日 < min_valid_label_days)样本丢弃
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from data.loader import DATE_COL, CLOSE_COL


class Samples:
    """(sym_pos, t) 样本索引的列式表示。

    为什么不用 tuple 列表:1.2M 样本的 `[(si, t), ...]` 要 ~134 MB
    (每个 tuple 112 B 再加列表指针),两个 int32 数组只要 9.6 MB;更要紧的是
    标签能整段向量化取(省掉 N 次 Python 调用),而且 GPU 侧按索引 gather
    窗口本来就需要整数张量,不需要 Python tuple。
    """

    __slots__ = ("sym_pos", "t", "dates")

    def __init__(self, sym_pos, t, dates):
        self.sym_pos = np.asarray(sym_pos, dtype=np.int32)
        self.t = np.asarray(t, dtype=np.int32)
        self.dates = np.asarray(dates, dtype="datetime64[ns]")

    def __len__(self) -> int:
        return int(self.sym_pos.shape[0])

    def __getitem__(self, i) -> tuple[int, int]:
        """单个样本的 (sym_pos, t) —— 抽样断言用,勿进热循环。"""
        return int(self.sym_pos[i]), int(self.t[i])

    def select(self, mask) -> "Samples":
        """按布尔掩码或下标数组取子集(折内 train/valid/OOS 切分)。"""
        return Samples(self.sym_pos[mask], self.t[mask], self.dates[mask])

    @property
    def keys(self) -> np.ndarray:
        """打包成单个 int64 的样本身份,供集合运算(重叠检查)。"""
        return (self.sym_pos.astype(np.int64) << 32) | self.t.astype(np.int64)

    @property
    def date_ints(self) -> np.ndarray:
        """epoch 天数 (int64) —— 按日截面分组算 Rank IC 用。"""
        return self.dates.astype("datetime64[D]").astype(np.int64)

    def labels(self, store: "SequenceStore") -> np.ndarray:
        """整段向量化取标签,返回与自身同序的 (N,) float32。"""
        return store.label_flat[store.label_offsets[self.sym_pos] + self.t]


class SequenceStore:
    """将因子面板长表转为 {symbol: (T, F) float32 矩阵} 并生成前向收益标签。

    属性:
        factor_names: 因子列名
        n_features: 因子数(25)
        symbols: 排序后的股票代码列表
        dates_by_symbol: {symbol: (T,) datetime64 数组}
        factors_by_symbol: {symbol: (T, F) float32 矩阵,已填充无 NaN}
        label_flat: (总行数,) float32 前向收益(已截面缩尾),尾部 horizon
                    行为 NaN;按 symbols 排序顺序拼接
        valid_flat: (总行数,) int16 标签窗口内有效交易日数,同上顺序
        label_offsets: (n_symbols + 1,) int64 —— symbol si 的行是
                       label_flat[label_offsets[si]:label_offsets[si+1]]
        global_dates: 全部出现过的交易日(sorted datetime64 数组)
    """

    def __init__(self, factor_panel: pd.DataFrame,
                 daily_dict: dict[str, pd.DataFrame],
                 cfg: dict):
        """
        Args:
            factor_panel: 长表因子面板(含 date/symbol 列)
            daily_dict: {symbol: 日线 DataFrame(中文列名,含 '日期','收盘')}
            cfg: config dict,取 model.horizon / model.seq_len /
                 training.label_winsor / data.min_valid_label_days
        """
        m = cfg["model"]
        t = cfg["model"]["training"]
        d = cfg["data"]

        self.horizon = int(m["horizon"])
        self.seq_len = int(m["seq_len"])
        self.label_winsor = tuple(t.get("label_winsor", [0.01, 0.99]))
        self.min_valid_label_days = int(d.get("min_valid_label_days", 15))

        self.factor_names = [c for c in factor_panel.columns
                             if c not in ["date", "symbol"]]
        self.n_features = len(self.factor_names)
        logger.info(f"构建序列存储: seq_len={self.seq_len}, "
                    f"horizon={self.horizon}, 因子数={self.n_features}")

        self._pivot(factor_panel)
        self._build_labels(daily_dict)

    # ==================== 面板转 per-symbol 矩阵 ====================

    def _pivot(self, factor_panel: pd.DataFrame):
        """长表 → {symbol: (T, F) float32 连续矩阵}。

        NaN 处理:每 symbol 内按行 ffill → 残余 fillna(0.0)。
        面板已截面 zscore,0 = 当日截面均值,是无偏填充,且不引入
        需要"拟合"的统计量(零泄漏)。填充在构建时一次性完成。
        """
        self.dates_by_symbol = {}
        self.factors_by_symbol = {}
        self.symbols = []

        for sym, grp in factor_panel.groupby("symbol", sort=True):
            grp = grp.sort_values("date")
            dates = grp["date"].to_numpy()
            mat = grp[self.factor_names].to_numpy(dtype=np.float32)

            # 停牌/上市初期可能产生 NaN:先按时间 ffill,残余填 0(截面均值)
            mat = pd.DataFrame(mat).ffill().fillna(0.0).to_numpy(np.float32)

            self.symbols.append(sym)
            self.dates_by_symbol[sym] = dates
            self.factors_by_symbol[sym] = np.ascontiguousarray(mat)

        self.global_dates = np.unique(
            np.concatenate(list(self.dates_by_symbol.values())))
        n_rows = sum(len(d) for d in self.dates_by_symbol.values())
        logger.info(f"per-symbol 矩阵构建完成: {len(self.symbols)} 只股票, "
                    f"共 {n_rows:,} 行, {len(self.global_dates)} 个交易日")

    # ==================== 标签构建 ====================

    def _build_labels(self, daily_dict: dict[str, pd.DataFrame]):
        """对每只股票计算前向 horizon 日收益标签并做截面缩尾。

        标签在日线空间(该股票的日历交易日序列)计算,再对齐回因子
        日期,与 quantlab DatasetBuilder.build_label 口径一致:
            label(t) = close[t+horizon] / close[t] - 1
        """
        raw_labels = {}
        raw_valid = {}

        for sym in self.symbols:
            dates = self.dates_by_symbol[sym]
            T = len(dates)
            df = daily_dict.get(sym)

            if df is None or df.empty:
                raw_labels[sym] = np.full(T, np.nan, dtype=np.float32)
                raw_valid[sym] = np.zeros(T, dtype=np.int16)
                continue

            close = df.set_index(DATE_COL)[CLOSE_COL]
            close = close[~close.index.duplicated(keep="last")]

            # 因子日期对齐到日线索引(因子日期应为日线日期的子集)
            factor_dates = pd.Index(dates)
            close_aligned = close.reindex(factor_dates)

            # 前向收益(日线索引位置平移 horizon 个交易日)
            fwd = close_aligned.shift(-self.horizon) / close_aligned - 1.0

            # 标签窗口 [pos, pos+horizon] 内有效收盘价个数(向量化):
            # NaN 个数 = cum_nan[pos+h] - cum_nan[pos] + isnan[pos]
            is_nan = close.isna().astype(np.float64)
            cum_nan = is_nan.cumsum()
            fwd_nan = (cum_nan.shift(-self.horizon, fill_value=cum_nan.iloc[-1])
                       - cum_nan + is_nan)
            valid = ((self.horizon + 1) - fwd_nan).clip(0, self.horizon + 1)

            raw_labels[sym] = fwd.to_numpy(dtype=np.float32)
            raw_valid[sym] = valid.reindex(factor_dates).to_numpy(
                dtype=np.int16)

        # --- 标签按日截面缩尾(只使用当日横截面,与未来无关) ---
        date_concat = np.concatenate([self.dates_by_symbol[s]
                                      for s in self.symbols])
        lab_concat = np.concatenate([raw_labels[s] for s in self.symbols])
        val_concat = np.concatenate([raw_valid[s] for s in self.symbols])

        df = pd.DataFrame({"d": date_concat, "lab": lab_concat})
        df["lab"] = df.groupby("d")["lab"].transform(
            lambda x: x.clip(*x.quantile(self.label_winsor))
            if x.notna().any() else x)

        lengths = np.fromiter((len(self.dates_by_symbol[s]) for s in self.symbols),
                              dtype=np.int64, count=len(self.symbols))
        self.label_offsets = np.concatenate([[0], np.cumsum(lengths)])
        self.label_flat = df["lab"].to_numpy(dtype=np.float32)
        self.valid_flat = val_concat

        n_lab = int(np.isfinite(self.label_flat).sum())
        logger.info(f"标签构建完成: {n_lab:,} 条有效前向收益标签"
                    f"(缩尾 [{self.label_winsor[0]:.0%}, "
                    f"{self.label_winsor[1]:.0%}])")

    # ==================== 样本枚举 ====================

    def _candidate_index(self, min_date, max_date, require_label: bool
                         ) -> Samples:
        """按窗口完整性(可选标签有效性)枚举样本。

        Args:
            require_label: True 时额外要求标签有效(训练/评估用);
                           False 时仅要求窗口完整(推理用,最新
                           horizon 个交易日没有标签也能预测)
        """
        sym_parts, t_parts, date_parts = [], [], []

        for si, sym in enumerate(self.symbols):
            dates = self.dates_by_symbol[sym]
            T = len(dates)
            if T <= self.seq_len:
                continue

            lo, hi = self.label_offsets[si], self.label_offsets[si + 1]
            ok = np.zeros(T, dtype=bool)
            ok[self.seq_len - 1:] = True
            if require_label:
                ok &= np.isfinite(self.label_flat[lo:hi])
                ok &= self.valid_flat[lo:hi] >= self.min_valid_label_days
            if min_date is not None:
                ok &= dates >= np.datetime64(min_date)
            if max_date is not None:
                ok &= dates <= np.datetime64(max_date)

            ts = np.nonzero(ok)[0]
            if ts.size:
                sym_parts.append(np.full(ts.size, si, dtype=np.int32))
                t_parts.append(ts.astype(np.int32))
                date_parts.append(dates[ts])

        if not date_parts:
            return Samples(np.empty(0, np.int32), np.empty(0, np.int32),
                           np.empty(0, dtype="datetime64[ns]"))
        return Samples(np.concatenate(sym_parts), np.concatenate(t_parts),
                       np.concatenate(date_parts))

    def sample_index(self,
                     min_date: np.datetime64 | pd.Timestamp | None = None,
                     max_date: np.datetime64 | pd.Timestamp | None = None
                     ) -> Samples:
        """枚举所有可训练样本。

        条件:
            - t ≥ seq_len-1(窗口完整)
            - 标签非 NaN(日线空间 t+horizon 存在)
            - 标签窗口有效交易日 ≥ min_valid_label_days(剔除长期停牌)
            - min_date ≤ date[t] ≤ max_date(含端点;Walk-Forward 用
              max_date = ws - 1 天实现严格早于 OOS 窗口)

        Args:
            min_date/max_date: 可选日期范围(含端点)

        Returns:
            Samples —— sym_pos/t/dates 三个数组同序
        """
        return self._candidate_index(min_date, max_date, require_label=True)

    def inference_index(self,
                        min_date: np.datetime64 | pd.Timestamp | None = None,
                        max_date: np.datetime64 | pd.Timestamp | None = None
                        ) -> Samples:
        """枚举所有可推理样本(不要求标签)。

        用于实盘信号:数据尾部 horizon 个交易日没有标签,但仍可预测。
        """
        return self._candidate_index(min_date, max_date, require_label=False)

    # ==================== 窗口切片 ====================

    def get_window(self, sym_pos: int, t: int) -> np.ndarray:
        """返回样本 (sym_pos, t) 的因子序列窗口 (seq_len, F) float32。

        窗口 = 最近 seq_len 个有因子值的交易日,位置 0 最远,
        位置 seq_len-1 为样本当日 t。
        """
        mat = self.factors_by_symbol[self.symbols[sym_pos]]
        # .copy() 保证可写且 C 连续(torch.from_numpy 需要)
        return mat[t - self.seq_len + 1: t + 1].copy()

    def get_label(self, sym_pos: int, t: int) -> float:
        """返回样本 (sym_pos, t) 的前向收益标签。"""
        return float(self.label_flat[self.label_offsets[sym_pos] + t])

    def symbols_of(self, samples: Samples) -> np.ndarray:
        """样本对应的股票代码数组(与 samples 同序)。"""
        return np.take(self.symbols, samples.sym_pos)


class SequenceDataset(Dataset):
    """惰性切片数据集。

    __getitem__ 返回 (x, y, date_int):
        x: (seq_len, F) float32 因子序列
        y: 标量 float32 前向收益
        date_int: 样本日期(epoch 天数 int64,评估 Rank IC 按日分组用)
    """

    def __init__(self, store: SequenceStore, samples: Samples):
        self.store = store
        self.sym_pos = samples.sym_pos
        self.t = samples.t
        # 标签和日期整段预取:每样本一次 Python 调用换成一次 fancy index
        self.labels = samples.labels(store)
        self.date_ints = samples.date_ints

    def __len__(self) -> int:
        return int(self.sym_pos.shape[0])

    def __getitem__(self, i: int) -> tuple:
        x = torch.from_numpy(self.store.get_window(int(self.sym_pos[i]),
                                                   int(self.t[i])))
        return (x, torch.tensor(self.labels[i], dtype=torch.float32),
                self.date_ints[i])


def make_loader(store: SequenceStore,
                samples: Samples,
                batch_size: int,
                shuffle: bool = False,
                num_workers: int = 0,
                pin_memory: bool = False,
                drop_last: bool = False,
                seed: int = 42) -> DataLoader:
    """构建一个 DataLoader(标签请用 samples.labels(store) 单独取)。

    Args:
        num_workers: >0 时自动开 persistent_workers。Windows 是 spawn 启动,
            每个 worker 要把整个 store(~136 MB)重新 pickle 一遍,不持久化
            就每 epoch 重付一次这笔钱;要真正提速得把取窗口挪到 GPU 上,
            所以这里只是把 config 的值接进来,不擅自改默认。
        seed: shuffle 复现用
    """
    ds = SequenceDataset(store, samples)
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      pin_memory=pin_memory, drop_last=drop_last,
                      generator=gen)


# ==================== Walk-Forward 窗口 ====================

@dataclass
class FoldWindow:
    """一个 Walk-Forward 折的 OOS 预测窗口。"""
    fold: int
    ws: pd.Timestamp   # OOS 窗口起点(含)
    we: pd.Timestamp   # OOS 窗口终点(不含)
    n_days: int = 0    # 窗口内实际交易日数


def walk_forward_windows(all_dates: np.ndarray, min_train_months: int,
                         retrain_months: int,
                         min_oos_days: int = 20) -> list[FoldWindow]:
    """生成 Walk-Forward OOS 窗口序列。

    与 quantlab 同构:首个窗口 = 数据起点 + min_train_months 之后,
    之后每 retrain_months 滚动;窗口起点超出最后日期即停止。

    尾部不足 min_oos_days 个交易日的窗口直接丢弃:几日的截面 IC 噪声极大,
    而折均值是按折(而非按日)平均的,一个 8 日折能把头条指标从 0.056 抬到
    0.114(实测于 2026-08-15 的 Fold 8,OOS_IC = 0.5215)。

    Args:
        all_dates: 全部交易日 datetime64 数组(无序亦可)
        min_train_months: 首个 OOS 窗口前的最短训练期(月)
        retrain_months: 每次滚动月数
        min_oos_days: 单折 OOS 窗口最少交易日数

    Returns:
        [FoldWindow, ...]
    """
    dates = pd.DatetimeIndex(np.unique(all_dates)).sort_values()
    start = dates[0]
    windows = []
    fold = 1
    dropped = []
    current = start + pd.DateOffset(months=min_train_months)

    while True:
        later = dates[dates >= current]
        if len(later) == 0:
            break
        ws = later[0]
        we = ws + pd.DateOffset(months=retrain_months)
        n_days = int(((dates >= ws) & (dates < we)).sum())
        if n_days < min_oos_days:
            dropped.append((ws, n_days))
        else:
            windows.append(FoldWindow(fold=fold, ws=ws, we=we, n_days=n_days))
            fold += 1
        current = we
        if we > dates[-1]:
            break

    for ws, n_days in dropped:
        logger.warning(f"丢弃尾部折 {ws.date()}: OOS 窗口仅 {n_days} 个交易日"
                       f" < min_oos_days={min_oos_days}")
    if not windows:
        raise RuntimeError(
            f"没有可用折:数据不足(min_train_months={min_train_months},"
            f"min_oos_days={min_oos_days})")
    logger.info(f"Walk-Forward 划分: {len(windows)} 折 "
                f"({windows[0].ws.date()} → {windows[-1].we.date()},"
                f" 共 {sum(w.n_days for w in windows)} 个 OOS 交易日)")
    return windows
