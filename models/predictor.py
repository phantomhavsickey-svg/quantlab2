"""
预测器 — 模型批量推理 → 截面排名 → Top-K 选股 → 交易信号。

信号输出 schema 与 quantlab 的 Predictor 完全一致:
    DataFrame MultiIndex (date, symbol),列 = [score, rank, weight]
    score  = 预测前向收益(越高越好)
    rank   = 截面降序名次
    weight = 持仓权重(等权 1/top_k 或按信号强度归一化)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from models.sequence_data import SequenceStore, Samples, make_loader


def _normalize_multiindex(mi: pd.MultiIndex) -> pd.MultiIndex:
    """规范化 MultiIndex 层级:date → datetime64[ns],symbol → object。

    不同来源构造的 MultiIndex 层级 dtype 可能不一致(str vs object、
    datetime64[s] vs datetime64[ns] vs object-of-Timestamp),混用时
    reindex 会静默失配(权重全部变 0)。统一 dtype 后按值对齐。
    """
    dates = pd.to_datetime(mi.get_level_values(0)).astype("datetime64[ns]")
    syms = mi.get_level_values(1).astype(str).astype(object)
    return pd.MultiIndex.from_arrays([dates, syms], names=mi.names)


def signals_from_predictions(predictions: pd.Series, top_k: int,
                             position_sizing: str = "equal_weight",
                             tradable: pd.Series | None = None
                             ) -> pd.DataFrame:
    """从预测值 Series 生成选股信号(与 quantlab 语义一致)。

    Args:
        predictions: Series MultiIndex (date, symbol),值为预测收益/分数
        top_k: 每期持仓股票数
        position_sizing: "equal_weight"(等权) / "signal_strength"(按强度加权)
        tradable: 可交易性布尔 Series(同索引),可选

    Returns:
        DataFrame MultiIndex (date, symbol),列 [score, rank, weight]
    """
    if len(predictions) == 0:
        logger.warning("预测为空,无法生成信号")
        return pd.DataFrame(columns=["score", "rank", "weight"],
                            index=pd.MultiIndex.from_arrays(
                                [[], []], names=["date", "symbol"]))

    if isinstance(predictions.index, pd.MultiIndex):
        predictions = predictions.copy()
        predictions.index = _normalize_multiindex(predictions.index)

    # 1. 截面降序排名(高分在前)
    rankings = predictions.groupby(level="date").rank(ascending=False)

    # 2. Top-K 选股 + 权重
    # 注意:day/top 的索引是 MultiIndex (date, symbol),遍历索引得到的
    # 是 (date, symbol) 元组,必须显式取 symbol 层级,并用位置索引取值
    rows = []
    for d, day in predictions.groupby(level="date"):
        day = day.dropna()
        if day.empty:
            continue
        # 过滤不可交易股票(按 symbol 层级)
        if tradable is not None and isinstance(tradable.index, pd.MultiIndex):
            try:
                day_tradable = tradable.xs(d, level="date")
                allowed = day_tradable[day_tradable].index
                mask = day.index.get_level_values("symbol").isin(allowed)
                day = day[mask]
            except KeyError:
                pass
        if day.empty:
            continue

        top = day.nlargest(top_k)
        if position_sizing == "signal_strength" and top.sum() > 0:
            weights = top / top.sum()
        else:
            weights = pd.Series(1.0 / len(top), index=top.index)

        for i, sym in enumerate(top.index.get_level_values("symbol")):
            rows.append({"date": d, "symbol": sym,
                         "weight": weights.iloc[i], "score": top.iloc[i]})

    if not rows:
        logger.warning("没有生成任何选股信号")
        return pd.DataFrame(columns=["score", "rank", "weight"],
                            index=pd.MultiIndex.from_arrays(
                                [[], []], names=["date", "symbol"]))

    weights_df = pd.DataFrame(rows).set_index(["date", "symbol"])
    weights_df.index = _normalize_multiindex(weights_df.index)

    # 3. 合并
    signals = pd.DataFrame({"score": predictions, "rank": rankings})
    signals["weight"] = weights_df["weight"].reindex(signals.index).fillna(0.0)

    logger.info(f"信号生成完毕: {(signals['weight'] > 0).sum()} 条持仓信号,"
                f" 覆盖 {signals.index.get_level_values('date').nunique()} 个交易日")
    return signals


class TransformerPredictor:
    """Transformer 模型预测器(批量推理 + 信号生成)。"""

    def __init__(self, model: nn.Module, store: SequenceStore,
                 cfg: dict, device: torch.device,
                 batch_size: int = 4096):
        """
        Args:
            model: 已训练模型(内部会 .to(device).eval())
            store: SequenceStore
            cfg: 完整 config dict(取 predict 段的 top_k/position_sizing)
            device: torch.device
            batch_size: 推理批大小(无梯度,可开大)
        """
        self.model = model.to(device).eval()
        self.store = store
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self.num_workers = int(cfg["model"]["training"].get("num_workers", 0))

    # ==================== 批量推理 ====================

    def _predict_indices(self, samples: Samples) -> pd.Series:
        """对样本集合批量推理。

        Returns:
            Series MultiIndex (date, symbol),name="prediction"
        """
        if len(samples) == 0:
            return pd.Series(dtype=float,
                             name="prediction",
                             index=pd.MultiIndex.from_arrays(
                                 [[], []], names=["date", "symbol"]))
        loader = make_loader(self.store, samples,
                             batch_size=self.batch_size, shuffle=False,
                             num_workers=self.num_workers)
        with torch.no_grad():
            preds = self._run_loader(loader)
        index = pd.MultiIndex.from_arrays(
            [samples.dates, self.store.symbols_of(samples)],
            names=["date", "symbol"])
        return pd.Series(preds, index=index, name="prediction")

    def predict(self, min_date=None, max_date=None) -> pd.Series:
        """对 store 中可建窗口的全部样本(可选日期范围)批量推理。

        注意:用 inference_index(不要求标签),数据尾部也能预测。
        """
        samples = self.store.inference_index(min_date, max_date)
        logger.info(f"推理样本数: {len(samples):,}")
        return self._predict_indices(samples)

    def predict_asof(self, asof: pd.Timestamp | str) -> pd.Series:
        """对 ≤ asof 的最近一个截面完整的交易日做全截面预测。

        不要求标签存在,因此数据最新日期(无前向收益)也能出信号。
        最新交易日可能只有部分股票更新了因子(数据源异步),
        故在最近 5 个交易日中选截面覆盖股票数最多的一天。
        """
        d = np.datetime64(pd.Timestamp(asof))
        avail = self.store.global_dates[self.store.global_dates <= d]
        if len(avail) == 0:
            raise ValueError(f"asof={asof} 早于全部数据")

        target = avail[-1]
        best_n = self._n_symbols_on(target)
        for c in avail[-5:-1]:
            n = self._n_symbols_on(c)
            if n > best_n:
                target, best_n = c, n

        samples = self.store.inference_index(min_date=target, max_date=target)
        if len(samples) == 0:
            raise RuntimeError(f"日期 {pd.Timestamp(target).date()} "
                               f"无法构建样本窗口(seq_len 不足?)")
        logger.info(f"asof={asof} → 截面日期 "
                    f"{pd.Timestamp(target).date()}, {len(samples)} 只股票")
        return self._predict_indices(samples)

    def _n_symbols_on(self, d: np.datetime64) -> int:
        """统计某交易日有因子行的股票数(截面覆盖率)。"""
        n = 0
        for dates in self.store.dates_by_symbol.values():
            i = np.searchsorted(dates, d)
            if i < len(dates) and dates[i] == d:
                n += 1
        return n

    @torch.no_grad()
    def _run_loader(self, loader) -> np.ndarray:
        """对 loader 逐批推理(模型需已 eval)。"""
        preds = []
        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type,
                                enabled=(self.device.type == "cuda"
                                         and self.cfg["model"]["training"]
                                         .get("use_amp", True))):
                pred = self.model(x)
            preds.append(pred.squeeze(-1).float().cpu().numpy())
        if not preds:
            return np.array([], dtype=np.float32)
        return np.concatenate(preds)

    # ==================== 信号生成 ====================

    def generate_signals(self, predictions: pd.Series | None = None,
                         asof=None,
                         tradable: pd.Series | None = None
                         ) -> pd.DataFrame:
        """完整信号生成管道。

        Args:
            predictions: 外部预测 Series(为 None 时自动推理;
                         asof 给定时用 predict_asof)
            asof: 目标交易日(与 predictions 二选一)
            tradable: 可交易性掩码(可选)

        Returns:
            DataFrame MultiIndex (date, symbol),列 [score, rank, weight]
        """
        if predictions is None:
            predictions = (self.predict_asof(asof) if asof is not None
                           else self.predict())

        cfg_pred = self.cfg["predict"]
        return signals_from_predictions(
            predictions, top_k=cfg_pred["top_k"],
            position_sizing=cfg_pred["position_sizing"],
            tradable=tradable)
