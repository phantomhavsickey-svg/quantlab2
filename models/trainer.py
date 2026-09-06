"""
训练器 — 训练循环 / Rank IC 评估 / 早停 / checkpoint / Walk-Forward。

关键设计:
    - 早停指标 = 验证集按日截面 Spearman Rank IC(比 loss 更贴合选股
      目标);验证集 IC 不可算(样本不足)时回退用 -loss
    - AMP 仅 cuda;每折结束 empty_cache;OOM 自动减半 batch 重试一次
    - checkpoint 只存 state_dict + JSON 兼容元数据
      (torch ≥ 2.6 默认 weights_only=True 可安全加载)
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from models.transformer import build_model
from models.sequence_data import SequenceStore, make_loaders, \
    walk_forward_windows


# ==================== Rank IC ====================

def _group_bounds(date_ints: np.ndarray) -> list[tuple[int, int]]:
    """按日期整数值分组,返回每组 [start, end) 区间。

    Args:
        date_ints: (N,) int64 日期(epoch 天数),任意顺序

    Returns:
        [(start, end), ...] 按日期升序
    """
    order = np.argsort(date_ints, kind="stable")
    d = date_ints[order]
    bounds = np.nonzero(d[1:] != d[:-1])[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(d)]])
    # 返回排序后的组区间(调用方持有 order)
    return order, starts, ends


def compute_rank_ic(pred: np.ndarray, y_true: np.ndarray,
                    date_ints: np.ndarray) -> dict:
    """按日截面计算 Spearman Rank IC。

    规则:当日样本数 < 30 或 pred/y 任一方 std < 1e-8 → 跳过
    (记入 skipped),避免无方差截面产生 NaN IC。

    Args:
        pred: (N,) 预测值
        y_true: (N,) 真实标签
        date_ints: (N,) int64 日期(epoch 天数)

    Returns:
        {mean_ic, icir, ic_by_date (pd.Series), n_days, skipped}
    """
    from scipy.stats import spearmanr

    order, starts, ends = _group_bounds(date_ints)
    p = pred[order]
    y = y_true[order]
    d = date_ints[order]

    ics = {}
    skipped = 0
    for s, e in zip(starts, ends):
        if e - s < 30:
            skipped += 1
            continue
        ps, ys = p[s:e], y[s:e]
        if ps.std() < 1e-8 or ys.std() < 1e-8:
            skipped += 1
            continue
        ic = spearmanr(ps, ys).correlation
        if not np.isfinite(ic):
            skipped += 1
            continue
        ics[int(d[s])] = ic

    ic_series = pd.Series(ics)
    ic_series.index = pd.to_datetime(ic_series.index, unit="D")
    if len(ic_series) == 0:
        return {"mean_ic": np.nan, "icir": np.nan,
                "ic_by_date": ic_series, "n_days": 0, "skipped": skipped}
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    icir = mean_ic / std_ic if std_ic > 1e-8 else np.nan
    return {"mean_ic": mean_ic, "icir": icir,
            "ic_by_date": ic_series, "n_days": len(ic_series),
            "skipped": skipped}


def compute_daily_auc(pred: np.ndarray, y_binary: np.ndarray,
                      date_ints: np.ndarray) -> dict:
    """按日截面计算平均 AUC(仅 classifier 模式使用)。

    Returns:
        {mean_auc, n_days, skipped}
    """
    from sklearn.metrics import roc_auc_score

    order, starts, ends = _group_bounds(date_ints)
    p = pred[order]
    y = y_binary[order]

    aucs = []
    skipped = 0
    for s, e in zip(starts, ends):
        if e - s < 30:
            skipped += 1
            continue
        if len(np.unique(y[s:e])) < 2:
            skipped += 1
            continue
        try:
            aucs.append(float(roc_auc_score(y[s:e], p[s:e])))
        except ValueError:
            skipped += 1
    if not aucs:
        return {"mean_auc": np.nan, "n_days": 0, "skipped": skipped}
    return {"mean_auc": float(np.mean(aucs)), "n_days": len(aucs),
            "skipped": skipped}


# ==================== 早停 ====================

class EarlyStopping:
    """早停:patience 轮无改善即停止(统一用 max 方向,调用方取反)。"""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best = -math.inf
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, metric: float, epoch: int) -> bool:
        """返回 True 表示应停止。"""
        if metric > self.best:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


# ==================== 训练器 ====================

class TransformerTrainer:
    """PyTorch 训练器:训练循环 + Rank IC 评估 + 早停 + checkpoint。"""

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device,
                 seed: int = 42):
        """
        Args:
            model: 待训练模型(未加载到设备也可,内部会 .to(device))
            cfg: 完整 config dict
            device: torch.device
            seed: 随机种子(DataLoader shuffle 复现用)
        """
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.model_type = cfg["model"]["type"]
        self.n_features = model.input_proj.in_features
        self.factor_names = []

        tr = cfg["model"]["training"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tr["lr"],
            weight_decay=tr["weight_decay"], eps=1e-8)
        self.grad_clip = tr.get("grad_clip", 1.0)
        self.loss_name = tr.get("loss", "mse")
        self.use_amp = bool(tr.get("use_amp", True)) and device.type == "cuda"
        self.scheduler = None
        self.global_step = 0
        self._reset_optimizer()

    def _reset_optimizer(self):
        """(重新)绑定优化器到当前模型参数。

        断点续训 load_checkpoint 会用新模型替换 self.model,
        此时必须重建优化器/AMP scaler,否则 optimizer 仍持有旧模型
        参数(梯度不更新新模型,AMP 会报
        'No inf checks were recorded for this optimizer')。
        """
        tr = self.cfg["model"]["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tr["lr"],
            weight_decay=tr["weight_decay"], eps=1e-8)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scheduler = None
        self.global_step = 0

    # ==================== 目标与损失 ====================

    def _target(self, y: torch.Tensor) -> torch.Tensor:
        """classifier 模式把连续前向收益转为 0/1 涨跌标签。"""
        if self.model_type == "classifier":
            return (y > 0).float()
        return y

    def _loss_fn(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.model_type == "classifier":
            return nn.functional.binary_cross_entropy_with_logits(
                pred.squeeze(-1), y)
        if self.loss_name == "huber":
            return nn.functional.smooth_l1_loss(pred.squeeze(-1), y, beta=0.1)
        return nn.functional.mse_loss(pred.squeeze(-1), y)

    # ==================== 学习率调度 ====================

    def _build_scheduler(self, total_steps: int):
        """warmup 线性升温 → cosine 衰减到 lr_min。"""
        tr = self.cfg["model"]["training"]
        warmup = tr.get("warmup_steps", 1000)
        lr_min = tr.get("lr_min", 1e-5)
        lr = tr["lr"]

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            return lr_min / lr + (1 - lr_min / lr) * cos

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ==================== 训练/评估循环 ====================

    def train_one_epoch(self, loader, epoch: int) -> float:
        """训练一个 epoch,返回平均 loss。"""
        self.model.train()
        total_loss = 0.0
        n = 0
        bar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch in bar:
            x = batch[0].to(self.device, non_blocking=True)
            y = self._target(batch[1].to(self.device, non_blocking=True))

            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp):
                pred = self.model(x)
                loss = self._loss_fn(pred, y)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            self.global_step += 1

            lr = self.optimizer.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{lr:.2e}")

        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """在验证集上评估,返回 {loss, mean_ic, icir, n_samples, ...}。"""
        self.model.eval()
        total_loss = 0.0
        n = 0
        preds, ys, ds = [], [], []
        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            y = self._target(batch[1].to(self.device, non_blocking=True))
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp):
                pred = self.model(x)
                loss = self._loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(pred.squeeze(-1).float().cpu().numpy())
            ys.append(y.cpu().numpy())
            ds.append(batch[2].numpy())

        pred = np.concatenate(preds)
        y_true = np.concatenate(ys)
        dates = np.concatenate(ds)

        result = {"loss": total_loss / max(n, 1), "n_samples": n}
        result.update(compute_rank_ic(pred, y_true, dates))
        if self.model_type == "classifier":
            result.update(compute_daily_auc(pred, y_true, dates))
        return result

    @torch.no_grad()
    def predict_loader(self, loader) -> np.ndarray:
        """批量推理,返回 (N,) float32 预测值。

        注意:classifier 模式返回 logits(与 sigmoid 概率同序,
        用于截面排名不受影响)。
        """
        self.model.eval()
        preds = []
        bar = tqdm(loader, desc="推理", leave=False)
        for batch in bar:
            x = batch[0].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_amp):
                pred = self.model(x)
            preds.append(pred.squeeze(-1).float().cpu().numpy())
        if not preds:
            return np.array([], dtype=np.float32)
        return np.concatenate(preds)

    # ==================== 主训练循环 ====================

    def fit(self, train_loader, valid_loader, epochs: int,
            fold: int, save_dir: str) -> dict:
        """训练主循环,早停指标 = 验证集 Rank IC(不可算时回退 -loss)。

        Returns:
            {fold, best_epoch, best_rank_ic, best_loss, early_stopped,
             path, history}
        """
        tr = self.cfg["model"]["training"]
        self.global_step = 0
        self.scheduler = self._build_scheduler(epochs * len(train_loader))
        es = EarlyStopping(patience=int(tr.get("early_stop_patience", 10)))

        best_metric = -math.inf
        best_state = None
        best_epoch = 0
        best_ic = np.nan
        best_loss = np.nan
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            valid = self.evaluate(valid_loader)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            **{k: v for k, v in valid.items()
                               if k not in ("ic_by_date",)}})

            # 早停指标:rank_ic(默认,越大越好);IC 不可算时回退 -loss
            if tr.get("early_stop_metric", "rank_ic") == "loss":
                metric = -valid["loss"]
            elif np.isfinite(valid["mean_ic"]):
                metric = valid["mean_ic"]
            else:
                metric = -valid["loss"]

            logger.info(
                f"Fold {fold} Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.6f} | valid_loss={valid['loss']:.6f}"
                f" | valid_IC={valid['mean_ic']:.4f}"
                f" | ICIR={valid['icir']:.2f}")

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                best_ic = valid["mean_ic"]
                best_loss = valid["loss"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                self.save_checkpoint(
                    os.path.join(save_dir, "best.pt"), epoch, valid,
                    is_best=True)

            if es(metric, epoch):
                logger.info(f"Fold {fold} 早停触发: {es.patience} 轮无改善"
                            f" (epoch {epoch}, best={es.best:.4f} @ "
                            f"{es.best_epoch})")
                break

        # 恢复 best 权重
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.save_checkpoint(os.path.join(save_dir, "last.pt"),
                             epoch, history[-1], is_best=False)

        return {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_rank_ic": best_ic,
            "best_loss": best_loss,
            "early_stopped": epoch < epochs,
            "path": os.path.join(save_dir, "best.pt"),
            "history": history,
        }

    # ==================== checkpoint ====================

    def save_checkpoint(self, path: str, epoch: int, metrics: dict,
                        is_best: bool = False):
        """保存 checkpoint(仅 state_dict + JSON 兼容元数据)。"""
        payload = {
            "model_state": self.model.state_dict(),
            "meta": {
                "cfg": self.cfg,  # yaml.safe_load 结果,JSON 兼容
                "model_type": self.model_type,
                "factor_names": self.factor_names,
                "seq_len": self.cfg["model"]["seq_len"],
                "n_features": self.n_features,
                "epoch": epoch,
                "metrics": {k: v for k, v in metrics.items()
                            if k != "ic_by_date"},
                "seed": self.seed,
                "torch_version": str(torch.__version__),  # 纯 str,weights_only 安全
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)
        tag = "best" if is_best else "last"
        logger.info(f"Fold checkpoint 已保存: {path} ({tag})")

    @staticmethod
    def load_checkpoint(path: str, device: torch.device
                        ) -> tuple[nn.Module, dict]:
        """加载 checkpoint,用 meta 中的 cfg 重建模型。

        Returns:
            (model, meta): 模型已 .to(device).eval()
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        payload = torch.load(path, map_location=device, weights_only=True)
        meta = payload["meta"]
        model = build_model(meta["cfg"]["model"],
                            n_features=meta["n_features"])
        model.load_state_dict(payload["model_state"])
        model.to(device).eval()
        logger.info(f"checkpoint 已加载: {path}"
                    f" (epoch={meta['epoch']}, "
                    f"best_IC={meta['metrics'].get('mean_ic', np.nan):.4f})")
        return model, meta

    # ==================== Walk-Forward ====================

    def _predict_oos(self, store: SequenceStore, all_idx: list,
                     all_dates: np.ndarray, w, batch_size: int
                     ) -> tuple[pd.DataFrame | None, float, float]:
        """对折 OOS 窗口 [ws, we) 做样本外预测(要求模型已加载好权重)。

        Returns:
            (OOS 预测 DataFrame | None, oos_ic, oos_rmse)
        """
        ws = np.datetime64(w.ws)
        we = np.datetime64(w.we)
        oos_mask = (all_dates >= ws) & (all_dates < we)
        oos_idx = [all_idx[i] for i in np.nonzero(oos_mask)[0]]
        oos_dates = all_dates[oos_mask]

        if not oos_idx:
            return None, np.nan, np.nan

        pin = self.device.type == "cuda"
        oos_loader, oos_labels = make_loaders(
            store, oos_idx, oos_dates, batch_size=batch_size,
            shuffle=False, pin_memory=pin)
        oos_pred = self.predict_loader(oos_loader)
        ic_res = compute_rank_ic(oos_pred, oos_labels,
                                 (oos_dates.astype("datetime64[D]")
                                  .astype(np.int64)))
        oos_ic = ic_res["mean_ic"]
        oos_rmse = float(np.sqrt(np.mean((oos_pred - oos_labels) ** 2)))
        symbols = [store.symbols[si] for si, _ in oos_idx]
        oos_df = pd.DataFrame({
            "date": oos_dates, "symbol": symbols,
            "prediction": oos_pred, "forward_return": oos_labels,
        })
        return oos_df, oos_ic, oos_rmse

    def _run_fold(self, store: SequenceStore, all_idx: list,
                  all_dates: np.ndarray, w, batch_size: int,
                  epochs: int, save_dir: str
                  ) -> tuple[dict, pd.DataFrame | None]:
        """执行单折:切分 → 训练 → OOS 预测。

        Returns:
            (fold 指标 dict, OOS 预测 DataFrame | None)
        """
        wf_cfg = self.cfg["model"]["walk_forward"]

        ws = np.datetime64(w.ws)
        train_mask = all_dates < ws
        val_start = w.ws - pd.DateOffset(months=int(wf_cfg.get("val_months", 3)))
        valid_mask = train_mask & (all_dates >= np.datetime64(val_start))

        train_idx = [all_idx[i] for i in np.nonzero(train_mask)[0]]
        valid_idx = [all_idx[i] for i in np.nonzero(valid_mask)[0]]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise RuntimeError(f"Fold {w.fold}: 训练/验证样本不足")

        pin = self.device.type == "cuda"
        train_loader, _ = make_loaders(
            store, train_idx, all_dates[train_mask], batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=pin, seed=self.seed)
        valid_loader, _ = make_loaders(
            store, valid_idx, all_dates[valid_mask], batch_size=batch_size,
            shuffle=False, pin_memory=pin)

        fold_dir = os.path.join(save_dir, f"fold_{w.fold}")
        result = self.fit(train_loader, valid_loader,
                          epochs=epochs, fold=w.fold,
                          save_dir=fold_dir)

        # --- OOS 预测(样本外) ---
        oos_df, oos_ic, oos_rmse = self._predict_oos(
            store, all_idx, all_dates, w, batch_size)

        train_dates = all_dates[train_mask]
        oos_mask = (all_dates >= ws) & (all_dates < np.datetime64(w.we))
        fold_metrics = {
            "fold": w.fold,
            "train_start": pd.Timestamp(train_dates.min()),
            "train_end": pd.Timestamp(train_dates.max()),
            "oos_start": w.ws,
            "oos_end": w.we - pd.Timedelta(days=1),
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_oos": int(oos_mask.sum()),
            "best_rank_ic": result["best_rank_ic"],
            "best_loss": result["best_loss"],
            "best_epoch": result["best_epoch"],
            "oos_rank_ic": oos_ic,
            "oos_rmse": oos_rmse,
        }
        return fold_metrics, oos_df

    def walk_forward_train(self, store: SequenceStore, cfg: dict,
                           save_dir: str = "models/saved",
                           epochs: int | None = None,
                           batch_size: int | None = None,
                           resume: bool = True
                           ) -> tuple[list[dict], pd.DataFrame]:
        """Walk-Forward 滚动训练(与 quantlab 同构)。

        每折:
            train = 日期 < 折 OOS 窗口起点的样本(滚动扩展)
            valid = 训练尾部 val_months 个月的样本(早停,同样早于 OOS)
            fit → 用 best checkpoint 对 [ws, we) 窗口做 OOS 预测

        断点续训(resume=True):
            - 该折 OOS 窗口已有预测(predictions.parquet 覆盖) → 跳过
            - 该折 checkpoint 已存在但预测缺失 → 复用权重只做 OOS 预测
            - 每折完成后增量落盘 predictions.parquet(中断不丢结果)

        Args:
            store: SequenceStore
            cfg: 完整 config dict
            save_dir: checkpoint 输出目录(每折一个子目录 fold_N/)
            epochs/batch_size: 覆盖 config(CLI 参数)
            resume: 是否启用断点续训

        Returns:
            (folds 指标列表, OOS 预测 DataFrame)
            OOS DataFrame 列: date, symbol, prediction, forward_return
        """
        self.factor_names = store.factor_names
        tr = cfg["model"]["training"]
        wf_cfg = cfg["model"]["walk_forward"]
        epochs = epochs or int(tr["epochs"])
        batch_size = batch_size or int(tr["batch_size"])

        windows = walk_forward_windows(store.global_dates,
                                       int(wf_cfg["min_train_months"]),
                                       int(wf_cfg["retrain_months"]))
        all_idx, all_dates = store.sample_index()
        logger.info(f"全量样本: {len(all_idx):,} | "
                    f"Walk-Forward {len(windows)} 折, batch={batch_size}")

        # --- 断点续训:加载已有 OOS 预测 ---
        pred_path = cfg["data"]["prediction_cache"]
        existing = None
        if resume and os.path.exists(pred_path):
            existing = pd.read_parquet(pred_path)
            existing["date"] = pd.to_datetime(existing["date"])
            existing["symbol"] = existing["symbol"].astype(str)
            logger.info(f"断点续训: 已存在 {len(existing):,} 条 OOS 预测")

        folds = []
        oos_parts = []

        for w in windows:
            ws = np.datetime64(w.ws)
            we = np.datetime64(w.we)
            logger.info(f"===== Fold {w.fold}/{len(windows)}: "
                        f"OOS [{w.ws.date()}, {w.we.date()}) =====")

            # 该折预测已存在 → 直接复用
            if resume and existing is not None and len(existing) > 0:
                covered = ((existing["date"] >= ws)
                           & (existing["date"] < we)).any()
                if covered:
                    fold_part = existing[(existing["date"] >= ws)
                                         & (existing["date"] < we)].copy()
                    di = fold_part["date"].values.astype(
                        "datetime64[D]").astype(np.int64)
                    ic = compute_rank_ic(fold_part["prediction"].to_numpy(),
                                         fold_part["forward_return"]
                                         .to_numpy(), di)
                    train_end = pd.Timestamp(all_dates[all_dates < ws].max())
                    folds.append({
                        "fold": w.fold,
                        "train_start": pd.Timestamp(all_dates[0]),
                        "train_end": train_end,
                        "oos_start": w.ws,
                        "oos_end": w.we - pd.Timedelta(days=1),
                        "n_train": 0, "n_valid": 0,
                        "n_oos": len(fold_part),
                        "best_rank_ic": np.nan, "best_loss": np.nan,
                        "best_epoch": 0,
                        "oos_rank_ic": ic["mean_ic"], "oos_rmse": np.nan,
                    })
                    oos_parts.append(fold_part)
                    logger.info(f"Fold {w.fold}: 已有 OOS 预测,跳过"
                                f" (OOS_IC={ic['mean_ic']:.4f})")
                    continue

            # checkpoint 已存在但预测缺失 → 复用权重只做 OOS 预测
            fold_dir = os.path.join(save_dir, f"fold_{w.fold}")
            ckpt = os.path.join(fold_dir, "best.pt")
            if resume and os.path.exists(ckpt):
                self.model, meta = self.load_checkpoint(ckpt, self.device)
                self._reset_optimizer()  # 新模型 → 必须重建优化器
                oos_df, oos_ic, oos_rmse = self._predict_oos(
                    store, all_idx, all_dates, w, batch_size)
                train_end = pd.Timestamp(all_dates[all_dates < ws].max())
                fold_metrics = {
                    "fold": w.fold,
                    "train_start": pd.Timestamp(all_dates[0]),
                    "train_end": train_end,
                    "oos_start": w.ws,
                    "oos_end": w.we - pd.Timedelta(days=1),
                    "n_train": 0, "n_valid": 0,
                    "n_oos": int(((all_dates >= ws) & (all_dates < we))
                                 .sum()),
                    "best_rank_ic": meta["metrics"].get("mean_ic", np.nan),
                    "best_loss": meta["metrics"].get("loss", np.nan),
                    "best_epoch": meta["epoch"],
                    "oos_rank_ic": oos_ic, "oos_rmse": oos_rmse,
                }
            else:
                try:
                    fold_metrics, oos_df = self._run_fold(
                        store, all_idx, all_dates, w, batch_size, epochs,
                        save_dir)
                except RuntimeError as e:
                    # 跨版本兼容的 OOM 捕获(torch.cuda.OutOfMemoryError /
                    # torch.OutOfMemoryError 均为 RuntimeError 子类)
                    if ("out of memory" in str(e).lower()
                            and batch_size >= 128
                            and self.device.type == "cuda"):
                        batch_size //= 2
                        logger.warning(f"OOM → batch 减半为 {batch_size} 重试")
                        fold_metrics, oos_df = self._run_fold(
                            store, all_idx, all_dates, w, batch_size, epochs,
                            save_dir)
                    else:
                        raise

            folds.append(fold_metrics)
            if oos_df is not None:
                oos_parts.append(oos_df)
                # 增量落盘:每折完成即持久化,中断不丢结果
                save_df = (pd.concat([existing, oos_df], ignore_index=True)
                           if existing is not None and len(existing) > 0
                           else oos_df)
                save_df = save_df.sort_values("date") \
                    .drop_duplicates(subset=["date", "symbol"],
                                     keep="last") \
                    .reset_index(drop=True)
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                save_df.to_parquet(pred_path)
                existing = save_df

            logger.info(
                f"Fold {w.fold} 完成: train_end={fold_metrics['train_end']:%Y-%m-%d}"
                f" | best_IC={fold_metrics['best_rank_ic']:.4f}"
                f" | OOS_IC={fold_metrics['oos_rank_ic']:.4f}"
                f" | OOS_RMSE={fold_metrics['oos_rmse']:.5f}")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if not oos_parts:
            logger.warning("没有任何 OOS 预测(数据不足?)")
            oos_df = pd.DataFrame(columns=["date", "symbol",
                                           "prediction", "forward_return"])
        else:
            oos_df = pd.concat(oos_parts, ignore_index=True)
            oos_df = oos_df.sort_values("date") \
                .drop_duplicates(subset=["date", "symbol"],
                                 keep="last") \
                .reset_index(drop=True)

        mean_ic = np.mean([f["oos_rank_ic"] for f in folds
                           if np.isfinite(f["oos_rank_ic"])])
        logger.info(f"Walk-Forward 全部完成: 平均 OOS RankIC = {mean_ic:.4f}"
                    f" (>0.02 可用, >0.05 优秀, <0 无效)")
        return folds, oos_df
