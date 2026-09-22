#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QuantLab2 — 基于 Transformer 的 A 股深度学习选股框架 CLI 入口。

用法:
    python main.py train      Walk-Forward 滚动训练(约 8 折)
    python main.py backtest   用 OOS 预测回测(必须先 train)
    python main.py predict    用最新 checkpoint 生成最新交易日调仓信号
    python main.py pipeline   train + backtest 串联
    python main.py smoke      快速冒烟测试(<3 分钟,验证全管道)
"""

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.device import get_device, seed_everything


def load_config(path: str = "config.yaml") -> dict:
    """加载 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==================== 公共:数据加载 ====================

def build_store(config: dict, panel: pd.DataFrame,
                columns: list[str] | None = None):
    """从因子面板 + 日线构建 SequenceStore。

    日线默认只要 日期/收盘 两列(算前向收益标签够用),parquet 是列存,
    裁剪列能砍掉绝大部分解析量 —— 999 个文件逐个读时这个差别是分钟级。
    实盘引擎还要用 开盘/最高/最低/成交量/涨跌幅 做撮合与盯市,那种调用方
    自己传 columns(SequenceStore 只认 日期/收盘,多给不改变任何结果)。
    """
    from data.loader import load_daily_dict, DATE_COL, CLOSE_COL
    from models.sequence_data import SequenceStore

    symbols = sorted(panel["symbol"].unique())
    cols = list(columns) if columns is not None else [DATE_COL, CLOSE_COL]
    daily = load_daily_dict(config["data"]["daily_dir"], symbols,
                            columns=cols)
    store = SequenceStore(panel, daily, config)
    return store, daily


# ==================== train ====================

def cmd_train(args):
    """Walk-Forward 滚动训练,落盘全样本外预测。"""
    config = load_config(args.config)
    setup_logger(config["logging"]["level"], config["logging"]["file"])
    seed_everything(config["model"]["training"]["seed"])
    device = get_device()

    from data.loader import load_factor_panel
    from models.transformer import build_model, count_parameters
    from models.trainer import TransformerTrainer

    panel = load_factor_panel(config["data"]["factor_panel"])
    store, _ = build_store(config, panel)

    model = build_model(config["model"], n_features=store.n_features)
    logger.info(f"模型参数量: {count_parameters(model):,}")

    trainer = TransformerTrainer(model, config, device,
                                 seed=config["model"]["training"]["seed"])
    folds, oos_df = trainer.walk_forward_train(
        store, config,
        save_dir=config["model"]["save_dir"],
        epochs=args.epochs, batch_size=args.batch_size)

    # 折指标汇总
    logger.info("=" * 70)
    logger.info("  Walk-Forward 结果汇总")
    logger.info("=" * 70)
    for f in folds:
        logger.info(
            f"  Fold {f['fold']}: train [{f['train_start']:%Y-%m} → "
            f"{f['train_end']:%Y-%m}] | OOS [{f['oos_start']:%Y-%m} → "
            f"{f['oos_end']:%Y-%m}] | best_IC={f['best_rank_ic']:.4f}"
            f" | OOS_IC={f['oos_rank_ic']:.4f} | OOS_RMSE={f['oos_rmse']:.5f}")

    ics = [f["oos_rank_ic"] for f in folds if np.isfinite(f["oos_rank_ic"])]
    if ics:
        mean_ic = np.mean(ics)
        logger.info(f"平均 OOS RankIC = {mean_ic:.4f} "
                    f"(>0.02 可用, >0.05 优秀, <0 无效)")
        logger.info(f"OOS RankIC 稳定性: 正 IC 折数 "
                    f"{sum(1 for x in ics if x > 0)}/{len(ics)}")

    # 落盘 OOS 预测(回测唯一输入,杜绝全样本回退)
    out = config["data"]["prediction_cache"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    oos_df.to_parquet(out)
    logger.info(f"OOS 预测已保存: {out} ({len(oos_df):,} 条)")
    logger.info(f"下一步: python main.py backtest")


# ==================== backtest ====================

def cmd_backtest(args):
    """用 OOS 预测回测(只消费样本外预测,杜绝未来函数)。"""
    config = load_config(args.config)
    setup_logger(config["logging"]["level"], config["logging"]["file"])

    pred_path = config["data"]["prediction_cache"]
    if not os.path.exists(pred_path):
        logger.error(f"未找到 OOS 预测文件: {pred_path}")
        logger.error("请先运行 python main.py train 完成 Walk-Forward 训练")
        sys.exit(1)

    preds = pd.read_parquet(pred_path)
    preds["date"] = pd.to_datetime(preds["date"])
    logger.info(f"加载 OOS 预测: {len(preds):,} 条 "
                f"({preds['date'].min().date()} → {preds['date'].max().date()})")

    # OOS Rank IC 汇总(与 LightGBM 对比的核心指标)
    from models.trainer import compute_rank_ic
    date_ints = preds["date"].values.astype("datetime64[D]").astype(np.int64)
    ic = compute_rank_ic(preds["prediction"].to_numpy(),
                         preds["forward_return"].to_numpy(), date_ints)
    logger.info(f"OOS RankIC = {ic['mean_ic']:.4f} | ICIR = {ic['icir']:.2f}"
                f" | 有效截面 {ic['n_days']} 天")

    # 日线数据:先加载再选股 —— 可交易掩码要知道信号日有没有成交
    from data.loader import load_daily_dict, EXEC_DAILY_COLS
    from utils.market_rules import build_tradable_mask
    symbols = sorted(preds["symbol"].unique())
    daily = load_daily_dict(config["data"]["daily_dir"], symbols,
                            columns=EXEC_DAILY_COLS)

    # 生成信号(信号日已停牌/无成交的股票不占 Top-K 名额)
    from models.predictor import signals_from_predictions, scores_from_predictions
    from utils.position_policy import policy_from_config
    predictions = preds.set_index(["date", "symbol"])["prediction"]
    signal_dates = predictions.index.get_level_values("date").unique()
    tradable = build_tradable_mask(daily, signal_dates)
    policy = policy_from_config(config)
    if policy is None:
        signals = signals_from_predictions(
            predictions,
            top_k=config["backtest"]["max_positions"],
            position_sizing=config["backtest"]["position_sizing"],
            tradable=tradable)
    else:
        # 策略要的是全截面分数;评估频率就是它的补/减仓节奏
        if config["backtest"]["rebalance_frequency"] == "monthly":
            logger.warning(
                "position_policy 已启用但 backtest.rebalance_frequency=monthly:"
                "月末才评估会把月内的补仓/减仓档位全丢掉,单票 16% 上限也只能"
                "月末才压得住。建议改成 daily 或 weekly 再对比结果")
        signals = scores_from_predictions(predictions, tradable=tradable)

    # 基准指数(akshare 不可用时回退到股票池等权基准)
    from data.loader import load_benchmark, build_universe_benchmark
    bm = load_benchmark(config["backtest"]["benchmark"],
                        str(preds["date"].min().date()),
                        str(preds["date"].max().date()),
                        config["data"]["benchmark_cache"])
    if bm is None:
        logger.warning("基准指数不可用,改用股票池等权基准"
                       "(满仓持有全部股票池的收益)")
        bm = build_universe_benchmark(daily)

    # 回测
    from backtest.engine import BacktestEngine
    from backtest.cost import TransactionCostModel
    mkt = config["market"]
    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital"],
        rebalance_frequency=config["backtest"]["rebalance_frequency"],
        max_positions=config["backtest"]["max_positions"],
        cost_model=TransactionCostModel(
            mkt["commission_rate"], mkt["min_commission"],
            mkt["stamp_tax_rate"], mkt["slippage_rate"]),
        lot_size=int(mkt.get("lot_size", 100)),
        policy=policy)
    result = engine.run(daily, signals, bm)

    if not result:
        logger.error("回测失败: 无净值数据")
        sys.exit(1)

    # 报告
    from backtest.reporter import ReportGenerator
    reporter = ReportGenerator("reports")
    print(reporter.console_report(result["metrics"]))
    if args.html:
        reporter.html_report(
            result["equity_curve"], result.get("benchmark_curve"),
            result["daily_returns"], result["trades"], result["metrics"])

    # 保存净值曲线 CSV(方便与 quantlab 对比)
    result["equity_curve"].to_csv("reports/equity_curve.csv",
                                  encoding="utf-8-sig")
    logger.info("净值曲线已保存: reports/equity_curve.csv")


# ==================== predict ====================

def latest_checkpoint(save_dir: str) -> str | None:
    """找最新一折的 best.pt;找不到则任意 best.pt。"""
    root = Path(save_dir)
    if not root.exists():
        return None
    fold_dirs = sorted([d for d in root.iterdir()
                        if d.is_dir() and d.name.startswith("fold_")],
                       key=lambda d: int(d.name.split("_")[1]))
    for d in reversed(fold_dirs):
        p = d / "best.pt"
        if p.exists():
            return str(p)
    for p in sorted(root.rglob("best.pt")):
        return str(p)
    return None


def cmd_predict(args):
    """用最新 checkpoint 生成目标交易日调仓信号。"""
    config = load_config(args.config)
    setup_logger(config["logging"]["level"], config["logging"]["file"])
    device = get_device()

    from data.loader import load_factor_panel, EXEC_DAILY_COLS
    from utils.market_rules import build_tradable_mask
    from models.trainer import TransformerTrainer
    from models.predictor import TransformerPredictor

    checkpoint = args.checkpoint or latest_checkpoint(config["model"]["save_dir"])
    if checkpoint is None:
        logger.error(f"未找到 checkpoint,请先运行 python main.py train")
        sys.exit(1)

    model, meta = TransformerTrainer.load_checkpoint(checkpoint, device)
    panel = load_factor_panel(config["data"]["factor_panel"])
    store, daily = build_store(config, panel, columns=EXEC_DAILY_COLS)

    predictor = TransformerPredictor(model, store, config, device)
    asof = args.asof or datetime.now().strftime("%Y-%m-%d")
    predictions = predictor.predict_asof(asof)
    # 信号日已停牌/无成交的股票不占 Top-K 名额
    tradable = build_tradable_mask(
        daily, predictions.index.get_level_values("date").unique())
    from models.predictor import scores_from_predictions
    from utils.position_policy import policy_from_config
    policy = policy_from_config(config)
    if policy is None:
        signals = predictor.generate_signals(predictions=predictions,
                                             tradable=tradable)
        holding = signals[signals["weight"] > 0].reset_index()
    else:
        # 策略模式下"该买谁"不是一个 Top-K 名单,而是分数 + 现有持仓 + 状态
        # 的函数;predict 只负责把过建仓线的分数摊开给人看/校准,
        # 真正的建仓·补仓·减仓指令由 python main.py live 出。
        scores = scores_from_predictions(predictions, tradable=tradable)
        holding = scores[scores["score"] >= policy.buy_score].reset_index()
        logger.info(f"策略模式: 建仓线 {policy.buy_score:.4f} 以上 "
                    f"{len(holding)} 只(全截面 {len(scores)} 只);"
                    f"实际指令请跑 python main.py live")

    # 输出持仓信号 CSV
    out_dir = config["predict"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"signals_{datetime.now().strftime('%Y%m%d')}.csv")
    holding.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"持仓信号已输出: {out} ({len(holding)} 只)")
    cols = ["symbol", "score", "rank"]
    print(holding[cols + ([] if policy else ["weight"])].head(60).to_string())


# ==================== live ====================

def cmd_live_status(args):
    """查看模拟盘持仓状态(现金/持仓明细/盈亏)。"""
    import json

    config = load_config(args.config)
    setup_logger(config["logging"]["level"], config["logging"]["file"])

    state_path = os.path.join(config["live"].get("state_dir", "live/state"),
                              "simulate_state.json")
    if not os.path.exists(state_path):
        logger.error("还没有模拟盘持仓,请先运行 "
                     "python main.py live --broker simulate")
        sys.exit(1)

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    cash = float(state.get("cash", 0))
    initial = float(state.get("initial_cash", 0))
    positions = state.get("positions", {})
    trade_date = state.get("trade_date")

    # 用日线最后收盘价补最新市值(实时行情盘中才有)
    prices = {}
    if positions:
        try:
            from data.loader import load_daily_dict
            daily = load_daily_dict(config["data"]["daily_dir"],
                                    list(positions.keys()))
            for sym, df in daily.items():
                if len(df) > 0:
                    prices[sym] = float(df["收盘"].iloc[-1])
        except Exception as e:
            logger.warning(f"日线加载失败,盈亏按成本价计算: {e}")

    # 表格输出
    mv = 0.0
    pnl = 0.0
    lines = []
    lines.append("=" * 78)
    lines.append(f"  模拟盘持仓状态(截至 {trade_date or '未知'})")
    lines.append("=" * 78)
    lines.append(f"  {'symbol':<8s} {'持股':>7s} {'可卖':>7s} {'成本价':>8s} "
                 f"{'现价':>8s} {'市值':>11s} {'盈亏%':>8s}")
    lines.append("-" * 78)
    for sym in sorted(positions.keys()):
        p = positions[sym]
        shares = int(p.get("shares", 0))
        avail = int(p.get("available_shares", shares))
        cost = float(p.get("avg_cost", 0))
        price = prices.get(sym, cost)
        value = shares * price
        pct = (price / cost - 1) * 100 if cost > 0 else 0.0
        mv += value
        pnl += (price - cost) * shares
        lines.append(f"  {sym:<8s} {shares:>7,d} {avail:>7,d} {cost:>8.2f} "
                     f"{price:>8.2f} {value:>11,.0f} {pct:>+7.2f}%")
    lines.append("-" * 78)
    total = cash + mv
    lines.append(f"  持仓 {len(positions)} 只 | 市值 {mv:,.0f} | "
                 f"现金 {cash:,.0f} | 总资产 {total:,.0f}")
    if initial > 0:
        lines.append(f"  累计收益 {total/initial-1:+.2%} | "
                     f"持仓浮盈 {pnl:+,.0f} 元")
    lines.append("=" * 78)
    print("\n".join(lines))


def cmd_live(args):
    """实时交易:盘中轮询 + 定时调仓,支持 simulate/qmt/none 后端。"""
    config = load_config(args.config)
    setup_logger(config["logging"]["level"], config["logging"]["file"])

    from live.broker import create_broker
    from live.engine import LiveEngine

    # qmt 实盘必须显式 --confirm 才允许真实下单
    broker_kind = args.broker
    if broker_kind == "qmt" and not args.confirm:
        from loguru import logger as lg
        lg.warning("QMT 实盘模式未加 --confirm:仅生成指令文件,"
                   "不向柜台提交任何订单(安全模式)")

    try:
        broker = create_broker(broker_kind, config)
    except RuntimeError as e:
        logger.error(f"券商初始化失败: {e}")
        sys.exit(1)

    engine = LiveEngine(config, broker, confirm=args.confirm,
                        once=args.once, asof=args.asof)
    engine.run()


# ==================== pipeline ====================

def cmd_pipeline(args):
    """train → backtest 串联。"""
    cmd_train(args)
    cmd_backtest(args)


# ==================== smoke ====================

def cmd_smoke(args):
    """快速冒烟测试:子集数据 + 小模型,验证全管道(<3 分钟)。

    覆盖:
        1. 数据加载与 SequenceStore(防未来函数泄漏断言)
        2. Walk-Forward 窗口划分(全量日期,≥7 折)
        3. 小模型 2 epoch 训练(loss 下降)
        4. 过拟合测试(64 样本 loss → <1e-3)
        5. checkpoint 保存/加载往返(输出一致)
        6. 信号生成 + 回测出净值
    """
    config = load_config(args.config)
    setup_logger(config["logging"]["level"], config["logging"]["file"])
    seed_everything(42)
    device = get_device()

    from data.loader import load_factor_panel
    from models.sequence_data import SequenceStore, make_loader, \
        walk_forward_windows
    from models.transformer import build_model, count_parameters
    from models.trainer import TransformerTrainer, compute_rank_ic

    # ===== 1. 数据加载 =====
    logger.info("[smoke 1/6] 数据加载与 SequenceStore ...")
    panel_full = load_factor_panel(config["data"]["factor_panel"])
    assert len(panel_full) > 1_000_000, "因子面板行数异常"
    assert panel_full["symbol"].nunique() >= 900, "股票数异常"
    n_factors = len([c for c in panel_full.columns
                     if c not in ["date", "symbol"]])
    assert n_factors == 25, f"因子数异常: {n_factors}"

    # Walk-Forward 窗口(用全量日期,不建全量 store)
    windows = walk_forward_windows(
        panel_full["date"].to_numpy(), 24, 6)
    assert len(windows) >= 7, f"Walk-Forward 折数不足: {len(windows)}"
    logger.info(f"Walk-Forward 窗口: {len(windows)} 折")

    # 子集:100 只股票,2024-06 起
    symbols = sorted(panel_full["symbol"].unique())[:100]
    panel = panel_full[panel_full["symbol"].isin(symbols)]
    panel = panel[panel["date"] >= "2024-06-01"]
    del panel_full

    from data.loader import load_daily_dict
    daily = load_daily_dict(config["data"]["daily_dir"], symbols)
    store = SequenceStore(panel, daily, config)

    # 泄漏断言
    samples = store.sample_index()
    dates = samples.dates
    assert len(samples) > 5000, f"样本太少: {len(samples)}"
    assert np.isfinite(samples.labels(store)).all(), "训练样本标签含 NaN"
    rng = np.random.default_rng(42)
    for i in rng.choice(len(samples), 50, replace=False):
        si, t = samples[i]
        sym = store.symbols[si]
        win_dates = store.dates_by_symbol[sym][t - store.seq_len + 1: t + 1]
        assert win_dates[-1] == store.dates_by_symbol[sym][t]
        assert np.all(np.diff(win_dates.astype("int64")) > 0), \
            "窗口日期非严格递增"
        assert win_dates.max() <= dates[i], "泄漏:窗口含未来日期"
        assert np.isfinite(store.get_window(si, t)).all(), "窗口含 NaN"
        assert np.isfinite(store.get_label(si, t)), "标签为 NaN"
    logger.info("泄漏断言通过(50 个随机样本)")

    # ===== 2. 小模型训练 =====
    logger.info("[smoke 2/6] 小模型训练 2 epoch ...")
    smoke_cfg = copy.deepcopy(config)
    smoke_cfg["model"]["seq_len"] = 10
    smoke_cfg["model"]["arch"].update({"d_model": 32, "n_heads": 4,
                                       "n_layers": 2, "dim_ff": 64})
    smoke_cfg["model"]["training"].update({"batch_size": 256, "epochs": 2})
    store.seq_len = 10

    samples = store.sample_index()
    dates = samples.dates
    uniq = np.unique(dates)
    split_date = uniq[int(len(uniq) * 0.8)]
    train_mask = dates <= split_date
    train = samples.select(train_mask)
    valid = samples.select(~train_mask)

    train_loader = make_loader(store, train, batch_size=256, shuffle=True,
                               seed=42)
    valid_loader = make_loader(store, valid, batch_size=256, shuffle=False)

    model = build_model(smoke_cfg["model"], n_features=store.n_features)
    logger.info(f"smoke 模型参数量: {count_parameters(model):,}")
    trainer = TransformerTrainer(model, smoke_cfg, device, seed=42)
    res = trainer.fit(train_loader, valid_loader, epochs=2, fold=0,
                      save_dir="models/saved/smoke")
    l0, l1 = res["history"][0]["train_loss"], res["history"][-1]["train_loss"]
    assert np.isfinite(l1), "训练 loss 非有限"
    assert l1 < l0, f"训练 loss 未下降: {l0:.4f} → {l1:.4f}"
    logger.info(f"训练 loss: {l0:.4f} → {l1:.4f} ✓")

    # ===== 3. 过拟合测试 =====
    logger.info("[smoke 3/6] 过拟合测试(64 样本) ...")
    pos = rng.choice(len(samples), 64, replace=False)
    tiny = samples.select(pos)
    tiny_loader = make_loader(store, tiny, batch_size=64, shuffle=True,
                              seed=42)

    ov_cfg = copy.deepcopy(smoke_cfg)
    # 64 样本的过拟合测试:关 dropout、恒定较大 lr(cosine 衰减会拖慢
    # 记忆速度),否则 50 步内学不动
    ov_cfg["model"]["arch"]["dropout"] = 0.0
    ov_cfg["model"]["training"].update({
        "early_stop_patience": 9999,  # 过拟合测试不早停
        "lr": 0.03, "warmup_steps": 0, "lr_min": 0.03})
    ov_model = build_model(ov_cfg["model"], n_features=store.n_features)
    ov_trainer = TransformerTrainer(ov_model, ov_cfg, device, seed=42)
    ov_res = ov_trainer.fit(tiny_loader, tiny_loader, epochs=150, fold=0,
                            save_dir="models/saved/smoke_overfit")
    ov_loss = ov_res["history"][-1]["train_loss"]
    assert ov_loss < 1e-3, f"过拟合测试失败: loss={ov_loss:.6f}"
    logger.info(f"过拟合 loss = {ov_loss:.6f} (<1e-3) ✓")

    # ===== 4. checkpoint 往返 =====
    logger.info("[smoke 4/6] checkpoint 保存/加载往返 ...")
    import torch
    from models.trainer import TransformerTrainer as TT
    ckpt_path = ov_res["path"]
    m2, meta2 = TT.load_checkpoint(ckpt_path, device)
    x = torch.from_numpy(store.get_window(*tiny[0])).unsqueeze(0) \
        .to(device)
    with torch.no_grad():
        y1 = ov_trainer.model(x)
        y2 = m2(x)
    assert torch.allclose(y1, y2, atol=1e-5), "checkpoint 往返输出不一致"
    logger.info("checkpoint 往返输出一致 ✓")

    # ===== 5. 信号生成 =====
    logger.info("[smoke 5/6] 信号生成与回测 ...")
    from models.predictor import signals_from_predictions
    # 用训练好的 2-epoch 小模型对验证集推理(不取 overfit 模型,避免失真)
    valid_pred = trainer.predict_loader(valid_loader)
    pred_series = pd.Series(
        valid_pred,
        index=pd.MultiIndex.from_arrays(
            [valid.dates, store.symbols_of(valid)],
            names=["date", "symbol"]),
        name="prediction")
    signals = signals_from_predictions(pred_series, top_k=20,
                                       position_sizing="equal_weight")
    assert (signals["weight"] > 0).sum() > 0, "未生成任何持仓信号"
    assert list(signals.columns) == ["score", "rank", "weight"], \
        "信号 schema 错误"
    logger.info(f"信号生成: {(signals['weight'] > 0).sum()} 条持仓 ✓")

    # ===== 6. 回测 =====
    from backtest.engine import BacktestEngine
    from backtest.cost import TransactionCostModel
    mkt = config["market"]
    engine = BacktestEngine(
        initial_capital=1_000_000, max_positions=20,
        cost_model=TransactionCostModel(
            mkt["commission_rate"], mkt["min_commission"],
            mkt["stamp_tax_rate"], mkt["slippage_rate"]))
    result = engine.run(daily, signals, None)
    assert result and not result["equity_curve"].empty, "回测无净值"
    logger.info(f"冒烟回测: 累计收益={result['metrics']['cumulative_return']:.2%}"
                f" | 夏普={result['metrics']['sharpe_ratio']:.2f} ✓")

    # 成本模型断言
    cost = TransactionCostModel(0.0003, 5.0, 0.0005, 0.001)
    assert abs(cost.total_cost(10_000, "buy") - (5.0 + 10.0)) < 1e-6, \
        "佣金最低 5 元断言失败"
    assert abs(cost.total_cost(100_000, "sell") -
               (30.0 + 50.0 + 100.0)) < 1e-6, "卖出成本断言失败"

    logger.info("=" * 60)
    logger.info("  冒烟测试全部通过 ✓ (数据/训练/过拟合/checkpoint/信号/回测)")
    logger.info("=" * 60)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLab2 — Transformer A股深度学习选股框架")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Walk-Forward 滚动训练")
    p_train.add_argument("--config", default="config.yaml")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="覆盖 config 的 epochs")
    p_train.add_argument("--batch-size", type=int, default=None,
                         help="覆盖 config 的 batch_size")

    p_backtest = sub.add_parser("backtest", help="用 OOS 预测回测")
    p_backtest.add_argument("--config", default="config.yaml")
    p_backtest.add_argument("--html", action="store_true",
                            help="生成 HTML 报告")

    p_predict = sub.add_parser("predict", help="生成调仓信号")
    p_predict.add_argument("--config", default="config.yaml")
    p_predict.add_argument("--asof", default=None,
                           help="目标交易日 YYYY-MM-DD(默认今天)")
    p_predict.add_argument("--checkpoint", default=None,
                           help="checkpoint 路径(默认最新一折 best.pt)")

    p_pipeline = sub.add_parser("pipeline", help="train + backtest 串联")
    p_pipeline.add_argument("--config", default="config.yaml")
    p_pipeline.add_argument("--epochs", type=int, default=None)
    p_pipeline.add_argument("--batch-size", type=int, default=None)
    p_pipeline.add_argument("--html", action="store_true")

    p_smoke = sub.add_parser("smoke", help="快速冒烟测试(<3 分钟)")
    p_smoke.add_argument("--config", default="config.yaml")

    p_live = sub.add_parser("live", help="实时交易(盘中轮询/单次执行)")
    p_live.add_argument("--config", default="config.yaml")
    p_live.add_argument("--broker", default=None,
                        help="券商后端: simulate / qmt / none"
                             "(默认 config live.broker)")
    p_live.add_argument("--confirm", action="store_true",
                        help="QMT 实盘真实下单(默认仅出指令文件)")
    p_live.add_argument("--once", action="store_true",
                        help="单次执行后退出(测试/计划任务用)")
    p_live.add_argument("--asof", default=None,
                        help="信号基准日期 YYYY-MM-DD(默认今天)")

    p_live_status = sub.add_parser("live-status", help="查看模拟盘持仓状态")
    p_live_status.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "live-status":
        cmd_live_status(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "smoke":
        cmd_smoke(args)


if __name__ == "__main__":
    main()

