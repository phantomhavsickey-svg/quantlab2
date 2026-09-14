#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日定时任务入口 — Windows 计划任务每日 16:00 调用
(晚于 quantlab 每日批处理 15:30,确保因子面板已更新)。

流程:
    1. 非交易日 → 直接退出
    2. 月末交易日(调仓日)→ 完整 LiveEngine once 流程
       (模型信号 → 指令 → 风控 → 模拟盘撮合 → 盯市落盘)
    3. 普通交易日 → 轻量盯市:
       只做 T+1 解锁 + 按最新日线更新市价 + 每日快照落盘,
       不加载模型(秒级完成)

参数:
    --config 配置文件路径(默认 config.yaml)
    --broker 券商后端(默认 config live.broker)
    --confirm qmt 实盘真实下单
    --asof    覆盖信号基准日(测试用)
    --dry-run 只打印将执行的动作,不实际执行
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from utils.logger import setup_logger
from utils.calendar import is_trading_day


def is_month_end_trading_day(today: pd.Timestamp) -> bool:
    """今天是否为本月最后一个交易日。"""
    from utils.calendar import get_trading_calendar
    month_end = (today + pd.offsets.MonthEnd(0)).date()
    try:
        days = get_trading_calendar(str(today.date()), str(month_end))
    except Exception as e:
        logger.warning(f"交易日历获取失败({e}),按工作日判断")
        days = [d for d in pd.bdate_range(today.date(), month_end)]
    return len(days) > 0 and today.date() == days[-1]


def mark_to_market_only(config: dict, today):
    """轻量盯市(非调仓日):T+1 解锁 + 更新市价 + 快照落盘。

    不加载模型/因子面板,只依赖模拟盘状态文件与日线缓存。
    """
    from live.broker import create_broker
    from live.journal import TradeJournal

    broker = create_broker("simulate", config)
    broker.connect()

    positions = broker.positions_dict()
    if not positions:
        logger.info("无持仓,无需盯市")
        broker.disconnect()
        return

    # 用日线最后收盘构造当日 bar 快照
    from data.loader import load_daily_dict
    daily = load_daily_dict(config["data"]["daily_dir"],
                            sorted(positions.keys()))
    snapshot = {}
    for sym, df in daily.items():
        if len(df) == 0:
            continue
        row = df.iloc[-1]
        snapshot[sym] = {
            "open": float(row["开盘"]), "high": float(row["最高"]),
            "low": float(row["最低"]), "close": float(row["收盘"]),
            "volume": float(row.get("成交量", 0) or 0),
            "at_limit_up": False, "at_limit_down": False,
        }

    # process_daily: 解锁 T+1 + 更新市价(无待成交订单时只做这两件事)
    broker.process_daily(today.date(), snapshot)

    # 每日快照
    journal = TradeJournal(output_dir=config["live"].get("order_dir",
                                                         "live/output"))
    total = broker.get_total_value()
    peak = total  # 简化:仅记录当日值(完整净值曲线见 live 引擎)
    journal.log_daily_snapshot(today.date(), {
        "cash": broker.get_cash(),
        "market_value": broker.get_market_value(),
        "total_value": total,
        "n_positions": len(broker.get_positions()),
        "pnl": total - float(config["backtest"].get("initial_capital",
                                                    1_000_000)),
        "drawdown": 0.0,
    })
    journal.export_csv(f"daily_{today.strftime('%Y%m%d')}")
    broker.disconnect()
    logger.info(f"盯市完成: 总资产 {total:,.0f}, "
                f"持仓 {len(broker.get_positions())} 只")


def main():
    parser = argparse.ArgumentParser(description="quantlab2 每日定时任务")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--asof", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的动作")
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    setup_logger(config["logging"]["level"], config["logging"]["file"])

    today = pd.Timestamp(args.asof or datetime.now().date())

    # 1. 交易日检查
    if not is_trading_day(today.date()):
        logger.info(f"{today.date()} 非交易日,退出")
        return

    # 2. 调仓日判断
    if is_month_end_trading_day(today):
        action = f"调仓日: 完整 LiveEngine 调仓流程"
        logger.info(action)
        if args.dry_run:
            print(f"[dry-run] {action}")
            return
        from live.broker import create_broker
        from live.engine import LiveEngine
        broker_kind = args.broker
        if broker_kind == "qmt" and not args.confirm:
            logger.warning("QMT 未加 --confirm: 仅生成指令文件(安全模式)")
        try:
            broker = create_broker(broker_kind, config)
        except RuntimeError as e:
            logger.error(f"券商初始化失败: {e}")
            sys.exit(1)
        engine = LiveEngine(config, broker, confirm=args.confirm,
                            once=True,
                            asof=args.asof or today.strftime("%Y-%m-%d"))
        engine.run()
    else:
        action = f"普通交易日: 轻量盯市(T+1 解锁 + 更新市价)"
        logger.info(action)
        if args.dry_run:
            print(f"[dry-run] {action}")
            return
        mark_to_market_only(config, today)


if __name__ == "__main__":
    main()
