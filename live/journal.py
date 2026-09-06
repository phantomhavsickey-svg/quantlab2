"""
交易日志 — 记录每笔交易、每日快照,导出 CSV(移植自 quantlab paper_trade/
journal.py 并改进)。

改进:log_daily_snapshot 以快照 dict 入参,不直接摸 broker 内部属性
(quantlab 版本依赖 broker.cash/broker.positions,与 QMTBroker 不兼容)。
"""

import os
from datetime import date, datetime

import pandas as pd
import numpy as np
from loguru import logger

from live.broker import OrderResult, PositionInfo


class TradeJournal:
    """交易日志本(订单/成交/每日快照,内存记录 + CSV 导出)。"""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.orders_log: list[dict] = []
        self.trades_log: list[dict] = []
        self.daily_snapshots: list[dict] = []

    # ==================== 记录 ====================

    def log_order_result(self, req, result: OrderResult):
        """记录下单结果。"""
        self.orders_log.append({
            "time": datetime.now().isoformat(),
            "order_id": result.order_id,
            "symbol": req.symbol,
            "side": req.side,
            "quantity": req.quantity,
            "order_type": req.order_type,
            "ref_price": req.ref_price,
            "status": result.status.value
            if hasattr(result.status, "value") else str(result.status),
            "message": result.message,
        })

    def log_trade(self, order_id: str, symbol: str, side: str,
                  quantity: int, price: float, commission: float = 0.0,
                  stamp_tax: float = 0.0, slippage: float = 0.0,
                  trade_date=None):
        """记录一笔成交。"""
        amount = price * quantity
        total_cost = commission + stamp_tax + slippage
        self.trades_log.append({
            "date": str(trade_date or date.today()),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "amount": amount,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "slippage": slippage,
            "total_cost": total_cost,
            "net_proceeds": (amount - total_cost)
            * (1 if side == "sell" else -1),
        })

    def log_daily_snapshot(self, dt, snapshot: dict):
        """记录每日快照(snapshot: dict,与券商实现解耦)。"""
        self.daily_snapshots.append({
            "date": str(dt),
            "cash": snapshot.get("cash", 0.0),
            "market_value": snapshot.get("market_value", 0.0),
            "total_value": snapshot.get("total_value", 0.0),
            "n_positions": snapshot.get("n_positions", 0),
            "pnl": snapshot.get("pnl", 0.0),
            "drawdown": snapshot.get("drawdown", 0.0),
        })

    # ==================== 统计 ====================

    def compute_statistics(self) -> dict:
        """计算交易统计(费用统计与买卖笔数无关,全部成交都计入)。"""
        if not self.trades_log:
            return {"total_trades": 0}

        stats = {
            "total_trades": len(self.trades_log),
            "total_commission": sum(t["commission"] for t in self.trades_log),
            "total_stamp_tax": sum(t["stamp_tax"] for t in self.trades_log),
            "total_slippage": sum(t["slippage"] for t in self.trades_log),
            "total_cost": sum(t["total_cost"] for t in self.trades_log),
        }

        sells = [t for t in self.trades_log if t["side"] == "sell"]
        if not sells:
            stats.update({
                "completed_round_trips": 0, "win_rate": 0.0,
                "avg_pnl_per_trade": 0.0, "total_pnl": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0,
                "profit_factor": 0.0,
            })
            return stats

        net_proceeds = [s["net_proceeds"] for s in sells]
        losses = [x for x in net_proceeds if x < 0]
        stats.update({
            "completed_round_trips": len(sells),
            "win_rate": sum(1 for x in net_proceeds if x > 0)
            / len(net_proceeds),
            "avg_pnl_per_trade": float(np.mean(net_proceeds)),
            "total_pnl": float(sum(net_proceeds)),
            "best_trade": float(max(net_proceeds)),
            "worst_trade": float(min(net_proceeds)),
            "profit_factor": (sum(x for x in net_proceeds if x > 0)
                              / abs(sum(losses))) if losses else float("inf"),
        })
        return stats

    # ==================== 导出 ====================

    def export_csv(self, filename_prefix: str | None = None):
        """导出所有日志为 CSV。"""
        if filename_prefix is None:
            filename_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.trades_log:
            path = os.path.join(self.output_dir,
                                f"{filename_prefix}_trades.csv")
            pd.DataFrame(self.trades_log).to_csv(path, index=False,
                                                 encoding="utf-8-sig")
            logger.info(f"成交记录已导出: {path}")
        if self.daily_snapshots:
            path = os.path.join(self.output_dir,
                                f"{filename_prefix}_daily.csv")
            pd.DataFrame(self.daily_snapshots).to_csv(path, index=False,
                                                      encoding="utf-8-sig")
            logger.info(f"每日快照已导出: {path}")
        if self.orders_log:
            path = os.path.join(self.output_dir,
                                f"{filename_prefix}_orders.csv")
            pd.DataFrame(self.orders_log).to_csv(path, index=False,
                                                 encoding="utf-8-sig")
            logger.info(f"订单记录已导出: {path}")

    def generate_report(self) -> str:
        """生成文本版交易报告。"""
        stats = self.compute_statistics()
        lines = ["\n" + "=" * 50, "  模拟盘交易报告", "=" * 50]
        if stats["total_trades"] == 0:
            lines.append("  暂无交易记录")
            return "\n".join(lines)
        lines += [
            f"  总交易次数:  {stats['total_trades']}",
            f"  完整来回:    {stats.get('completed_round_trips', 0)}",
            f"  胜率:        {stats.get('win_rate', 0)*100:.1f}%",
            f"  总盈亏:      {stats.get('total_pnl', 0):,.0f} 元",
            f"  平均盈亏:    {stats.get('avg_pnl_per_trade', 0):,.0f} 元/笔",
            f"  最佳交易:    {stats.get('best_trade', 0):,.0f} 元",
            f"  最差交易:    {stats.get('worst_trade', 0):,.0f} 元",
            f"  利润因子:    {stats.get('profit_factor', 0):.2f}",
            f"  总佣金:      {stats.get('total_commission', 0):,.0f} 元",
            f"  总印花税:    {stats.get('total_stamp_tax', 0):,.0f} 元",
            f"  总交易成本:  {stats.get('total_cost', 0):,.0f} 元",
            "=" * 50,
        ]
        return "\n".join(lines)
