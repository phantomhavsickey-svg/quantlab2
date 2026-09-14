"""
报告生成 — 控制台文本报告和 Plotly HTML 图表。

plotly 缺失时 HTML 报告静默降级为仅控制台输出。
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class ReportGenerator:
    """生成回测报告。"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ==================== 控制台文本报告 ====================

    @staticmethod
    def console_report(metrics: dict) -> str:
        """生成格式化控制台文本报告。

        Returns:
            格式化的字符串
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  QuantLab2 (Transformer) 回测报告")
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # 绩效部分
        lines.append("")
        lines.append("[Performance]")
        lines.append("-" * 70)
        lines.append(f"  累计收益:    {metrics.get('cumulative_return', 0)*100:8.2f}%")
        lines.append(f"  年化收益:    {metrics.get('annual_return', 0)*100:8.2f}%")
        lines.append(f"  年化波动:    {metrics.get('annual_volatility', 0)*100:8.2f}%")
        lines.append(f"  夏普比率:    {metrics.get('sharpe_ratio', 0):8.2f}")
        lines.append(f"  Calmar比率:  {metrics.get('calmar_ratio', 0):8.2f}")
        lines.append(f"  最大回撤:    {metrics.get('max_drawdown', 0)*100:8.2f}%")
        lines.append(f"  胜率(日):    {metrics.get('win_rate', 0)*100:8.2f}%")

        if "information_ratio" in metrics:
            lines.append(f"  信息比率:    {metrics['information_ratio']:8.2f}")
        if "benchmark_cumulative_return" in metrics:
            lines.append(f"  基准收益:    {metrics['benchmark_cumulative_return']*100:8.2f}%")
            lines.append(f"  超额收益:    {metrics.get('excess_return', 0)*100:8.2f}%")

        # 交易部分
        if "total_trades" in metrics:
            lines.append("")
            lines.append("[Trades]")
            lines.append("-" * 70)
            lines.append(f"  总交易次数:  {metrics['total_trades']}")
            lines.append(f"  买入次数:    {metrics.get('n_buys', 'N/A')}")
            lines.append(f"  卖出次数:    {metrics.get('n_sells', 'N/A')}")
            lines.append(f"  交易总成本:  {metrics.get('total_cost', 0):,.0f} 元")
            lines.append(f"  交易胜率:    {metrics.get('win_rate_by_trade', 0)*100:8.2f}%")
            lines.append(f"  利润因子:    {metrics.get('profit_factor', 0):8.2f}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    # ==================== HTML 报告 ====================

    def html_report(self, equity_curve: pd.Series,
                    benchmark_curve: pd.Series | None = None,
                    daily_returns: pd.Series | None = None,
                    trades: pd.DataFrame | None = None,
                    metrics: dict | None = None,
                    filename: str | None = None) -> str:
        """生成包含 Plotly 图表的 HTML 报告。

        Returns:
            报告文件路径(plotly 未安装时返回 "")
        """
        if not HAS_PLOTLY:
            logger.warning("plotly 未安装，跳过 HTML 报告")
            return ""

        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        figs = [self.plot_equity_curve(equity_curve, benchmark_curve),
                self.plot_drawdown(equity_curve)]
        if daily_returns is not None:
            figs.append(self.plot_monthly_heatmap(
                self._returns_to_equity(daily_returns)))

        html_parts = ["<html><head><meta charset='utf-8'>"
                      "<title>QuantLab2 回测报告</title>"
                      "<style>body{font-family:Arial,sans-serif;"
                      "max-width:1200px;margin:0 auto;padding:20px;"
                      "background:#1e1e1e;color:#d4d4d4;}"
                      "h1{color:#569cd6;}"
                      ".metric{display:inline-block;margin:10px 20px;"
                      "padding:15px;background:#2d2d2d;"
                      "border-radius:8px;min-width:150px;text-align:center;}"
                      ".metric .value{font-size:28px;font-weight:bold;"
                      "color:#4ec9b0;}"
                      ".metric .label{font-size:12px;color:#888;}"
                      "</style></head><body>"
                      "<h1>📊 QuantLab2 (Transformer) 回测报告</h1>"]

        if metrics:
            html_parts.append("<div>")
            key_metrics = [
                ("累计收益", f"{metrics.get('cumulative_return', 0)*100:.2f}%"),
                ("年化收益", f"{metrics.get('annual_return', 0)*100:.2f}%"),
                ("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}"),
                ("最大回撤", f"{metrics.get('max_drawdown', 0)*100:.2f}%"),
                ("Calmar", f"{metrics.get('calmar_ratio', 0):.2f}"),
                ("胜率", f"{metrics.get('win_rate', 0)*100:.1f}%"),
            ]
            for label, value in key_metrics:
                html_parts.append(
                    f"<div class='metric'>"
                    f"<div class='value'>{value}</div>"
                    f"<div class='label'>{label}</div>"
                    f"</div>"
                )
            html_parts.append("</div>")

        for fig in figs:
            html_parts.append(fig.to_html(full_html=False,
                                           include_plotlyjs="cdn"))

        html_parts.append(f"<p style='color:#888;margin-top:40px;'>"
                          f"QuantLab2 Report · "
                          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                          f"</p></body></html>")

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        logger.info(f"HTML 报告已保存: {filepath}")
        return filepath

    # ==================== 图表 ====================

    @staticmethod
    def plot_equity_curve(equity_curve: pd.Series,
                          benchmark_curve: pd.Series | None = None) -> "go.Figure":
        """绘制净值曲线（对数坐标）。"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index, y=equity_curve.values,
            mode="lines", name="策略净值",
            line=dict(color="#4ec9b0", width=2),
        ))
        if benchmark_curve is not None:
            bm = benchmark_curve.reindex(equity_curve.index).ffill()
            bm = bm / bm.iloc[0]
            fig.add_trace(go.Scatter(
                x=bm.index, y=bm.values,
                mode="lines", name="基准净值",
                line=dict(color="#569cd6", width=1.5, dash="dash"),
                opacity=0.7,
            ))
        fig.update_layout(
            title="净值曲线",
            xaxis_title="日期", yaxis_title="净值",
            yaxis_type="log",
            template="plotly_dark",
            hovermode="x unified",
            height=500,
        )
        return fig

    @staticmethod
    def plot_drawdown(equity_curve: pd.Series) -> "go.Figure":
        """绘制回撤曲线。"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            mode="lines", fill="tozeroy", name="回撤",
            line=dict(color="#f44747", width=1),
            fillcolor="rgba(244,71,71,0.3)",
        ))
        fig.update_layout(
            title="回撤曲线",
            xaxis_title="日期", yaxis_title="回撤 (%)",
            template="plotly_dark",
            hovermode="x unified",
            height=350,
        )
        return fig

    @staticmethod
    def plot_monthly_heatmap(equity_curve: pd.Series) -> "go.Figure":
        """绘制月度收益热力图。"""
        monthly = equity_curve.resample("ME").last().pct_change() * 100
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.year, monthly.index.month],
            names=["year", "month"]
        )
        heatmap_data = monthly.unstack(level="month")
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=["1月","2月","3月","4月","5月","6月",
               "7月","8月","9月","10月","11月","12月"],
            y=heatmap_data.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(heatmap_data.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 10, "color": "#333"},
            hoverongaps=False,
        ))
        fig.update_layout(
            title="月度收益热力图 (%)",
            template="plotly_dark",
            height=400,
            xaxis=dict(side="top"),
        )
        return fig

    @staticmethod
    def _returns_to_equity(returns: pd.Series) -> pd.Series:
        return (1 + returns).cumprod()
