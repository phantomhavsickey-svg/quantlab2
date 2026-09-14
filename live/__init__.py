"""
Live 实时行情模块 — 新浪财经 HTTP API(免费,无需 key)。

格式: http://hq.sinajs.cn/list=sh600000,sz000001
字段解析 (A股):
  0:  股票名称
  1:  今开盘
  2:  昨收盘
  3:  当前价
  4:  最高价
  5:  最低价
  6:  买一价
  7:  卖一价
  8:  成交量(股)
  9:  成交额(元)
  10-14: 买一~买五量
  15-19: 卖一~卖五量
  30: 日期
  31: 时间
"""

import time
import re
import requests
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Quote:
    """实时行情快照。"""
    symbol: str
    name: str
    price: float        # 当前价
    open: float         # 今开
    high: float         # 最高
    low: float          # 最低
    prev_close: float   # 昨收
    volume: float       # 成交量(股)
    amount: float       # 成交额(元)
    bid: float          # 买一价
    ask: float          # 卖一价
    change_pct: float   # 涨跌幅 %
    time: str           # 时间 HH:MM:SS
    date: str           # 日期 YYYY-MM-DD

    def __repr__(self):
        sign = "+" if self.change_pct >= 0 else ""
        return (f"<{self.name}({self.symbol}) {self.price:.2f} "
                f"({sign}{self.change_pct:.2f}%) "
                f"V={self.volume/10000:.0f}万手>")


class SinaQuoteFeed:
    """新浪财经实时行情源。

    用法:
        feed = SinaQuoteFeed()
        quotes = feed.fetch(["000001", "600519"])
    """

    BASE_URL = "http://hq.sinajs.cn/list="
    HEADERS = {"Referer": "https://finance.sina.com.cn"}

    # 新浪代码前缀
    @staticmethod
    def to_sina_code(symbol: str) -> str:
        """将 6位代码 转为新浪格式: 000001 -> sz000001, 600519 -> sh600519。"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith(("60", "68")):
            return f"sh{symbol}"
        else:
            return f"sz{symbol}"

    @staticmethod
    def from_sina_code(code: str) -> str:
        """从新浪格式转回6位代码。"""
        return code[2:]

    def fetch(self, symbols: list[str]) -> dict[str, Quote]:
        """获取实时行情。

        Args:
            symbols: 6位股票代码列表

        Returns:
            {symbol: Quote} 字典(网络失败/解析失败自动跳过)
        """
        codes = [self.to_sina_code(s) for s in symbols]
        results = {}
        batch_size = 50
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            url = self.BASE_URL + ",".join(batch)
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                resp.encoding = "gbk"
                results.update(self._parse_response(resp.text))
            except Exception as e:
                logger.warning(f"行情请求失败 (batch {i}): {e}")
            time.sleep(0.1)  # 批次间短暂间隔
        return results

    # ==================== 解析 ====================

    def _parse_response(self, text: str) -> dict[str, Quote]:
        """解析新浪返回的 var hq_str_xxx="..." 格式。"""
        results = {}
        pattern = r'var hq_str_(\w+)="([^"]*)"'
        for code, data_str in re.findall(pattern, text):
            symbol = self.from_sina_code(code)
            try:
                quote = self._parse_quote(symbol, data_str)
                if quote:
                    results[symbol] = quote
            except Exception as e:
                logger.debug(f"解析 {symbol} 失败: {e}")
        return results

    def _parse_quote(self, symbol: str, data_str: str) -> Optional[Quote]:
        """解析单只股票的行情数据。"""
        fields = data_str.split(",")
        if len(fields) < 32:
            return None

        try:
            price = self._float(fields[3])
            if price <= 0:
                return None
            prev_close = self._float(fields[2])
            change_pct = 0.0
            if prev_close > 0:
                change_pct = (price - prev_close) / prev_close * 100

            return Quote(
                symbol=symbol,
                name=fields[0],
                price=price,
                open=self._float(fields[1]),
                high=self._float(fields[4]),
                low=self._float(fields[5]),
                prev_close=prev_close,
                volume=self._float(fields[8]),
                amount=self._float(fields[9]),
                bid=self._float(fields[6]),
                ask=self._float(fields[7]),
                change_pct=round(change_pct, 2),
                date=fields[30],
                time=fields[31] if len(fields) > 31 else "",
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"解析 {symbol} 字段错误: {e}")
            return None

    @staticmethod
    def _float(val: str) -> float:
        """安全转 float。"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
