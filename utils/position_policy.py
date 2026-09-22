"""
分数带位持仓策略 — 建仓 / 补仓 / 减仓 / 清仓的唯一状态机。

回测与实盘共用这一份实现:这里只把"分数"翻成"目标权重 + 状态推进意图",
股数算术仍然交给 utils/sizing.py,所以不会重现"回测按目标建仓、实盘按现金
摊派"那类两端口径分叉。

四条带位(分数 = 模型输出;regressor + mse 口径下是 20 个交易日的预测收益率,
0.03 = 预期 +3%):

    分数 ≥ 买入线,未持有      建仓。权重在 base_weight ~ max_entry_weight 之间
                              按分数在 [buy_score, strong_score] 上线性插值。
    分数 ≥ 买入线,已持有      要比"上一次动作的参考分数"再涨够一个步长才补一档,
                              补仓后单票市值不超过 max_position_weight。
    卖出线 ≤ 分数 < 买入线    比参考分数跌够一个步长就减仓;减完不足
                              min_hold_weight 时直接清仓(不留一手以下的碎仓)。
    分数 < 卖出线             清仓。全策略里唯一不受 min_trade_value 限制的动作。

步长在建仓时冻结:Δ = max(step_score_floor, step_multiplier × (建仓分数 − 买入线))。
含义是"这只票当初比买入线强多少,往后就要比上次动作再强同样多才值得加一档";
补仓与减仓共用同一个 Δ,方向相反 —— 这就是"按分数的相对变化触发"。

**单子没成交就不推进状态**:最小交易额不够、预算不够、涨停/跌停/停牌没成交、
只部分成交,一律保留原参考分数。下一轮条件仍然成立,会自动补足差额。否则会出现
"分数已经用掉了但仓位没变"这种单向漏掉的坑。

上面四条带只管**单票**。组合级"总共敢下多少注"由 utils/exposure.py 按轮给一个
`max_total_pct_override`:上限低于当前仓位时等比降杠杆(de_gross 动作,不推进
分数状态、不留减仓价),门控触发时补仓与新仓一并挂起 —— 降杠杆那一轮再把回笼的
钱花出去等于没降。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from datetime import date as _date

from loguru import logger

from utils.sizing import Plan, rebalance_plan

_EPS = 1e-9
_INT_FIELDS = ("max_names", "max_steps_per_eval", "trim_price_ttl_days")


# ==================== 配置 ====================

@dataclass(frozen=True)
class PolicyConfig:
    """带位参数。全部字段可在 config.yaml 的 position_policy 段覆盖。"""

    # --- 分数带 ---
    buy_score: float = 0.02             # 买入线:低于它不建仓
    sell_score: float = 0.0             # 清仓线:低于它无条件清仓
    strong_score: float = 0.05          # 建仓给到 max_entry_weight 的分数
    step_score_floor: float = 0.005     # 一档的最小分数步长(Δ 下限)
    step_multiplier: float = 1.0        # Δ = multiplier × (建仓分数 − 买入线)

    # --- 仓位带 ---
    base_weight: float = 0.05           # 建仓基准仓位(占总资产)
    max_entry_weight: float = 0.08      # 分数很高时的建仓上限
    max_position_weight: float = 0.16   # 补仓后的单票上限
    min_hold_weight: float = 0.025      # 减仓后的最低仓位,低于它就清仓
    add_step_weight: float = 0.05       # 补一档加的仓位
    trim_step_weight: float = 0.05      # 减一档降的仓位
    max_steps_per_eval: int = 2         # 一次评估最多补/减几档

    # --- 执行约束 ---
    min_trade_value: float = 10000.0    # 单笔补/减的最小名义额(清仓不受限)
    max_names: int = 12                 # 同时在市股票数上限
    add_reserve_weight: float = 0.05    # 留给后续补仓的额度,新仓不能把它吃光
    max_total_pct: float = 0.95         # 组合总仓位上限
    trim_price_ttl_days: int = 60       # "减仓价之上禁止补仓"的有效期(自然日)
    # 买入单边费率(佣金+滑点)。计入预算是必须的:不按"含费"计,策略会把仓位
    # 买到 95% 的名义额,现金就不够付手续费,回测端 scale_buys_to_budget 于是
    # 每轮缩量 → 补仓永远算"未足量成交"、参考分数永不推进、下一轮重复补仓。
    fee_rate_buy: float = 0.0013

    # ==================== 构造与自检 ====================

    def __post_init__(self):
        # 直接构造(测试/代码里)和从 config 构造走同一条自检
        self.validate()

    @classmethod
    def from_dict(cls, d: dict | None, **overrides) -> "PolicyConfig":
        """从 config.yaml 的 position_policy 段构造。写错的参数当场报错。

        overrides 用来把外部口径灌进来(live.risk.max_total_pct、
        market 的佣金+滑点),配置里显式写过的参数优先。
        """
        d = dict(d or {})
        fields = set(cls.__dataclass_fields__)
        unknown = (set(d) | set(overrides)) - fields
        if unknown:
            raise ValueError(
                f"position_policy 有未知参数 {sorted(unknown)},"
                f" 可用参数: {sorted(fields)}")
        kwargs = {k: (int(v) if k in _INT_FIELDS else float(v))
                  for k, v in d.items()}
        for k, v in overrides.items():
            if k in kwargs or v is None:
                continue
            kwargs[k] = int(v) if k in _INT_FIELDS else float(v)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self):
        """互相矛盾的写法在启动时就挡掉 —— 这些洞不挡会表现为"策略静默不生效"。"""
        e = []
        if self.sell_score > self.buy_score:
            e.append(f"sell_score({self.sell_score}) 不能高于 buy_score"
                     f"({self.buy_score}),否则减仓带为空、建仓即挂")
        if not self.base_weight <= self.max_entry_weight <= self.max_position_weight:
            e.append("仓位必须满足 base_weight ≤ max_entry_weight ≤ max_position_weight")
        if self.strong_score <= self.buy_score:
            e.append("strong_score 必须高于 buy_score,否则建仓权重插值除零")
        if self.step_score_floor <= 0 or self.step_multiplier <= 0:
            e.append("step_score_floor / step_multiplier 必须为正,否则补仓条件恒成立")
        if self.max_names <= 0 or self.max_steps_per_eval <= 0:
            e.append("max_names / max_steps_per_eval 必须为正")
        if not 0 < self.max_total_pct <= 1:
            e.append("max_total_pct 必须落在 (0, 1]")
        if not 0 <= self.add_reserve_weight < self.max_total_pct:
            e.append("add_reserve_weight 必须落在 [0, max_total_pct) 内")
        if not 0 <= self.min_hold_weight < self.base_weight:
            e.append("min_hold_weight 必须落在 [0, base_weight) 内,"
                     "否则减一档必然直接清仓,减仓带形同虚设")
        if self.add_step_weight <= 0 or self.trim_step_weight <= 0:
            e.append("add_step_weight / trim_step_weight 必须为正")
        if self.min_trade_value < 0:
            e.append("min_trade_value 不能为负")
        # 建仓名额必须装得进"扣掉补仓预留后的额度",否则预留是空的、补仓无钱可用
        room = int((self.max_total_pct - self.add_reserve_weight)
                   / (self.base_weight * (1.0 + self.fee_rate_buy)))
        if self.max_names > room:
            e.append(f"max_names({self.max_names}) × base_weight"
                     f"({self.base_weight}) 超过建仓可用额度(含费) "
                     f"(max_total_pct − add_reserve_weight)"
                     f"/(1+fee) = "
                     f"{(self.max_total_pct - self.add_reserve_weight)/(1+self.fee_rate_buy):.2f},"
                     f"最多只进得去 {room} 只;请调低 max_names 或仓位基准")
        if e:
            raise ValueError("position_policy 参数矛盾: " + " | ".join(e))

    # ==================== 派生量 ====================

    def entry_weight(self, score: float) -> float:
        """建仓权重:买入线 → strong_score 之间,从 base_weight 线性升到上限。"""
        t = (float(score) - self.buy_score) / (self.strong_score - self.buy_score)
        t = min(max(t, 0.0), 1.0)
        return self.base_weight + (self.max_entry_weight - self.base_weight) * t

    def step_of(self, entry_score: float) -> float:
        """该标的的触发步长 Δ,建仓时冻结,之后不随分数变化。"""
        return max(self.step_score_floor,
                   self.step_multiplier * (float(entry_score) - self.buy_score))


# ==================== 每只股票的状态 ====================

@dataclass
class NameState:
    """单只股票的策略状态。实盘序列化进 live/state/policy_state.json。"""
    entry_score: float                  # 建仓时分数
    ref_score: float                    # 最近一次动作后的分数基准(步长从它起算)
    step: float                         # 冻结的触发步长 Δ
    trim_price: float | None = None     # 最近一次**实际成交**的减仓价
    trim_date: str | None = None        # 上一条的日期(ISO),用于有效期
    adds: int = 0
    trims: int = 0

    def to_dict(self) -> dict:
        return {"entry_score": self.entry_score, "ref_score": self.ref_score,
                "step": self.step, "trim_price": self.trim_price,
                "trim_date": self.trim_date, "adds": self.adds,
                "trims": self.trims}

    @classmethod
    def from_dict(cls, d: dict) -> "NameState":
        tp = d.get("trim_price")
        step = float(d.get("step", 0.0) or 0.0)
        return cls(entry_score=float(d.get("entry_score", 0.0)),
                   ref_score=float(d.get("ref_score", 0.0)),
                   step=step if step > 0 else PolicyConfig().step_score_floor,
                   trim_price=None if tp is None else float(tp),
                   trim_date=d.get("trim_date"),
                   adds=int(d.get("adds", 0) or 0),
                   trims=int(d.get("trims", 0) or 0))


@dataclass
class Intent:
    """一次"想动"的提议。只有对应订单**足量成交**后才写回 NameState。"""
    kind: str            # entry / add / trim / cap / exit / exit_by_trim / de_gross
    symbol: str
    weight: float = 0.0          # 目标权重
    delta_value: float = 0.0     # 计划成交名义额(正=买,负=卖)
    shares: int = 0              # 计划成交股数(整手取整后)
    score: float = 0.0           # 触发时的分数,组合预算排队用
    state: NameState | None = None   # 成交后要落的状态(entry/add/trim 都有)


@dataclass
class PolicyPlan:
    """策略一次评估的完整输出,回测与实盘拿到的是同一个结构。"""
    plan: Plan
    weights: dict = field(default_factory=dict)
    keep: frozenset = frozenset()
    intents: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)   # kind → 次数
    gross_weight: float = 0.0

    @property
    def n_trades(self) -> int:
        return self.plan.n_trades


# ==================== 主逻辑 ====================

def policy_from_config(config: dict) -> PolicyConfig | None:
    """config.yaml → PolicyConfig;position_policy.enabled=false 时返回 None。

    max_total_pct 与买入费率不在策略段里重复一遍,而是从 live.risk / market 取:
    回测和实盘因此共用同一个总仓位上限、同一个含费预算,两端算出的股数才对得上。
    策略段里显式写过的值优先。
    """
    pp = dict(config.get("position_policy") or {})
    if not pp.pop("enabled", False):
        return None
    mkt = config.get("market") or {}
    fee = float(mkt.get("commission_rate", 0.0003)) + \
        float(mkt.get("slippage_rate", 0.001))
    cap = ((config.get("live") or {}).get("risk") or {}).get("max_total_pct")
    return PolicyConfig.from_dict(pp, max_total_pct=cap, fee_rate_buy=fee)


def decide(scores: dict, held: dict, prices: dict, states: dict,
           total_value: float, cfg: PolicyConfig, asof=None, *,
           max_total_pct_override: float | None = None,
           entry_allowed: bool = True):
    """分数 → (目标权重, 冻结名单, 动作意图, 未动作原因, 当前权重)。

    Args:
        scores: {symbol: 分数},**全截面**(含分数不够高的候选)
        held: {symbol: 当前股数}
        prices: {symbol: 成交价},与 utils/sizing 同一基准
        states: {symbol: NameState};就地修改两处 —— 减仓价到期解除、存量持仓补基线
        total_value: 本次评估时点的总资产(现金 + 持仓市值)
        asof: 评估日(date / ISO 字符串 / Timestamp),用于减仓价有效期
        max_total_pct_override: 组合级暴露层(`utils/exposure.py`)给出的本轮总仓位
                上限,None = 用 cfg.max_total_pct。它只在本轮生效,所以**不走
                validate()** —— 用低上限去套 max_names 会把一次性降杠杆判成
                "参数矛盾",而实际要的是"这一轮进不去那么多"。
        entry_allowed: False = 本轮不许开新仓也不许补仓(门控触发)。已挂的减仓/
                清仓不受影响,那是在降风险。

    Returns:
        weights: {symbol: 目标权重};0.0 = 清仓;**缺席 = 本轮不碰**(缺价/缺分数)
        keep:    即便算得出差额也不下单的名字(带内不动、被预算/最小额砍掉)
        intents: {symbol: Intent}
        notes:   {symbol: 为什么没动}
        cur:     {symbol: 当前权重},作废意图时把目标还原成它
    """
    weights: dict[str, float] = {}
    keep: set[str] = set()
    intents: dict[str, Intent] = {}
    notes: dict[str, str] = {}

    tv = float(total_value)
    # 暴露层只允许**往下**调上限:写反了也不会因为低波动而自动加杠杆
    cap = (cfg.max_total_pct if max_total_pct_override is None
           else min(float(max_total_pct_override), cfg.max_total_pct))
    px = {str(s): float(p) for s, p in (prices or {}).items()
          if _finite(p) and float(p) > 0}
    sc = {str(s): float(v) for s, v in (scores or {}).items() if _finite(v)}
    hd = {str(s): int(q) for s, q in (held or {}).items() if int(q) > 0}
    if not _finite(tv) or tv <= 0:
        return weights, keep, intents, {"_": "总资产非正,本轮不动"}, {}
    today = _as_date(asof)

    # 减仓价约束到期解除 —— 不然一次减仓会把这只票永久锁死
    for s, st in list((states or {}).items()):
        if st.trim_price is None:
            continue
        age = _age_days(st.trim_date, today)
        if age is None or age > cfg.trim_price_ttl_days:
            notes[s] = (f"减仓价 {st.trim_price:.2f} 的约束"
                        f"{'缺日期无法计时' if st.trim_date is None else f'已 {age} 天'}"
                        f",超过 {cfg.trim_price_ttl_days} 天有效期,解除")
            states[s] = replace(st, trim_price=None, trim_date=None)

    cur = {s: hd[s] * px[s] / tv for s in hd if s in px}
    entries: list[tuple[float, str, float]] = []   # (分数, symbol, 建仓权重)

    for s in sorted(set(hd) | set(sc)):
        x, p = sc.get(s), px.get(s)
        w0 = cur.get(s, 0.0)

        if p is None or x is None:
            if s in hd:
                keep.add(s)      # 没有价格就没有下单依据,没有分数就无从判断带位
                notes[s] = "缺成交价" if p is None else "当日无预测分数"
            continue

        st = (states or {}).get(s)

        # ---------------- 未持有 ----------------
        if s not in hd:
            if x < cfg.buy_score:
                continue
            if st is not None and st.trim_price is not None \
                    and p > st.trim_price + _EPS:
                notes[s] = (f"分数达建仓线,但现价 {p:.2f} 高于减仓价 "
                            f"{st.trim_price:.2f},禁止重建")
                continue
            entries.append((x, s, cfg.entry_weight(x)))
            continue

        # ---------------- 已持有 ----------------
        if x < cfg.sell_score:
            weights[s] = 0.0
            intents[s] = Intent("exit", s, 0.0, -w0 * tv)
            continue

        if st is None:
            # 存量持仓 / 状态文件丢失:按当前分数补一个基线,本轮先不动
            base = max(x, cfg.buy_score)
            states[s] = NameState(entry_score=base, ref_score=x,
                                  step=cfg.step_of(base))
            weights[s] = w0
            keep.add(s)
            notes[s] = "无策略状态,已按当前分数建立基线,本轮不动"
            continue

        # 价格漂移造成的超配先压回上限(风险动作,优先于分数动作)
        if w0 > cfg.max_position_weight + _EPS:
            tgt = cfg.max_position_weight
            cut = (w0 - tgt) * tv
            if cut < cfg.min_trade_value:
                weights[s] = w0
                keep.add(s)
                notes[s] = (f"超单票上限({w0:.1%}>{cfg.max_position_weight:.0%}),"
                            f"但压回只需卖 {cut:,.0f} 元,不足最小交易额,不动")
            else:
                weights[s] = tgt
                intents[s] = Intent("cap", s, tgt, -cut, state=st)
            continue

        gate = st.step
        if x >= st.ref_score + gate - _EPS:
            n = _steps(x - st.ref_score, gate, cfg.max_steps_per_eval)
            tgt = min(w0 + n * cfg.add_step_weight, cfg.max_position_weight)
            add = (tgt - w0) * tv
            if tgt <= w0 + _EPS:
                weights[s] = w0
                keep.add(s)
                notes[s] = f"涨够一档,但已在 {cfg.max_position_weight:.0%} 上限,不补"
            elif st.trim_price is not None and p > st.trim_price + _EPS:
                weights[s] = w0
                keep.add(s)
                notes[s] = (f"分数涨够一档,但现价 {p:.2f} 高于减仓价 "
                            f"{st.trim_price:.2f},禁止补仓")
            elif add < cfg.min_trade_value:
                weights[s] = w0
                keep.add(s)
                notes[s] = (f"补一档 {add:,.0f} 元 < 最小交易额 "
                            f"{cfg.min_trade_value:,.0f},不补")
            else:
                weights[s] = tgt
                intents[s] = Intent(
                    "add", s, tgt, add, score=x,
                    state=replace(st, ref_score=st.ref_score + n * gate,
                                  adds=st.adds + 1))
        elif x <= st.ref_score - gate + _EPS:
            n = _steps(st.ref_score - x, gate, cfg.max_steps_per_eval)
            tgt = w0 - n * cfg.trim_step_weight
            cut = (w0 - tgt) * tv
            if tgt < cfg.min_hold_weight - _EPS:
                weights[s] = 0.0
                intents[s] = Intent(
                    "exit_by_trim", s, 0.0, -w0 * tv,
                    state=replace(st, ref_score=st.ref_score - n * gate,
                                  trims=st.trims + 1))
                notes[s] = (f"减一档只剩 {tgt:.1%} < min_hold_weight "
                            f"{cfg.min_hold_weight:.1%},改为清仓")
            elif cut < cfg.min_trade_value:
                weights[s] = w0
                keep.add(s)
                notes[s] = (f"减一档 {cut:,.0f} 元 < 最小交易额 "
                            f"{cfg.min_trade_value:,.0f},不减")
            else:
                weights[s] = tgt
                intents[s] = Intent(
                    "trim", s, tgt, -cut,
                    state=replace(st, ref_score=st.ref_score - n * gate,
                                  trims=st.trims + 1))
        else:
            weights[s] = w0          # 带内:不补不减,按现价原样持有
            keep.add(s)

    # ---------------- 组合级降杠杆 ----------------
    # 上限被暴露层压低时,"减半"必须真的卖出去。分数带自己不会降总仓位:
    # 分数还在买入线之上的名字会永远留在带内,所以只在权重上让名额是不够的。
    # 按**当前**权重等比缩,保住分数带决定的相对结构。
    cur_gross = sum(cur.values())
    allow_new = entry_allowed and cur_gross <= cap + _EPS
    if cap < cur_gross - _EPS:
        k = cap / cur_gross
        for s in sorted(cur, key=lambda x: (-cur[x], x)):
            if s in intents or weights.get(s, cur[s]) <= 0:
                continue            # 已挂减仓/清仓意图的由它自己释放预算
            w0 = cur[s]
            tgt = w0 * k
            cut = (w0 - tgt) * tv
            if cut < cfg.min_trade_value:
                weights[s] = w0
                keep.add(s)
                notes[s] = (f"总仓位上限 {cap:.1%},降杠杆只需卖 {cut:,.0f} 元 "
                            f"< 最小交易额 {cfg.min_trade_value:,.0f},不动")
                continue
            weights[s] = tgt
            # 带内持有的名字本来在 keep 里(锁住不动),降杠杆要卖它就必须先解锁,
            # 否则 rebalance_plan 会把目标锁回当前股数,这轮降杠杆静默失效。
            keep.discard(s)
            intents[s] = Intent("de_gross", s, tgt, -cut,
                                state=(states or {}).get(s))

    _apply_budget(weights, keep, intents, notes, cur, entries, cfg, tv,
                  cap, allow_new)
    return weights, frozenset(keep), intents, notes, cur


def _apply_budget(weights, keep, intents, notes, cur, entries,
                  cfg: PolicyConfig, tv: float, cap: float,
                  allow_new: bool = True) -> float:
    """组合层约束:总仓位上限、补仓预留额度、股票数上限。

    优先级是**已有持仓的风险动作 > 补仓 > 新仓**,同层按分数降序。超预算时先把
    目标压到剩余额度;压完仍不足最小交易额就整个作废(状态不推进,下轮预算宽了
    自动重试)。不做"人人等比缩一档",那会让每笔单子刚好卡在最小交易额附近,
    最小佣金占比最难控。

    cap 是**本轮生效**的总仓位上限(暴露层可能把它压到 cfg.max_total_pct 之下)。
    allow_new=False 时补仓与新仓一并挂起:降杠杆那一轮把钱再花出去等于没降。
    """
    # 基线用"动作前的权重":补仓/建仓要挤进预算,减仓与清仓则释放预算。
    # 直接对 weights 求和会把补仓自己的目标也算进基线,于是永远"总仓位已满"。
    base = {}
    for s in set(weights) | set(cur):
        w = weights.get(s, cur.get(s, 0.0))
        if s in intents and intents[s].kind == "add":
            w = cur.get(s, 0.0)
        base[s] = w
    gross = sum(base.values())
    # 新花钱的部分按含费口径占用预算(已有持仓的成本已经付过了,按 1:1 计)
    mult = 1.0 + cfg.fee_rate_buy

    if not allow_new:
        for s in [k for k, i in intents.items() if i.kind == "add"]:
            _void(weights, keep, intents, notes, s, cur,
                  "本轮不新增仓位(暴露层:降杠杆或 IC 门控),取消补仓")
        for _, s, _ in entries:
            notes[s] = "本轮不新增仓位(暴露层:降杠杆或 IC 门控),新仓挂起"
        return gross

    for s in sorted([s for s, i in intents.items() if i.kind == "add"],
                    key=lambda k: (-intents[k].score, k)):
        it = intents[s]
        w0 = cur.get(s, 0.0)
        room = (cap - gross) / mult
        if room <= _EPS:
            _void(weights, keep, intents, notes, s, cur, "总仓位已满,取消补仓")
            continue
        allow = min(it.weight, w0 + room)
        if (allow - w0) * tv < cfg.min_trade_value:
            _void(weights, keep, intents, notes, s, cur,
                  f"总仓位只剩 {room:.1%} 额度,补仓额不足最小交易额")
            continue
        it.weight = allow
        it.delta_value = (allow - w0) * tv
        weights[s] = allow
        gross += (allow - w0) * mult

    n_held = len([s for s in cur if weights.get(s, 0.0) > 0])
    # 新仓的额度既要扣掉补仓预留,更要扣掉**已经建好的仓位** —— 只在当轮内累加
    # used_new 会让每一轮都重新"满仓建仓",几轮下来总仓位就冲过上限了。
    entry_cap = min(cap - cfg.add_reserve_weight, cap - gross)
    used_new = 0.0
    for x, s, w in sorted(entries, key=lambda t: (-t[0], t[1])):
        if used_new + w * mult > entry_cap + _EPS:
            notes[s] = (f"新仓合计将达 {used_new + w * mult:.1%}(含费) > "
                        f"可用额度 {entry_cap:.1%}"
                        f"(max_total_pct − add_reserve_weight / − 已有仓位)")
            continue
        if n_held >= cfg.max_names:
            notes[s] = f"持仓数将达 {n_held + 1} > max_names {cfg.max_names}"
            continue
        if w * tv < cfg.min_trade_value:
            notes[s] = (f"建仓 {w * tv:,.0f} 元 < 最小交易额 "
                        f"{cfg.min_trade_value:,.0f}")
            continue
        weights[s] = w
        intents[s] = Intent("entry", s, w, w * tv, score=x,
                            state=NameState(entry_score=x, ref_score=x,
                                            step=cfg.step_of(x)))
        n_held += 1
        used_new += w * mult
        gross += w * mult
    return gross


def _void(weights, keep, intents, notes, s, cur, why):
    """作废一个意图:目标权重回到当前权重,参考分数不动(下轮自动重试)。"""
    intents.pop(s, None)
    weights[s] = cur.get(s, 0.0)
    keep.add(s)
    notes[s] = why


# ==================== 与 sizing 的接缝 ====================

def plan(scores: dict, held: dict, prices: dict, states: dict,
         total_value: float, cfg: PolicyConfig, *,
         lot_size: int = 100, asof=None,
         max_total_pct_override: float | None = None,
         entry_allowed: bool = True) -> PolicyPlan:
    """策略 → 可执行 Plan。回测引擎与实盘指令构造都走这一个函数。

    max_total_pct_override / entry_allowed 是组合级暴露层(utils/exposure.py)
    的输入,不传就是纯分数带口径(与旧的已发布数字一致)。
    """
    weights, keep_frozen, intents, notes, cur = decide(
        scores, held, prices, states, total_value, cfg, asof=asof,
        max_total_pct_override=max_total_pct_override,
        entry_allowed=entry_allowed)
    keep = set(keep_frozen)
    cap = (cfg.max_total_pct if max_total_pct_override is None
           else min(float(max_total_pct_override), cfg.max_total_pct))

    # 预算已按权重在 _apply_budget 里算完,这里不再二次裁剪(normalize=False)
    rp = rebalance_plan(weights, held, prices, total_value, lot_size=lot_size,
                        max_total_pct=1.0, normalize=False, keep=keep)
    enforce_min_trade_value(rp, {str(s): int(q) for s, q in (held or {}).items()},
                            prices, cfg.min_trade_value)

    # 整手取整 / 最小交易额砍掉的单子 → 意图一并作废,状态不推进
    for s in list(intents):
        it = intents[s]
        buy = it.delta_value > 0
        got = int(rp.buys.get(s, 0) if buy else rp.sells.get(s, 0))
        if got <= 0:
            _void(weights, keep, intents, notes, s, cur,
                  "整手取整后不足一手,买入作废" if buy
                  else "卖出单被最小交易额/整手规则砍掉,作废")
            continue
        it.shares = got
        it.weight = weights.get(s, it.weight)
        it.delta_value = got * float(prices[s]) * (1 if buy else -1)

    actions: dict[str, int] = {}
    for it in intents.values():
        actions[it.kind] = actions.get(it.kind, 0) + 1
    gross = sum(weights.values())
    if gross > cap + 1e-6:
        logger.debug(f"策略目标总仓位 {gross:.1%} 超过本轮上限 {cap:.0%}"
                     f"(含缺价持仓或降杠杆单被最小交易额砍掉,无法计量)")
    return PolicyPlan(plan=rp, weights=weights, keep=frozenset(keep),
                      intents=intents, notes=notes, actions=actions,
                      gross_weight=gross)


def sync_intent_shares(intents: dict, rp: Plan) -> None:
    """把撮合前被**现金预算**缩量的单子同步回意图,让状态按"实际下得出多少"推进。

    plan() 内部已经按整手/最小交易额同步过一次;回测引擎和实盘还会在最后一刻
    按可用现金再缩一次量(那是"这一档只能买到这么多",不是"这一档没做成")。
    不缩量回写的话,补仓会永远被判成未足量成交:参考分数不推进 → 下一轮同一档
    条件仍成立 → 再补一次,仓位一路滑到 16% 上限。就地修改 intents。
    """
    for s, it in intents.items():
        got = int((rp.buys if it.delta_value > 0 else rp.sells).get(s, 0))
        if got > 0:
            it.shares = got


def enforce_min_trade_value(rp: Plan, held: dict, prices: dict,
                            min_value: float, *,
                            allow_full_exit: bool = True) -> Plan:
    """砍掉名义额不足 min_value 的买卖单;卖到清仓不受此限制。

    A 股不足一手的零股只能在清仓时一次性卖出,所以"最后那一手"永远放行,
    否则碎仓会永远卡在账上。就地修改并返回同一个 Plan。
    """
    if min_value <= 0:
        return rp
    held = {str(s): int(q) for s, q in (held or {}).items() if int(q) > 0}
    buys, bv = {}, 0.0
    for s, q in rp.buys.items():
        amt = q * float(prices[s])
        if amt >= min_value - _EPS:
            buys[s] = q
            bv += amt
    sells, sv = {}, 0.0
    for s, q in rp.sells.items():
        amt = q * float(prices[s])
        if amt >= min_value - _EPS or (allow_full_exit and q >= held.get(s, 0)):
            sells[s] = q
            sv += amt
    rp.buys, rp.buy_value = buys, bv
    rp.sells, rp.sell_value = sells, sv
    return rp


# ==================== 成交回报 → 状态 ====================

def apply_fills(states: dict, intents: dict, before: dict, after: dict,
                fill_price: dict | None = None, asof=None) -> dict:
    """按**实际股数变化**提交状态,返回 {symbol: 提交结果} 便于打日志。

    只有足量成交才推进参考分数:部分成交时不推进,下轮同一档条件仍成立,会自动
    补足差额。减仓价记的是实际成交价,不是下单时的参考价。
    """
    out: dict[str, str] = {}
    before = {str(s): int(v) for s, v in (before or {}).items()}
    after = {str(s): int(v) for s, v in (after or {}).items()}
    fp = {str(s): float(v) for s, v in (fill_price or {}).items() if _finite(v)}
    today = _as_date(asof)
    iso = today.isoformat() if today else None

    for s, it in (intents or {}).items():
        b, a = before.get(s, 0), after.get(s, 0)
        d = a - b
        want = int(it.shares or 0)

        if it.kind in ("entry", "add") and want > 0 and d >= want and it.state:
            states[s] = it.state
            out[s] = (f"建仓提交 {d} 股(步长 Δ={it.state.step:.4f})"
                      if it.kind == "entry"
                      else f"补仓提交 +{d} 股,参考分数→{it.state.ref_score:.4f}")
        elif it.kind in ("trim", "cap", "exit_by_trim",
                         "de_gross") and want > 0 and -d >= want:
            price = fp.get(s)
            if it.kind in ("cap", "de_gross"):
                out[s] = (f"{'压上限' if it.kind == 'cap' else '降杠杆'}提交 "
                          f"-{d} 股(组合级/价格的风险动作,不改分数状态、"
                          f"不设减仓价约束)")
            elif price is None:
                out[s] = f"减仓提交 -{d} 股,但缺成交价,减仓价约束未设置"
            else:
                states[s] = replace(it.state, trim_price=price, trim_date=iso)
                out[s] = f"减仓提交 -{d} 股,减仓价={price:.2f} @ {iso}"
        elif it.kind == "exit" and a == 0 and b > 0:
            states.pop(s, None)      # 跌破清仓线是真止损,不留下价格约束
            out[s] = f"清仓提交 {b} 股,状态已忘记"
        else:
            out[s] = (f"未足量成交({d} 股 / 计划 {want} 股),状态不推进,"
                      f"下一轮自动重试")
    return out


# ==================== 状态持久化 ====================

def save_states(path: str, states: dict) -> None:
    """实盘状态落盘。先写临时文件再替换,进程被杀不会留下半个 JSON。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {str(s): st.to_dict() for s, st in (states or {}).items()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_states(path: str) -> dict:
    """读回 NameState。文件缺失/损坏一律当"还没有状态",不阻断交易。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(s): NameState.from_dict(v) for s, v in (raw or {}).items()}
    except Exception as e:                        # noqa: BLE001
        logger.error(f"策略状态文件 {path} 读取失败({e}),按空状态继续")
        return {}


# ==================== 小工具 ====================

def _steps(moved: float, gate: float, cap: int) -> int:
    """moved 里含几个完整步长。+1e-9 是防止浮点误差吃掉一整档。"""
    return min(int(math.floor(moved / gate + 1e-9)), cap)


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _as_date(d) -> _date | None:
    if d is None:
        return None
    if hasattr(d, "date"):                       # datetime / pd.Timestamp
        try:
            return d.date()
        except (ValueError, OSError):
            return None
    if isinstance(d, _date):
        return d
    try:
        return _date.fromisoformat(str(d)[:10])
    except ValueError:
        return None


def _age_days(iso: str | None, today: _date | None) -> int | None:
    if iso is None or today is None:
        return None
    try:
        return (today - _date.fromisoformat(str(iso)[:10])).days
    except ValueError:
        return None
