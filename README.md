# QuantLab2 — 基于 Transformer 的 A 股深度学习选股模型

> 与 [quantlab](../quantlab)（LightGBM 树模型）同数据、同口径的深度学习版本：
> 用 **Transformer Encoder** 对每只股票的因子时间序列建模，预测前向收益并选股。

## 与 quantlab 的关系

| 维度 | quantlab | quantlab2 |
|------|----------|-----------|
| 模型 | LightGBM 回归/分类 | Transformer Encoder（自注意力） |
| 输入 | 单日截面因子向量（25 维） | 最近 20 个交易日因子序列（20×25） |
| 标签 | 前向 20 日收益 | 前向 20 日收益（口径一致） |
| 数据 | 自己下载/计算 | **复用 quantlab 的因子面板与日线缓存**（路径在 config.yaml 可配） |
| 训练 | Walk-Forward 每 6 个月重训 | Walk-Forward 每 6 个月重训（一致） |
| 回测 | 向量化引擎 | 原样移植，成本参数一致（保证对比公平） |

## 快速开始

```bash
# 0. 先安装 GPU 版 torch（本机驱动 537.53 → 必须 cu121；见 requirements.txt 注释）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. 快速冒烟测试（<3 分钟，验证全管道）
python main.py smoke

# 2. 全量 Walk-Forward 训练（8 折，GPU 约 1 小时；CPU 慢 5-10 倍）
python main.py train

# 3. 回测（只消费样本外预测，杜绝未来函数）
python main.py backtest

# 4. 用最新 checkpoint 生成最新交易日调仓信号
python main.py predict

# 5. 实时交易（模拟盘单次执行 / 盘中轮询，见下方章节）
python main.py live --broker simulate --once --asof 2026-08-14
python main.py live --broker simulate          # 盘中 09:35 调仓 + 15:05 盯市
```

## 实时交易（live）

支持 **可插拔券商接口**：统一 `Broker` 抽象基类，新增券商只需继承实现
`place_order/get_cash/get_positions/get_total_value` 等并在
[live/broker.py](live/broker.py) 的工厂注册一行。

### 三种后端

| 后端 | 说明 | 命令 |
|------|------|------|
| `simulate` | 模拟盘：T+1、整手、涨跌停、佣金/印花税/滑点，状态持久化 | `python main.py live --broker simulate` |
| `qmt` | miniQMT 真实柜台（需 xtquant + 本机 miniQMT 客户端） | `python main.py live --broker qmt --confirm` |
| `none` | 仅导出指令 CSV（人工检查/其他系统消费） | `python main.py live --broker none` |

### 运行模式

- **盘中轮询**（默认）：每 5 秒拉新浪实时行情；到 `09:35` 且今天是调仓日
  （默认月末交易日）→ 调仓；`15:05` 收盘盯市。交易日盘中前台运行。
- **单次执行**（`--once`）：信号 → 指令 → 风控 → 执行 → 盯市一次后退出，
  任何日期可跑（测试/计划任务用）。建议显式 `--asof` 指定信号基准日。

### 信号口径（严格防未来函数）

- 因子**截止上一交易日收盘**（与训练 lag-1 口径一致），盘中实时行情只用于
  参考价/涨跌停过滤/撮合，绝不进入特征。
- 调仓流程：最新 checkpoint → Transformer 推理 → Top-K 等权目标
  → 先卖后买指令 → **阻断式风控**（单股仓位≤20%、总仓位≤95%、持仓≤50、
  涨停不买/跌停不卖/停牌跳过）→ 执行。

### 实盘安全设计

- `--confirm` 才真实下单；缺省只生成指令文件（`live/output/orders_*.csv`）
- QMT 下 FIX_PRICE 限价单（以参考价为限价，滑点保护）
- 模拟盘持仓状态每次成交后落盘（`live/state/simulate_state.json`），
  进程被杀重启无损

### QMT 实盘前置条件

1. 安装 `pip install xtquant`（仅实盘需要）
2. 本机运行 miniQMT 客户端并登录
3. 配置 `config.yaml` 的 `live.qmt.mini_qmt_path`（如 `D:\国金QMT\userdata_mini`）
   与 `live.qmt.account_id`（资金账号）

## 模型架构

每个样本 = 某只股票某日之前 **最近 20 个有因子值的交易日** 的 25 维因子序列：

```
(20, 25) 因子序列
   │  Linear(25→128)
   │  + 可学习位置嵌入(Embedding)
   │  + [CLS] token 拼接
   │  TransformerEncoder × 4 层(pre-LN, 8 头, dim_ff=256, dropout=0.1)
   │  取 CLS 隐状态 → LayerNorm
   │  MLP head(128→128→1, 末层小方差初始化)
   ▼
前向 20 日收益预测(标量)
```

- 参数量约 **60 万**（权重 2.4MB），8GB 显存轻松容纳（batch 1024 激活 <0.5GB）
- 训练：AdamW + warmup 1000 步 + cosine 衰减，梯度裁剪 1.0，AMP（仅 GPU）
- **模型选择指标 = 验证集按日截面 Spearman Rank IC**（比 MSE 更贴合选股目标），patience 10
- 验证集是从训练尾部**切出来**的，与训练集无交集，且与 OOS 窗口之间留了
  `horizon + embargo_days` 个交易日的 purge 缺口。2026-08 那次基线跑的 8 折
  里验证集是训练集的子集，IC≈0.9 恒优于训练集，早停从未触发（60 epoch 全部跑满）——
  该缺陷已修，历史数字不可与新结果混用

## 项目结构

```
quantlab2/
├── main.py                 # CLI 入口: train / backtest / predict / pipeline / smoke
├── config.yaml             # 全部超参(data 路径/模型/训练/回测)
├── requirements.txt        # 依赖清单(torch 单独装)
├── models/
│   ├── transformer.py      # FactorTransformer 模型定义
│   ├── sequence_data.py    # 序列数据集: per-symbol 矩阵 + 位置滑窗 + 标签
│   ├── trainer.py          # 训练循环 / Rank IC / 早停 / checkpoint / Walk-Forward
│   └── predictor.py        # 批量推理 → 截面排名 → Top-K → 信号
├── data/
│   └── loader.py           # 读因子面板 / 日线 / 基准指数
├── backtest/               # 移植自 quantlab(engine/cost/metrics/reporter)
├── utils/                  # 日志 / 交易日历 / 设备与随机种子
├── models/saved/           # checkpoint 输出
└── data/cache/             # OOS 预测 predictions.parquet
```

## 防未来函数规则（冒烟测试逐条断言）

1. **因子已 lag 1**：面板日期 t 的因子信息止于 close[t-1]（quantlab 处理阶段完成）
2. **窗口封闭性**：样本 t 的窗口只含日期 ≤ t 的因子行；停牌缺口用"最近 N 个有因子值的交易日"位置滑窗，不做日历对齐
3. **标签只作为 y**：前向收益绝不进入特征；标签按日截面缩尾只使用当日横截面
4. **Walk-Forward 严格性**：折内三段 `train [.., val_start) | valid [val_start, purge_start) | OOS [ws, we)`
   两两无交集，且训练/验证样本的 20 日标签窗口都够不到 OOS（purge 宽度 =
   `horizon + embargo_days` 个交易日）；尾部不足 `min_oos_days` 个交易日的折不参与。
   由 `tests/test_walk_forward_split.py` 逐条断言（含 `train ∩ valid == ∅`）
5. **回测只消费 OOS**：`backtest` 命令只读 `data/cache/predictions.parquet`（全样本外预测），找不到该文件直接报错，不做全样本回退
6. **无全局统计量**：NaN 填充用 0（面板已截面 zscore，0=截面均值，无偏且零泄漏），不拟合任何训练集统计量做再标准化

## 结果解读

- 训练结束打印每折 `best_rank_ic`（验证集选权重）与 `oos_rank_ic`（样本外），
  以及**按日合并**的全样本 OOS Rank IC + Newey-West(HAC) t 与有效样本量。
  20 日重叠标签下朴素 t 会虚高约 √VR ≈ 3 倍（实测 VR = 9.05，705 天 → N_eff ≈ 78），
  所以显著性只认 HAC t 与 moving-block bootstrap 置信区间
- 逐日 IC 落盘到 `reports/daily_ic.csv`（约 20 KB），显著性结论任何人可独立复算
- **平均 OOS RankIC > 0.02 可用，> 0.05 优秀，< 0 无效**（与 quantlab 口径一致，可直接对比 LightGBM）
- 回测报告输出年化收益/夏普/最大回撤/超额收益/信息比率，与 quantlab 在相同时间窗对比
- 断点续训：训练中断后重跑 `python main.py train` 会自动复用已有 checkpoint
  与逐折 OOS 预测；但缓存旁的 `predictions.parquet.meta.json` 记录了切分口径
  （horizon / val_months / embargo_days / 折宽度），口径一变即作废重训，
  避免新旧口径的折混进同一个均值

## 2026-08 基线结果（999 只股票，2023-01 → 2026-08 样本外）

**OOS Rank IC 对比（同窗口）** — 运行 `python compare_models.py` 可复现：

| 窗口 | LightGBM | Transformer |
|------|---------:|------------:|
| 2023 上半年 | 0.067 | 0.068 |
| 2023 下半年 | 0.076 | 0.074 |
| 2024 上半年 | 0.036 | **0.054** |
| 2024 下半年 | 0.045 | 0.048 |
| 2025 上半年 | 0.071 | 0.065 |
| 2025 下半年 | -0.003 | **0.031** |
| 2026 上半年 | — | 0.050 |
| **全样本** | **0.050** | **0.060** |

**回测**（月度调仓 Top-50 等权，佣金万三/印花税/滑点，2023-01 → 2026-08）：

- 累计收益 **+106.1%**（年化 +23.6%），股票池等权基准 +54.1%
- **超额收益 +52.1%**，夏普 0.65，信息比率 0.54
- 最大回撤 41.2%（小盘股属性，与中证1000 同期波动一致）
- 43 次月度调仓，交易成本合计 15.7 万元（已计入收益）

注意：
1. 上表的 **全样本 0.060** 是各折 IC 的**按折算术平均**，而按日合并是 0.056；
   2026-08 那次跑出的 `平均 OOS RankIC = 0.1141` 更不可用——它多含一个只有
   ~8 个交易日、OOS_IC = 0.5215 的尾部折（现在被 `min_oos_days` 丢弃）。
2. 表中的 Transformer 数字来自**修复前**的切分（验证集是训练集子集、无 purge），
   修复后需重跑才能与 LightGBM 并列比较。
3. "验证集 IC≈0.9 是被 20 日重叠标签放大的"这个解释不成立：同一交易日内的
   截面样本之间没有标签重叠。真实原因是验证集 ⊂ 训练集，模型在背它见过的答案。

## 要求

- Python 3.11+
- PyTorch 2.5.1+cu121（GPU）或 CPU 版
- 需要 quantlab 已生成 `data/cache/factor_panel.parquet` 与 `data/cache/daily/`
