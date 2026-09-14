"""
Transformer 编码器模型 — 因子序列 → 前向收益预测。

每个样本:最近 seq_len 个交易日的 n_features 维因子序列 (seq_len, n_features)
    → 输入投影 → + 可学习位置嵌入 → [CLS] 拼接
    → TransformerEncoder(pre-LN, gelu)
    → CLS 隐状态 → LayerNorm → MLP head → 标量预测(前向收益)

参数量约 60 万(d_model=128, 4 层),权重仅 2.4MB,8GB 显存轻松容纳。

可选扩展(预留,不进 v1):per-factor 嵌入、rotary 位置编码、
multi-horizon 多任务 head、对比学习。
"""

import torch
import torch.nn as nn


class FactorTransformer(nn.Module):
    """因子序列 Transformer。

    组成:
        input_proj  : Linear(n_features, d_model)      # 因子 → d_model
        cls_token   : Parameter(1, 1, d_model)          # 可学习 CLS 注意力池化
        pos_emb     : Embedding(max_seq_len, d_model)   # 可学习位置嵌入
        encoder     : TransformerEncoder(pre-LN)
        head_norm   : LayerNorm(d_model)
        head        : MLP(d_model → d_model → 1, 末层小方差初始化)
    """

    def __init__(self, n_features: int = 25, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 4, dim_ff: int = 256,
                 dropout: float = 0.1, max_seq_len: int = 64,
                 activation: str = "gelu"):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation=activation, batch_first=True,
            norm_first=True,  # pre-LN,训练更稳
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model))

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """CLS 与位置嵌入用 N(0, 0.02);head 末层小方差(回归输出)。"""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, seq_len, n_features) float32 因子序列
               (位置 0 最远,末尾最近)

        Returns:
            (B, 1) 预测前向收益(或分类 logits)
        """
        B, L, _ = x.shape
        assert L + 1 <= self.max_seq_len, \
            f"seq_len={L} 超出位置嵌入容量 {self.max_seq_len}"

        h = self.input_proj(x)  # (B, L, d_model)
        pos = self.pos_emb(torch.arange(L, device=x.device)).unsqueeze(0)
        h = h + pos

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)  # (B, L+1, d_model)

        h = self.encoder(h)  # (B, L+1, d_model)
        h = self.head_norm(h[:, 0])  # CLS 位置
        return self.head(h)  # (B, 1)


def build_model(model_cfg: dict, n_features: int = 25) -> FactorTransformer:
    """从 config["model"] 构建模型。

    Args:
        model_cfg: config.yaml 的 model 段
        n_features: 因子数量(运行时从因子面板提取,覆盖 arch 里的值)

    Returns:
        FactorTransformer
    """
    arch = dict(model_cfg["arch"])
    arch["n_features"] = n_features
    return FactorTransformer(**arch)


def count_parameters(model: nn.Module) -> int:
    """可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
