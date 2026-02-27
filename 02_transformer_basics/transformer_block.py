"""From-scratch self-attention and transformer encoder block in PyTorch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """Scaled dot-product self-attention.

    Q = XW_Q, K = XW_K, V = XW_V
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.W_key   = nn.Linear(embed_dim, embed_dim)
        self.W_value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """x: (batch, seq_len, embed_dim) -> same shape."""
        Q = self.W_query(x)                           # (B, T, D)
        K = self.W_key(x)                             # (B, T, D)
        V = self.W_value(x)                           # (B, T, D)

        scale = math.sqrt(self.embed_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, T, T)

        attention_weights = F.softmax(scores, dim=-1)            # (B, T, T)
        output = torch.matmul(attention_weights, V)              # (B, T, D)
        return output


class TransformerBlock(nn.Module):
    """Single transformer encoder block: attention + FFN + residual + LayerNorm."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len, embed_dim) -> same shape."""
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH, SEQ_LEN, EMBED_DIM = 2, 10, 64

    print("Testing SimpleSelfAttention...")
    attn = SimpleSelfAttention(embed_dim=EMBED_DIM)
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    out = attn(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"  Input  shape: {x.shape}")
    print(f"  Output shape: {out.shape}")

    print("\nTesting TransformerBlock...")
    block = TransformerBlock(embed_dim=EMBED_DIM, num_heads=4, ff_dim=128, dropout=0.1)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"  Input  shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
