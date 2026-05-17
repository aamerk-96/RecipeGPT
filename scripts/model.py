from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout: Optional[nn.Dropout] = None,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    q, k, v expected shape: (batch_size, num_heads, seq_len, head_dim)
    mask expected to be broadcastable to attention score shape.
    """
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(attention_scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # (B, T, C) -> (B, H, T, D)
        q = q.reshape(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention_output, attention_weights = scaled_dot_product_attention(
            q, k, v, self.dropout, mask
        )

        # (B, H, T, D) -> (B, T, C)
        attention_output = (
            attention_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        )
        output = self.wo(attention_output)
        return output, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        residual = x
        x = self.norm1(x)

        attention_output, attention_weights = self.attention(x, mask)
        x = residual + self.dropout(attention_output)

        residual = x
        x = self.norm2(x)

        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        return x, attention_weights


class RecipeGPT(nn.Module):

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        # x shape: (batch_size, seq_len)
        token_embeddings = self.token_embedding(x)  # (B, T, C)

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, T)
        positional_embeddings = self.positional_embedding(positions)  # (1, T, C)

        x = token_embeddings + positional_embeddings
        x = self.dropout(x)

        attention_weights_store: List[Tensor] = []
        for block in self.transformer_blocks:
            x, attention_weights = block(x, mask)
            attention_weights_store.append(attention_weights)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits, attention_weights_store

    @torch.no_grad()
    def generate(self, x: Tensor, max_new_tokens: int = 256, temp: float = 1.0) -> Tensor:
        for _ in range(max_new_tokens):
            if x.shape[1] > self.block_size:
                x = x[:, -self.block_size:]

            logits, _ = self.forward(x)
            logits = logits / temp

            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)  # (B, 1)

            x = torch.cat((x, next_tokens), dim=1)

        return x


def build_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Returns lower-triangular mask of shape (1, 1, seq_len, seq_len),
    broadcastable to attention scores.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
