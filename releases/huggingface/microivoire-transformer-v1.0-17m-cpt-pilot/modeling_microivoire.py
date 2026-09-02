"""Définition PyTorch autonome du Transformer MicroIvoire 17M."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    model_id: str = "microivoire_transformer_v1.0_17m_cpt_pilot"
    tokenizer_id: str = "ivoireslm_bpe_v0.4"
    corpus_id: str = "ivoireslm_pretraining_mix_v1.0.0_pilot"
    vocab_size: int = 8_192
    block_size: int = 512
    embedding_dim: int = 320
    attention_heads: int = 8
    transformer_layers: int = 12
    feed_forward_dim: int = 832
    dropout: float = 0.10
    batch_size: int = 16
    gradient_accumulation: int = 2
    max_steps: int = 20_800
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    evaluation_interval: int = 250
    checkpoint_interval: int = 500
    log_interval: int = 25
    seed: int = 20260830


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(dtype=values.dtype) * self.weight.to(dtype=values.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, maximum_length: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("la dimension de tête RoPE doit être paire")
        inverse = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(maximum_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer(
            "cosine", embedding.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sine", embedding.sin()[None, None, :, :], persistent=False
        )

    @staticmethod
    def rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply_rotary(self, values: torch.Tensor) -> torch.Tensor:
        length = values.shape[-2]
        cosine = self.cosine[:, :, :length].to(dtype=values.dtype)
        sine = self.sine[:, :, :length].to(dtype=values.dtype)
        return values * cosine + self.rotate_half(values) * sine


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        if config.embedding_dim % config.attention_heads:
            raise ValueError("embedding_dim doit être divisible par attention_heads")
        self.heads = config.attention_heads
        self.head_dim = config.embedding_dim // config.attention_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False)
        self.projection = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=False
        )
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)
        self.dropout = config.dropout

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        query, key, value = self.qkv(values).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        query = self.rope.apply_rotary(query)
        key = self.rope.apply_rotary(key)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.projection(output)


class SwiGLU(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.gate = nn.Linear(config.embedding_dim, config.feed_forward_dim, bias=False)
        self.up = nn.Linear(config.embedding_dim, config.feed_forward_dim, bias=False)
        self.down = nn.Linear(config.feed_forward_dim, config.embedding_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(values)) * self.up(values)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.embedding_dim)
        self.feed_forward = SwiGLU(config)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values))
        return values + self.feed_forward(self.feed_forward_norm(values))


class MicroIvoireTransformer17M(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.transformer_layers)
        )
        self.final_norm = RMSNorm(config.embedding_dim)
        self.language_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.shape[1] > self.config.block_size:
            raise ValueError("séquence plus longue que block_size")
        values = self.dropout(self.token_embedding(tokens))
        for block in self.blocks:
            values = block(values)
        logits = self.language_head(self.final_norm(values))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss
