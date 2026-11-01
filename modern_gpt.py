"""The Tiny GPT with each modern feature behind a switch, so each can be tested alone.

With every switch off, this model is the same as the one in Project 4.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tinygpt.model import FeedForward, GPTConfig


@dataclass(frozen=True)
class ModernConfig(GPTConfig):
    use_flash: bool = False  # PyTorch's optimized attention, with the same math
    use_rmsnorm: bool = False  # simpler normalization than LayerNorm
    use_rope: bool = False  # rotary positions instead of a learned position table
    n_kv_head: int | None = None  # grouped-query attention: fewer key/value heads


class RMSNorm(nn.Module):
    """Scale by root mean square. Unlike LayerNorm, this doesn't subtract the mean."""

    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return self.weight * x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()


class Rotary:
    """Rotary position embeddings (RoPE).

    Each pair of dimensions in a key or query is rotated by an angle that depends on
    its position. Their dot product then depends on the distance between the two
    positions, which gives attention a sense of relative position with no parameters.
    """

    def __init__(self, head_size, max_length, device=None, base=10000.0):
        if head_size % 2:
            raise ValueError("RoPE needs an even head size")
        frequencies = 1.0 / base ** (torch.arange(0, head_size, 2, device=device).float() / head_size)
        angles = torch.outer(torch.arange(max_length, device=device).float(), frequencies)
        self.cos, self.sin = angles.cos(), angles.sin()

    def apply(self, x, offset=0):
        # x has shape (batch, heads, time, head_size).
        length = x.shape[2]
        cos, sin = self.cos[offset : offset + length], self.sin[offset : offset + length]
        first, second = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
        return rotated.flatten(-2)


class ModernAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.head_size = config.n_embd // config.n_head
        self.use_flash = config.use_flash
        self.dropout_p = config.dropout
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        kv_size = self.n_kv_head * self.head_size
        self.key = nn.Linear(config.n_embd, kv_size, bias=False)
        self.value = nn.Linear(config.n_embd, kv_size, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = Rotary(self.head_size, config.block_size) if config.use_rope else None
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def _split(self, tensor, heads):
        batch, time, _ = tensor.shape
        return tensor.view(batch, time, heads, self.head_size).transpose(1, 2)

    def forward(self, x, past=None):
        batch, time, channels = x.shape
        past_length = past[0].shape[2] if past is not None else 0

        q = self._split(self.query(x), self.n_head)
        k = self._split(self.key(x), self.n_kv_head)
        v = self._split(self.value(x), self.n_kv_head)
        if self.rotary is not None:
            q = self.rotary.apply(q, past_length)
            k = self.rotary.apply(k, past_length)
        if past is not None:
            # KV cache: earlier keys and values don't change, so reuse them and only
            # compute the ones for the new tokens.
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)

        # Grouped-query attention: several query heads share one key/value head.
        repeat = self.n_head // self.n_kv_head
        k_all = k.repeat_interleave(repeat, dim=1)
        v_all = v.repeat_interleave(repeat, dim=1)

        total = past_length + time
        mask = self.mask[past_length:total, :total]
        dropout = self.dropout_p if self.training else 0.0
        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, dropout_p=dropout)
        else:
            scores = q @ k_all.transpose(-2, -1) / self.head_size**0.5
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.dropout(F.softmax(scores, dim=-1), dropout)
            out = weights @ v_all
        out = out.transpose(1, 2).reshape(batch, time, channels)
        return self.resid_dropout(self.proj(out)), present


class ModernBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm_1 = norm(config.n_embd)
        self.attention = ModernAttention(config)
        self.norm_2 = norm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x, past=None):
        attended, present = self.attention(self.norm_1(x), past)
        x = x + attended
        return x + self.feed_forward(self.norm_2(x)), present


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = None
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(ModernBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd) if config.use_rmsnorm else nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _run(self, ids, past=None):
        past_length = past[0][0].shape[2] if past is not None else 0
        total = past_length + ids.shape[1]
        if total > self.config.block_size:
            raise ValueError(f"input length {total} exceeds block_size {self.config.block_size}")
        x = self.token_embedding(ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(past_length, total, device=ids.device))
        x = self.dropout(x)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past is None else past[i])
            presents.append(present)
        return self.head(self.norm(x)), presents

    def forward(self, ids, targets=None):
        logits, _ = self._run(ids)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

    def forward_cached(self, ids, past=None):
        """Run only the new tokens in `ids`, reusing earlier keys and values from `past`."""
        return self._run(ids, past)

    @staticmethod
    def _sample(logits, temperature, generator):
        probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, 1, generator=generator)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0, generator=None):
        """Generate by re-running the whole context for every new token."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.config.block_size :])
            ids = torch.cat([ids, self._sample(logits, temperature, generator)], dim=1)
        self.train(was_training)
        return ids

    @torch.no_grad()
    def generate_cached(self, ids, max_new_tokens, temperature=1.0, generator=None):
        """Generate with a KV cache, so each step only processes the newest token."""
        was_training = self.training
        self.eval()
        block_size = self.config.block_size
        past, current = None, ids[:, -block_size:]
        for _ in range(max_new_tokens):
            logits, past = self.forward_cached(current, past)
            next_id = self._sample(logits, temperature, generator)
            ids = torch.cat([ids, next_id], dim=1)
            if past[0][0].shape[2] == block_size:
                # The cache is full, so drop it and rebuild from the most recent
                # window. Positions shift, so cached keys can't simply be trimmed.
                past, current = None, ids[:, -block_size:]
            else:
                current = next_id
        self.train(was_training)
        return ids

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
