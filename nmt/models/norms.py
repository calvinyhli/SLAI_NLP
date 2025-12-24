# nmt/models/norms.py
import torch
import torch.nn as nn
from typing import Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


def build_norm(norm_type: str, dim: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm type: {norm_type}")

# nmt/models/relpos.py
import torch
import torch.nn as nn


def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128):
    """
    T5-like bucketing for relative positions.
    relative_position: [q, k] where value = k_pos - q_pos
    returns bucket ids in [0, num_buckets)
    """
    # T5 uses signed buckets: half for <=0, half for >0
    ret = 0
    n = -relative_position
    num_buckets //= 2
    sign = (n < 0).to(torch.long)
    n = n.abs()

    # now n >= 0
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_small = n
    # log buckets
    val_if_large = max_exact + (
        (torch.log(n.float() / max_exact + 1e-6) / torch.log(torch.tensor(max_distance / max_exact))).to(n.device)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    bucket = torch.where(is_small, val_if_small, val_if_large)
    ret = bucket + sign * num_buckets
    return ret


class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias: [num_buckets, num_heads]
    Produces bias: [1, num_heads, q_len, k_len]
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        rel = k_pos - q_pos  # [q,k]
        buckets = _relative_position_bucket(rel, self.num_buckets, self.max_distance)  # [q,k]
        b = self.bias(buckets)  # [q,k,h]
        return b.permute(2, 0, 1).unsqueeze(0)  # [1,h,q,k]
    
class Attention(nn.Module):
    """
    Supports:
      - dot:       score = q^T k
      - general:   score = q^T W k   (multiplicative / Luong general)
      - additive:  score = v^T tanh(Wq q + Wk k) (Bahdanau)
    """

    def __init__(self, attn_type: str, query_dim: int, key_dim: int, attn_dim: int = 256):
        super().__init__()
        self.attn_type = attn_type.lower()
        self.query_dim = query_dim
        self.key_dim = key_dim

        if self.attn_type == "dot":
            if query_dim != key_dim:
                self.q_proj = nn.Linear(query_dim, key_dim, bias=False)
            else:
                self.q_proj = None

        elif self.attn_type in ("general", "multiplicative"):
            self.W = nn.Linear(key_dim, query_dim, bias=False)  # k -> query space

        elif self.attn_type in ("additive", "bahdanau"):
            self.Wq = nn.Linear(query_dim, attn_dim, bias=False)
            self.Wk = nn.Linear(key_dim, attn_dim, bias=False)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

    def forward(
        self,
        query: torch.Tensor,          # [B, Q]
        keys: torch.Tensor,           # [B, S, K]
        key_padding_mask: torch.Tensor # [B, S] True for PAD positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          context: [B, K]
          attn_weights: [B, S]
        """
        B, S, K = keys.size()

        if self.attn_type == "dot":
            q = self.q_proj(query) if self.q_proj is not None else query  # [B, K]
            scores = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1)         # [B, S]

        elif self.attn_type in ("general", "multiplicative"):
            # score = q^T Wk   where W maps key->query_dim, then dot with q
            wk = self.W(keys)                                            # [B, S, Q]
            scores = torch.bmm(wk, query.unsqueeze(-1)).squeeze(-1)       # [B, S]

        else:  # additive
            # score = v^T tanh(Wq q + Wk k)
            wq = self.Wq(query).unsqueeze(1)                              # [B, 1, A]
            wk = self.Wk(keys)                                            # [B, S, A]
            scores = self.v(torch.tanh(wq + wk)).squeeze(-1)              # [B, S]

        # mask PADs
        scores = scores.masked_fill(key_padding_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                              # [B, S]
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)           # [B, K]
        return context, attn


