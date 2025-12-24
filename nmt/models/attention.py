# nmt/models/attention.py
import torch
import torch.nn as nn
from typing import Tuple


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
