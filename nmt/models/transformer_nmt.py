# nmt/models/transformer_nmt.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import build_norm 
from .relposition import RelativePositionBias 


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    pos_emb: str = "absolute"   # absolute|relative
    norm: str = "layernorm"     # layernorm|rmsnorm
    max_len: int = 512
    relpos_buckets: int = 32
    relpos_max_dist: int = 128


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_relpos: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.use_relpos = use_relpos

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,H,T,dk]
        B, T, D = x.size()
        x = x.view(B, T, self.n_heads, self.dk).transpose(1, 2)
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # [B,H,T,dk] -> [B,T,D]
        B, H, T, dk = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * dk)

    def forward(
        self,
        q: torch.Tensor,              # [B,Tq,D]
        k: torch.Tensor,              # [B,Tk,D]
        v: torch.Tensor,              # [B,Tk,D]
        key_pad_mask: Optional[torch.Tensor],   # [B,Tk] True for PAD
        attn_mask: Optional[torch.Tensor],      # [Tq,Tk] True for masked (causal)
        relpos_bias: Optional[torch.Tensor],    # [1,H,Tq,Tk]
    ) -> torch.Tensor:
        Q = self._split(self.Wq(q))
        K = self._split(self.Wk(k))
        V = self._split(self.Wv(v))

        # scores: [B,H,Tq,Tk]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.dk ** 0.5)

        if relpos_bias is not None:
            scores = scores + relpos_bias

        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask[:, None, None, :], float("-inf"))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)  # [B,H,Tq,dk]
        out = self.Wo(self._merge(out))
        return out


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        use_relpos = (cfg.pos_emb == "relative")
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, use_relpos)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = build_norm(cfg.norm, cfg.d_model)
        self.norm2 = build_norm(cfg.norm, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_pad_mask: torch.Tensor, relpos_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-norm style
        h = self.norm1(x)
        h = self.self_attn(h, h, h, key_pad_mask=src_pad_mask, attn_mask=None, relpos_bias=relpos_bias)
        x = x + self.drop(h)

        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        x = x + self.drop(h2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        use_relpos = (cfg.pos_emb == "relative")
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, use_relpos)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, use_relpos=False)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)

        self.norm1 = build_norm(cfg.norm, cfg.d_model)
        self.norm2 = build_norm(cfg.norm, cfg.d_model)
        self.norm3 = build_norm(cfg.norm, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        src_pad_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        relpos_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, h, h, key_pad_mask=tgt_pad_mask, attn_mask=causal_mask, relpos_bias=relpos_bias)
        x = x + self.drop(h)

        h2 = self.norm2(x)
        h2 = self.cross_attn(h2, memory, memory, key_pad_mask=src_pad_mask, attn_mask=None, relpos_bias=None)
        x = x + self.drop(h2)

        h3 = self.norm3(x)
        h3 = self.ffn(h3)
        x = x + self.drop(h3)
        return x


class TransformerNMT(nn.Module):
    def __init__(self, cfg: TransformerConfig, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=pad_id)

        self.pos_emb = None
        if cfg.pos_emb == "absolute":
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.enc_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = build_norm(cfg.norm, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)

        self.drop = nn.Dropout(cfg.dropout)

        self.relpos = None
        if cfg.pos_emb == "relative":
            self.relpos = RelativePositionBias(cfg.n_heads, cfg.relpos_buckets, cfg.relpos_max_dist)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_emb is None:
            return x
        B, T, D = x.size()
        pos = torch.arange(T, device=x.device).clamp(max=self.cfg.max_len - 1)
        return x + self.pos_emb(pos)[None, :, :]

    def encode(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_pad_mask = src_ids.eq(self.pad_id)  # [B,S]
        x = self.drop(self._add_pos(self.src_emb(src_ids)))

        rel_bias = None
        if self.relpos is not None:
            rel_bias = self.relpos(q_len=x.size(1), k_len=x.size(1), device=x.device)

        for layer in self.enc_layers:
            x = layer(x, src_pad_mask, rel_bias)
        return x, src_pad_mask

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor) -> torch.Tensor:
        """
        tgt_in_ids: [B,T] (typically tgt[:, :-1])
        returns logits: [B,T,V]
        """
        memory, src_pad_mask = self.encode(src_ids)

        tgt_pad_mask = tgt_in_ids.eq(self.pad_id)
        x = self.drop(self._add_pos(self.tgt_emb(tgt_in_ids)))

        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        rel_bias = None
        if self.relpos is not None:
            rel_bias = self.relpos(q_len=T, k_len=T, device=x.device)

        for layer in self.dec_layers:
            x = layer(x, memory, tgt_pad_mask, src_pad_mask, causal_mask, rel_bias)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, max_len: int = 128) -> torch.Tensor:
        self.eval()
        memory, src_pad_mask = self.encode(src_ids)

        B = src_ids.size(0)
        out = torch.full((B, 1), self.bos_id, dtype=torch.long, device=src_ids.device)

        for _ in range(max_len):
            logits = self.forward(src_ids, out)              # [B,T,V]
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B,1]
            out = torch.cat([out, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.eos_id).all():
                break

        return out[:, 1:]  # drop BOS
