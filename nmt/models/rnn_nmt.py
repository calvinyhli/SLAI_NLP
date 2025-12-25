# nmt/models/rnn_nmt.py
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import Attention


@dataclass
class NMTConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    emb_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2              # fixed to 2 as required
    rnn_type: str = "gru"            # gru|lstm
    dropout: float = 0.2
    attn_type: str = "dot"           # dot|general|additive
    attn_dim: int = 256              # used by additive


def _make_rnn(rnn_type: str, input_size: int, hidden_size: int, num_layers: int, dropout: float):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                      dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
    if rnn_type == "lstm":
        return nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
    raise ValueError(f"Unknown rnn_type: {rnn_type}")


HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # GRU: h ; LSTM: (h,c)


class Encoder(nn.Module):
    def __init__(self, cfg: NMTConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.emb = nn.Embedding(cfg.src_vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self.rnn = _make_rnn(cfg.rnn_type, cfg.emb_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor) -> Tuple[torch.Tensor, HiddenState]:
        # src_ids: [B,S]
        emb = self.emb(src_ids)  # [B,S,E]
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h = self.rnn(packed)
        # out, _ = pad_packed_sequence(out_packed, batch_first=True)  # [B,S,H]
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=src_ids.size(1))  # [B,S,H]

        return out, h


class Decoder(nn.Module):
    def __init__(self, cfg: NMTConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.emb = nn.Embedding(cfg.tgt_vocab_size, cfg.emb_dim, padding_idx=pad_id)

        # Decoder RNN input includes embedding + context
        self.rnn = _make_rnn(cfg.rnn_type, cfg.emb_dim + cfg.hidden_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)

        self.attn = Attention(cfg.attn_type, query_dim=cfg.hidden_dim, key_dim=cfg.hidden_dim, attn_dim=cfg.attn_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim + cfg.hidden_dim, cfg.tgt_vocab_size)

    def forward_step(
        self,
        prev_token: torch.Tensor,          # [B]
        prev_state: HiddenState,           # GRU: [L,B,H] ; LSTM: (h,c)
        enc_out: torch.Tensor,             # [B,S,H]
        enc_pad_mask: torch.Tensor,        # [B,S] True where PAD
    ) -> Tuple[torch.Tensor, HiddenState, torch.Tensor]:
        """
        One decoding step.
        Returns:
          logits: [B, V]
          new_state
          attn_weights: [B, S]
        """
        emb = self.emb(prev_token).unsqueeze(1)  # [B,1,E]

        # query for attention: use top-layer hidden from prev_state
        if self.cfg.rnn_type.lower() == "lstm":
            h, c = prev_state
            query = h[-1]  # [B,H]
        else:
            h = prev_state
            query = h[-1]  # [B,H]

        context, attn = self.attn(query, enc_out, enc_pad_mask)           # [B,H], [B,S]
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)           # [B,1,E+H]

        rnn_out, new_state = self.rnn(rnn_in, prev_state)                 # rnn_out: [B,1,H]
        dec_out = rnn_out.squeeze(1)                                      # [B,H]

        # combine decoder output + context -> vocab logits
        logits = self.out_proj(torch.cat([dec_out, context], dim=-1))     # [B,V]
        return logits, new_state, attn


class RNNNMT(nn.Module):
    def __init__(self, cfg: NMTConfig, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.encoder = Encoder(cfg, pad_id)
        self.decoder = Decoder(cfg, pad_id)

        # optional bridge (kept simple: assumes same hidden_dim and same layers)
        # If you later change dims, add Linear mapping here.

    def encode(self, src_ids: torch.Tensor, src_lens: torch.Tensor) -> Tuple[torch.Tensor, HiddenState, torch.Tensor]:
        enc_out, enc_state = self.encoder(src_ids, src_lens)
        enc_pad_mask = (src_ids == self.pad_id)
        return enc_out, enc_state, enc_pad_mask
