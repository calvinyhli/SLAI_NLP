# nmt/utils.py
import math
import random
from typing import Tuple

import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_pad_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # True where PAD
    return ids.eq(pad_id)


def make_causal_mask(tgt_len: int, device: torch.device) -> torch.Tensor:
    # True where masked (upper triangle)
    return torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)


def safe_exp(x: float) -> float:
    return math.exp(x) if x < 50 else float("inf")
