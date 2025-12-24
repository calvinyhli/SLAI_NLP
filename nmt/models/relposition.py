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
