# nmt/data.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    src_ids: torch.Tensor        # [B, S]
    src_lens: torch.Tensor       # [B]
    tgt_ids: torch.Tensor        # [B, T]
    tgt_lens: torch.Tensor       # [B]


class JsonlIdsDataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        self.items: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "src_ids" in obj and "tgt_ids" in obj:
                    self.items.append(obj)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.items[i]


def _pad_1d(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lens.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lens


def collate_ids(batch: List[Dict[str, Any]], pad_id: int = 0) -> Batch:
    src = [x["src_ids"] for x in batch]
    tgt = [x["tgt_ids"] for x in batch]
    src_ids, src_lens = _pad_1d(src, pad_id)
    tgt_ids, tgt_lens = _pad_1d(tgt, pad_id)
    return Batch(src_ids=src_ids, src_lens=src_lens, tgt_ids=tgt_ids, tgt_lens=tgt_lens)
