# scripts/infer_transformer.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nmt.data import JsonlIdsDataset, collate_ids
from nmt.models.transformer_nmt import TransformerNMT, TransformerConfig


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {i: t for t, i in vocab.items()}


def ids_to_sentence(ids: List[int], id2tok: Dict[int, str], eos_id: int) -> str:
    toks = []
    for i in ids:
        if i == eos_id:
            break
        tok = id2tok.get(i, "<unk>")
        if tok in ("<pad>", "<bos>", "<eos>"):
            continue
        toks.append(tok)
    return " ".join(toks)


@torch.no_grad()
def beam_decode_one(
    model: TransformerNMT,
    src_ids_1: torch.Tensor,   # [1,S]
    beam_size: int,
    max_len: int,
    length_penalty: float = 0.0,
) -> List[int]:
    device = src_ids_1.device
    model.eval()

    # hypothesis: (tokens, score)
    hyps: List[Tuple[List[int], float]] = [([], 0.0)]
    finished: List[Tuple[List[int], float]] = []

    for t in range(max_len):
        new_hyps: List[Tuple[List[int], float]] = []
        for tokens, score in hyps:
            if len(tokens) > 0 and tokens[-1] == model.eos_id:
                finished.append((tokens, score))
                continue

            # build decoder input: BOS + tokens
            dec_in = torch.tensor([[model.bos_id] + tokens], device=device, dtype=torch.long)
            logits = model(src_ids_1, dec_in)      # [1, T, V]
            logp = F.log_softmax(logits[0, -1], dim=-1)  # [V]

            topk_logp, topk_ids = torch.topk(logp, k=beam_size)
            for lp, wid in zip(topk_logp.tolist(), topk_ids.tolist()):
                new_hyps.append((tokens + [wid], score + lp))

        def norm(s: float, L: int) -> float:
            if length_penalty <= 0:
                return s
            return s / ((L + 1) ** length_penalty)

        new_hyps.sort(key=lambda x: norm(x[1], len(x[0])), reverse=True)
        hyps = new_hyps[:beam_size]

        if len(finished) >= beam_size:
            break

    all_cands = finished + hyps
    all_cands.sort(key=lambda x: x[1] / ((len(x[0]) + 1) ** length_penalty) if length_penalty > 0 else x[1], reverse=True)
    best = all_cands[0][0]
    return best


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)

    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_hyp", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vocab_zh = load_vocab(args.vocab_zh)
    vocab_en = load_vocab(args.vocab_en)
    id2tok = invert_vocab(vocab_en)

    pad_id = vocab_zh.get("<pad>", 0)
    bos_id = vocab_en.get("<bos>", 1)
    eos_id = vocab_en.get("<eos>", 2)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = TransformerConfig(**ckpt["cfg"])

    device = torch.device(args.device)
    model = TransformerNMT(cfg, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_ds = JsonlIdsDataset(str(Path(args.data_dir) / "test.ids.jsonl"))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_ids(b, pad_id=pad_id),
        num_workers=2,
        pin_memory=True,
    )

    out_path = Path(args.out_hyp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        if args.decode == "greedy":
            for batch in loader:
                src = batch.src_ids.to(device)
                pred_ids = model.greedy_decode(src, max_len=args.max_len)  # [B,T]
                for row in pred_ids.tolist():
                    fout.write(ids_to_sentence(row, id2tok, eos_id) + "\n")
        else:
            # beam: safe per-sample
            for batch in loader:
                src = batch.src_ids.to(device)
                for i in range(src.size(0)):
                    best = beam_decode_one(model, src[i:i+1], args.beam_size, args.max_len)
                    fout.write(ids_to_sentence(best, id2tok, eos_id) + "\n")

    print("Saved hyp:", str(out_path))


if __name__ == "__main__":
    main()
