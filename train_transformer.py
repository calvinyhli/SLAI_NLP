# train_transformer.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nmt.data import JsonlIdsDataset, collate_ids
from nmt.utils import set_seed, safe_exp
from nmt.models.transformer_nmt import TransformerNMT, TransformerConfig


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    ap = argparse.ArgumentParser()

    # unified with RNN scripts
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_tgt_len", type=int, default=256)

    # transformer-specific
    ap.add_argument("--pos_emb", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=512)

    # optional quick decode preview
    ap.add_argument("--val_decode", choices=["none", "greedy"], default="none")
    ap.add_argument("--decode_max_len", type=int, default=64)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def train_one_epoch(model, loader, opt, criterion, device, max_tgt_len, grad_clip):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        src = batch.src_ids.to(device)
        tgt = batch.tgt_ids.to(device)

        # teacher forcing
        tgt_in = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        if tgt_in.size(1) > max_tgt_len:
            tgt_in = tgt_in[:, :max_tgt_len]
            tgt_y = tgt_y[:, :max_tgt_len]

        opt.zero_grad()
        logits = model(src, tgt_in)  # [B,T,V]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()

        with torch.no_grad():
            non_pad = (tgt_y != model.pad_id).sum().item()
        total_loss += loss.item() * max(1, non_pad)
        total_tokens += max(1, non_pad)

    avg = total_loss / max(1, total_tokens)
    return {"loss": avg, "ppl": safe_exp(avg)}


@torch.no_grad()
def eval_loss(model, loader, criterion, device, max_tgt_len):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        src = batch.src_ids.to(device)
        tgt = batch.tgt_ids.to(device)
        tgt_in = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        if tgt_in.size(1) > max_tgt_len:
            tgt_in = tgt_in[:, :max_tgt_len]
            tgt_y = tgt_y[:, :max_tgt_len]

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))

        non_pad = (tgt_y != model.pad_id).sum().item()
        total_loss += loss.item() * max(1, non_pad)
        total_tokens += max(1, non_pad)

    avg = total_loss / max(1, total_tokens)
    return {"loss": avg, "ppl": safe_exp(avg)}


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab_zh = load_vocab(args.vocab_zh)
    vocab_en = load_vocab(args.vocab_en)

    pad_id = vocab_zh.get("<pad>", 0)
    bos_id = vocab_en.get("<bos>", 1)
    eos_id = vocab_en.get("<eos>", 2)

    train_ds = JsonlIdsDataset(str(data_dir / "train.ids.jsonl"))
    valid_ds = JsonlIdsDataset(str(data_dir / "valid.ids.jsonl"))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_ids(b, pad_id=pad_id),
        num_workers=2, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_ids(b, pad_id=pad_id),
        num_workers=2, pin_memory=True
    )

    cfg = TransformerConfig(
        src_vocab_size=len(vocab_zh),
        tgt_vocab_size=len(vocab_en),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pos_emb=args.pos_emb,
        norm=args.norm,
        max_len=args.max_len,
    )

    device = torch.device(args.device)
    model = TransformerNMT(cfg, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, criterion, device, args.max_tgt_len, args.grad_clip)
        va = eval_loss(model, valid_loader, criterion, device, args.max_tgt_len)

        print(f"[epoch {ep}] train loss={tr['loss']:.4f} ppl={tr['ppl']:.2f} | valid loss={va['loss']:.4f} ppl={va['ppl']:.2f}")

        if args.val_decode == "greedy":
            b = next(iter(valid_loader))
            src = b.src_ids.to(device)[:2]
            pred = model.greedy_decode(src, max_len=args.decode_max_len)
            print("Greedy sample ids:", pred[0].tolist()[:40])

        if va["loss"] < best:
            best = va["loss"]
            ckpt = {
                "cfg": cfg.__dict__,
                "model": model.state_dict(),
                "pad_id": pad_id,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "vocab_zh": args.vocab_zh,
                "vocab_en": args.vocab_en,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "best.pt")
            print("  saved best.pt")

    print("Done. Best valid loss:", best)


if __name__ == "__main__":
    main()
