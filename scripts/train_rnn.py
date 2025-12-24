# train.py
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nmt.data import JsonlIdsDataset, collate_ids
from nmt.models.rnn_nmt import RNNNMT, NMTConfig
from nmt.models.decoding import greedy_decode, beam_search_decode_one


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: RNNNMT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    tf_ratio: float,     # 1.0 teacher forcing ; 0.0 free running
    max_tgt_len: int = 256,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src_ids = batch.src_ids.to(device)
        src_lens = batch.src_lens.to(device)
        tgt_ids = batch.tgt_ids.to(device)

        # decoder inputs and targets
        # tgt_ids includes BOS/EOS already.
        dec_in = tgt_ids[:, :-1]   # [B, T-1]
        dec_tg = tgt_ids[:, 1:]    # [B, T-1]

        # encode
        enc_out, state, enc_mask = model.encode(src_ids, src_lens)

        B, Tm1 = dec_in.size()
        Tm1 = min(Tm1, max_tgt_len)
        optimizer.zero_grad()

        # step decoding with scheduled input selection
        cur = dec_in[:, 0]  # BOS in most cases
        logits_steps = []

        for t in range(Tm1):
            logits, state, _ = model.decoder.forward_step(cur, state, enc_out, enc_mask)
            logits_steps.append(logits)  # [B,V]

            # choose next input
            if t + 1 < Tm1:
                if tf_ratio >= 1.0:
                    cur = dec_in[:, t + 1]
                elif tf_ratio <= 0.0:
                    cur = torch.argmax(logits, dim=-1)
                else:
                    # per-sample mixing
                    use_teacher = (torch.rand(B, device=device) < tf_ratio)
                    pred = torch.argmax(logits, dim=-1)
                    gold = dec_in[:, t + 1]
                    cur = torch.where(use_teacher, gold, pred)

        logits_all = torch.stack(logits_steps, dim=1)          # [B,T,V]
        loss = criterion(logits_all.reshape(-1, logits_all.size(-1)), dec_tg[:, :Tm1].reshape(-1))

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # token accounting (ignore PAD via criterion)
        with torch.no_grad():
            non_pad = (dec_tg[:, :Tm1] != model.pad_id).sum().item()
        total_loss += loss.item() * max(1, non_pad)
        total_tokens += max(1, non_pad)

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return {"loss": avg_loss, "ppl": ppl}


@torch.no_grad()
def eval_loss(model: RNNNMT, loader: DataLoader, criterion: nn.Module, device: torch.device, max_tgt_len: int = 256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src_ids = batch.src_ids.to(device)
        src_lens = batch.src_lens.to(device)
        tgt_ids = batch.tgt_ids.to(device)

        dec_in = tgt_ids[:, :-1]
        dec_tg = tgt_ids[:, 1:]

        enc_out, state, enc_mask = model.encode(src_ids, src_lens)

        B, Tm1 = dec_in.size()
        Tm1 = min(Tm1, max_tgt_len)

        cur = dec_in[:, 0]
        logits_steps = []
        for t in range(Tm1):
            logits, state, _ = model.decoder.forward_step(cur, state, enc_out, enc_mask)
            logits_steps.append(logits)
            if t + 1 < Tm1:
                cur = dec_in[:, t + 1]   # evaluation loss uses teacher forcing for stability

        logits_all = torch.stack(logits_steps, dim=1)
        loss = criterion(logits_all.reshape(-1, logits_all.size(-1)), dec_tg[:, :Tm1].reshape(-1))

        non_pad = (dec_tg[:, :Tm1] != model.pad_id).sum().item()
        total_loss += loss.item() * max(1, non_pad)
        total_tokens += max(1, non_pad)

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return {"loss": avg_loss, "ppl": ppl}


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True, help=".../word_tok directory containing train.ids.jsonl etc.")
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)

    # model
    ap.add_argument("--rnn_type", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)

    # attention
    ap.add_argument("--attn", choices=["dot", "general", "additive"], default="dot")
    ap.add_argument("--attn_dim", type=int, default=256)

    # training policy
    ap.add_argument("--tf_ratio", type=float, default=1.0, help="1.0 teacher forcing; 0.0 free running; (0,1) scheduled sampling")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_tgt_len", type=int, default=256)

    # decoding policy for (optional) qualitative validation
    ap.add_argument("--val_decode", choices=["none", "greedy", "beam"], default="none")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--decode_max_len", type=int, default=128)

    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return ap.parse_args()


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

    cfg = NMTConfig(
        src_vocab_size=len(vocab_zh),
        tgt_vocab_size=len(vocab_en),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        attn_type=args.attn,
        attn_dim=args.attn_dim,
    )

    device = torch.device(args.device)
    model = RNNNMT(cfg, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            tf_ratio=args.tf_ratio, max_tgt_len=args.max_tgt_len, grad_clip=args.grad_clip
        )
        va = eval_loss(model, valid_loader, criterion, device, max_tgt_len=args.max_tgt_len)

        print(f"[epoch {ep}] train loss={tr['loss']:.4f} ppl={tr['ppl']:.2f} | valid loss={va['loss']:.4f} ppl={va['ppl']:.2f}")

        # optional decode preview
        if args.val_decode != "none":
            batch = next(iter(valid_loader))
            src_ids = batch.src_ids.to(device)
            src_lens = batch.src_lens.to(device)

            if args.val_decode == "greedy":
                pred = greedy_decode(model, src_ids[:4], src_lens[:4], max_len=args.decode_max_len)
                print("Greedy sample ids:", pred[0].tolist()[:40])

            elif args.val_decode == "beam":
                one = beam_search_decode_one(
                    model, src_ids[:1], src_lens[:1],
                    beam_size=args.beam_size, max_len=args.decode_max_len
                )
                print("Beam sample ids:", one[:40])

        # save best
        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt = {
                "cfg": cfg.__dict__,
                "model": model.state_dict(),
                "vocab_zh": args.vocab_zh,
                "vocab_en": args.vocab_en,
                "pad_id": pad_id,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "best.pt")
            print("  saved best.pt")

    print("Done. Best valid loss:", best_val)


if __name__ == "__main__":
    main()
