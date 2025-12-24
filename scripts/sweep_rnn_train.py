# scripts/sweep_rnn_train.py
import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List


def make_run_name(cfg: Dict[str, Any]) -> str:
    # example: rnn=gru__attn=dot__tf=1.0
    return f"rnn={cfg['rnn_type']}__attn={cfg['attn']}__tf={cfg['tf_ratio']}"


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="train.py", help="Path to RNN train script (default: train.py).")
    ap.add_argument("--data_dir", required=True, help="word_tok dir containing train.ids.jsonl/valid.ids.jsonl/test.ids.jsonl")
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)
    ap.add_argument("--save_root", required=True, help="Root dir to save all runs, e.g., runs/rnn_sweep")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_tgt_len", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # sweep dimensions
    ap.add_argument("--rnn_types", nargs="+", default=["gru", "lstm"])
    ap.add_argument("--attns", nargs="+", default=["dot", "general", "additive"])
    ap.add_argument("--tf_ratios", nargs="+", type=float, default=[1.0, 0.0])  # teacher forcing vs free running

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if best.pt exists.")
    args = ap.parse_args()

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.rnn_types, args.attns, args.tf_ratios))

    for rnn_type, attn, tf_ratio in grid:
        cfg = {
            "rnn_type": rnn_type,
            "attn": attn,
            "tf_ratio": tf_ratio,
        }
        run_name = make_run_name(cfg)
        run_dir = save_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = run_dir / "best.pt"
        if args.skip_existing and ckpt_path.exists():
            print(f"[SKIP] {run_name} (best.pt exists)")
            continue

        # save run config snapshot
        run_cfg = {
            "grid": cfg,
            "train_args": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "max_tgt_len": args.max_tgt_len,
                "grad_clip": args.grad_clip,
                "device": args.device,
                "seed": args.seed,
            },
            "paths": {
                "data_dir": args.data_dir,
                "vocab_zh": args.vocab_zh,
                "vocab_en": args.vocab_en,
            }
        }
        with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)

        cmd = [
            "python", args.train_script,
            "--data_dir", args.data_dir,
            "--vocab_zh", args.vocab_zh,
            "--vocab_en", args.vocab_en,
            "--save_dir", str(run_dir),
            "--rnn_type", rnn_type,
            "--attn", attn,
            "--tf_ratio", str(tf_ratio),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--emb_dim", str(args.emb_dim),
            "--hidden_dim", str(args.hidden_dim),
            "--dropout", str(args.dropout),
            "--max_tgt_len", str(args.max_tgt_len),
            "--grad_clip", str(args.grad_clip),
            "--device", args.device,
            "--seed", str(args.seed),
            # optional: quick sanity decode during training
            "--val_decode", "none",
        ]

        rc = run_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[FAIL] {run_name} returncode={rc}")
            # keep going to run others
            continue

        print(f"[DONE] {run_name}")

    print("All sweeps finished.")


if __name__ == "__main__":
    main()
