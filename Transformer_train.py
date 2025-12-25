# scripts/sweep_transformer_train.py
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run(cmd: List[str], dry_run: bool) -> int:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def make_run_name(cfg: Dict[str, Any]) -> str:
    # concise & readable
    return (
        f"pos={cfg['pos_emb']}__norm={cfg['norm']}__"
        f"dm={cfg['d_model']}__L={cfg['n_layers']}__H={cfg['n_heads']}__ff={cfg['d_ff']}__"
        f"bs={cfg['batch_size']}__lr={cfg['lr']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="scripts/train_transformer.py")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)
    ap.add_argument("--save_root", required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_tgt_len", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    # sweep dimensions (can override from CLI)
    ap.add_argument("--pos_embs", nargs="+", default=["absolute", "relative"])
    ap.add_argument("--norms", nargs="+", default=["layernorm", "rmsnorm"])

    # model scale presets (you can pass multiple)
    # each item formatted as dm,nH,nL,dff e.g. 256,4,4,1024
    ap.add_argument(
        "--scales",
        nargs="+",
        default=["256,4,4,1024"],  # baseline
        help="List of scales: d_model,n_heads,n_layers,d_ff"
    )

    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[64])
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4])

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # parse scales
    scales = []
    for s in args.scales:
        dm, nh, nl, dff = [int(x) for x in s.split(",")]
        scales.append({"d_model": dm, "n_heads": nh, "n_layers": nl, "d_ff": dff})

    grid = list(itertools.product(args.pos_embs, args.norms, scales, args.batch_sizes, args.lrs))

    for pos_emb, norm, scale, bs, lr in grid:
        cfg = {
            "pos_emb": pos_emb,
            "norm": norm,
            **scale,
            "batch_size": bs,
            "lr": lr,
        }
        run_name = make_run_name(cfg)
        run_dir = save_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt = run_dir / "best.pt"
        if args.skip_existing and ckpt.exists():
            print(f"[SKIP] {run_name} (best.pt exists)")
            continue

        with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "grid": cfg,
                    "fixed": {
                        "epochs": args.epochs,
                        "dropout": args.dropout,
                        "max_tgt_len": args.max_tgt_len,
                        "grad_clip": args.grad_clip,
                        "weight_decay": args.weight_decay,
                        "device": args.device,
                        "seed": args.seed,
                    },
                    "paths": {
                        "data_dir": args.data_dir,
                        "vocab_zh": args.vocab_zh,
                        "vocab_en": args.vocab_en,
                    },
                },
                f, ensure_ascii=False, indent=2
            )

        cmd = [
            "python", args.train_script,
            "--data_dir", args.data_dir,
            "--vocab_zh", args.vocab_zh,
            "--vocab_en", args.vocab_en,
            "--save_dir", str(run_dir),

            "--pos_emb", pos_emb,
            "--norm", norm,
            "--d_model", str(scale["d_model"]),
            "--n_heads", str(scale["n_heads"]),
            "--n_layers", str(scale["n_layers"]),
            "--d_ff", str(scale["d_ff"]),
            "--dropout", str(args.dropout),

            "--batch_size", str(bs),
            "--lr", str(lr),
            "--weight_decay", str(args.weight_decay),
            "--epochs", str(args.epochs),
            "--max_tgt_len", str(args.max_tgt_len),
            "--grad_clip", str(args.grad_clip),

            "--device", args.device,
            "--seed", str(args.seed),

            "--val_decode", "none",
        ]

        rc = run(cmd, args.dry_run)
        if rc != 0:
            print(f"[FAIL] {run_name} rc={rc}")
        else:
            print(f"[DONE] {run_name}")

    print("All scratch-transformer training sweeps finished.")


if __name__ == "__main__":
    main()
