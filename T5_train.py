# scripts/sweep_t5_train.py
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


def run_name(cfg: Dict[str, Any]) -> str:
    return f"model={cfg['model_name']}__bs={cfg['batch_size']}__lr={cfg['lr']}__ms={cfg['max_src_len']}__mt={cfg['max_tgt_len']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="scripts/finetune_t5.py")
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--save_root", required=True)

    ap.add_argument("--models", nargs="+", default=["t5-small"])
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[16])
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_src_lens", nargs="+", type=int, default=[128])
    ap.add_argument("--max_tgt_lens", nargs="+", type=int, default=[128])

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.models, args.batch_sizes, args.lrs, args.max_src_lens, args.max_tgt_lens))
    for model_name, bs, lr, ms, mt in grid:
        cfg = {
            "model_name": model_name,
            "batch_size": bs,
            "lr": lr,
            "max_src_len": ms,
            "max_tgt_len": mt,
        }
        rd = save_root / run_name(cfg)
        rd.mkdir(parents=True, exist_ok=True)

        # HF trainer saves checkpoints; we mark completion by presence of "best" folder
        done_flag = rd / "best"
        if args.skip_existing and done_flag.exists():
            print(f"[SKIP] {rd.name} (best exists)")
            continue

        with (rd / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump({"grid": cfg, "epochs": args.epochs, "seed": args.seed}, f, ensure_ascii=False, indent=2)

        cmd = [
            "python", args.train_script,
            "--clean_dir", args.clean_dir,
            "--model_name", model_name,
            "--save_dir", str(rd),
            "--batch_size", str(bs),
            "--lr", str(lr),
            "--epochs", str(args.epochs),
            "--max_src_len", str(ms),
            "--max_tgt_len", str(mt),
            "--seed", str(args.seed),
        ]
        if args.fp16:
            cmd.append("--fp16")

        rc = run(cmd, args.dry_run)
        if rc != 0:
            print(f"[FAIL] {rd.name} rc={rc}")
        else:
            print(f"[DONE] {rd.name}")

    print("All T5 sweeps finished.")


if __name__ == "__main__":
    main()
