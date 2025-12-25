# scripts/sweep_transformer_bleu.py
import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run(cmd: List[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_ref_from_clean_jsonl(clean_test_jsonl: Path, out_ref_txt: Path, field: str = "en") -> None:
    out_ref_txt.parent.mkdir(parents=True, exist_ok=True)
    with clean_test_jsonl.open("r", encoding="utf-8") as fin, out_ref_txt.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field not in obj or not isinstance(obj[field], str):
                raise ValueError(f"Missing/invalid field={field} at line {i} in {clean_test_jsonl}")
            fout.write(obj[field].rstrip("\n") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_root", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)

    ap.add_argument("--infer_script", default="scripts/infer_transformer.py")
    ap.add_argument("--bleu_script", default="scripts/evaluate_bleu.py")

    ap.add_argument("--clean_test_jsonl", required=True)
    ap.add_argument("--ref_field", default="en")

    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--tokenize", default="13a")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    save_root = Path(args.save_root)
    ref_txt = save_root / "ref.test.en.txt"
    if not ref_txt.exists():
        export_ref_from_clean_jsonl(Path(args.clean_test_jsonl), ref_txt, field=args.ref_field)

    results: List[Dict[str, Any]] = []

    for run_dir in sorted([p for p in save_root.iterdir() if p.is_dir()]):
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            continue

        hyp_name = f"hyp.{args.decode}.txt" if args.decode == "greedy" else f"hyp.beam{args.beam_size}.txt"
        hyp_path = run_dir / hyp_name
        bleu_json = run_dir / (hyp_name + ".bleu.json")

        if args.skip_existing and bleu_json.exists():
            r = read_json(bleu_json)
            r["run_dir"] = str(run_dir)
            results.append(r)
            print(f"[SKIP] {run_dir.name}")
            continue

        # 1) infer
        infer_cmd = [
            "python", args.infer_script,
            "--ckpt", str(ckpt),
            "--data_dir", args.data_dir,
            "--vocab_zh", args.vocab_zh,
            "--vocab_en", args.vocab_en,
            "--decode", args.decode,
            "--max_len", str(args.max_len),
            "--batch_size", str(args.batch_size),
            "--out_hyp", str(hyp_path),
            "--device", args.device,
        ]
        if args.decode == "beam":
            infer_cmd += ["--beam_size", str(args.beam_size)]
        if run(infer_cmd) != 0:
            print(f"[FAIL INFER] {run_dir.name}")
            continue

        # 2) BLEU
        bleu_cmd = [
            "python", args.bleu_script,
            "--hyp", str(hyp_path),
            "--ref", str(ref_txt),
            "--tokenize", args.tokenize,
            "--out_json", str(bleu_json),
        ]
        if args.lowercase:
            bleu_cmd.append("--lowercase")
        if args.force:
            bleu_cmd.append("--force")

        if run(bleu_cmd) != 0:
            print(f"[FAIL BLEU] {run_dir.name}")
            continue

        r = read_json(bleu_json)
        r["run_dir"] = str(run_dir)
        results.append(r)

    results.sort(key=lambda x: x.get("bleu", -1e9), reverse=True)

    # summary
    suffix = f"{args.decode}" if args.decode == "greedy" else f"beam{args.beam_size}"
    out_json = save_root / f"summary_{suffix}.json"
    out_csv = save_root / f"summary_{suffix}.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    keys = ["bleu", "signature", "n_sentences", "run_dir", "hyp_path", "ref_path"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})

    print("Saved summary:")
    print("  JSON:", str(out_json))
    print("  CSV :", str(out_csv))


if __name__ == "__main__":
    main()
