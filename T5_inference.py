# scripts/sweep_t5_bleu.py
import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run(cmd: List[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def export_ref(clean_test_jsonl: Path, out_ref: Path, field: str = "en"):
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    with clean_test_jsonl.open("r", encoding="utf-8") as fin, out_ref.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field in obj and isinstance(obj[field], str):
                fout.write(obj[field].strip() + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_root", required=True)
    ap.add_argument("--clean_test_jsonl", required=True)

    ap.add_argument("--infer_script", default="scripts/infer_t5.py")
    ap.add_argument("--bleu_script", default="scripts/evaluate_bleu.py")

    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)

    ap.add_argument("--tokenize", default="13a")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    save_root = Path(args.save_root)
    ref_txt = save_root / "ref.test.en.txt"
    if not ref_txt.exists():
        export_ref(Path(args.clean_test_jsonl), ref_txt, "en")

    results: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in save_root.iterdir() if p.is_dir()]):
        model_dir = run_dir / "best"
        if not model_dir.exists():
            continue

        hyp_name = f"hyp.{args.decode}.txt" if args.decode == "greedy" else f"hyp.beam{args.beam_size}.txt"
        hyp = run_dir / hyp_name
        bleu_json = run_dir / (hyp_name + ".bleu.json")

        if args.skip_existing and bleu_json.exists():
            r = read_json(bleu_json)
            r["run_dir"] = str(run_dir)
            results.append(r)
            continue

        infer_cmd = [
            "python", args.infer_script,
            "--model_dir", str(model_dir),
            "--clean_test_jsonl", args.clean_test_jsonl,
            "--out_hyp", str(hyp),
            "--decode", args.decode,
            "--device", args.device,
        ]
        if args.decode == "beam":
            infer_cmd += ["--beam_size", str(args.beam_size)]
        if run(infer_cmd) != 0:
            print(f"[FAIL INFER] {run_dir.name}")
            continue

        bleu_cmd = [
            "python", args.bleu_script,
            "--hyp", str(hyp),
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

    suffix = args.decode if args.decode == "greedy" else f"beam{args.beam_size}"
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

    print("Saved:", out_json, out_csv)


if __name__ == "__main__":
    main()
