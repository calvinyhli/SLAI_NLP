# scripts/infer_t5.py
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_jsonl_field(path: Path, field: str) -> List[str]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field in obj and isinstance(obj[field], str):
                out.append(obj[field])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to saved model directory (e.g., runs/.../best)")
    ap.add_argument("--clean_test_jsonl", required=True)
    ap.add_argument("--out_hyp", required=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_gen_len", type=int, default=128)

    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    srcs = load_jsonl_field(Path(args.clean_test_jsonl), "zh")
    prefix = "translate Chinese to English: "
    srcs = [prefix + s for s in srcs]

    out_path = Path(args.out_hyp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(0, len(srcs), args.batch_size):
            batch = srcs[i:i+args.batch_size]
            enc = tok(batch, max_length=args.max_src_len, truncation=True, padding=True, return_tensors="pt").to(device)

            gen_kwargs = {"max_length": args.max_gen_len}
            if args.decode == "beam":
                gen_kwargs.update({"num_beams": args.beam_size})
            else:
                gen_kwargs.update({"num_beams": 1, "do_sample": False})

            out_ids = model.generate(**enc, **gen_kwargs)
            outs = tok.batch_decode(out_ids, skip_special_tokens=True)
            for s in outs:
                fout.write(s.strip() + "\n")

    print("Saved hyp:", str(out_path))


if __name__ == "__main__":
    main()
