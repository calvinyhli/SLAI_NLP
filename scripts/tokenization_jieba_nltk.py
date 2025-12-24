# tokenize_word_zh_en_v2.py
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

import jieba
from nltk.tokenize import WordPunctTokenizer


# -----------------------------
# Constants
# -----------------------------
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3


# -----------------------------
# IO helpers
# -----------------------------
def iter_jsonl(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    """Yield (line_no, obj, err_code). err_code in {None, 'empty_line', 'bad_json'}."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                yield line_no, None, "empty_line"
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield line_no, None, "bad_json"
                continue
            yield line_no, obj, None


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# -----------------------------
# Tokenizers
# -----------------------------
_space_re = re.compile(r"\s+")


def normalize_ws(s: str) -> str:
    return _space_re.sub(" ", s).strip()


class Tokenizers:
    """Tokenize zh with jieba, en with NLTK WordPunctTokenizer."""
    def __init__(self, jieba_hmm: bool = False):
        self.jieba_hmm = jieba_hmm
        self.en_tok = WordPunctTokenizer()

    def tok_zh(self, s: str) -> List[str]:
        s = normalize_ws(s)
        return [t for t in jieba.lcut(s, cut_all=False, HMM=self.jieba_hmm) if t.strip()]

    def tok_en(self, s: str) -> List[str]:
        s = normalize_ws(s)
        return [t for t in self.en_tok.tokenize(s) if t.strip()]


# -----------------------------
# Vocab
# -----------------------------
def build_vocab(counter: Counter, min_freq: int, max_vocab_size: int) -> Dict[str, int]:
    """
    Build token->id vocab, including SPECIAL_TOKENS.
    Ties are broken by lexicographic order for deterministic output.
    """
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    remain = max(0, max_vocab_size - len(SPECIAL_TOKENS))

    candidates = [(tok, c) for tok, c in counter.items() if c >= min_freq and tok not in vocab]
    candidates.sort(key=lambda x: (-x[1], x[0]))

    for tok, _ in candidates[:remain]:
        vocab[tok] = len(vocab)

    return vocab


def encode(tokens: List[str], vocab: Dict[str, int]) -> Tuple[List[int], int]:
    """Return (ids, unk_count) for the token list (without BOS/EOS)."""
    unk_id = vocab.get("<unk>", UNK)
    ids: List[int] = []
    unk = 0
    for t in tokens:
        i = vocab.get(t, unk_id)
        if i == unk_id:
            unk += 1
        ids.append(i)
    return ids, unk


# -----------------------------
# Stats
# -----------------------------
def len_bin(n: int) -> str:
    if n <= 32: return "0-32"
    if n <= 64: return "33-64"
    if n <= 128: return "65-128"
    if n <= 256: return "129-256"
    if n <= 512: return "257-512"
    return "513+"


@dataclass
class SplitStats:
    counts: Counter = field(default_factory=Counter)
    len_bins_src: Counter = field(default_factory=Counter)
    len_bins_tgt: Counter = field(default_factory=Counter)

    src_unk_tokens: int = 0
    tgt_unk_tokens: int = 0
    src_total_core_tokens: int = 0
    tgt_total_core_tokens: int = 0

    src_len_sum: int = 0
    tgt_len_sum: int = 0
    src_len_max: int = 0
    tgt_len_max: int = 0

    def on_keep(self, src_len: int, tgt_len: int, src_unk: int, tgt_unk: int, src_core: int, tgt_core: int) -> None:
        self.counts["keep"] += 1

        self.src_len_sum += src_len
        self.tgt_len_sum += tgt_len
        self.src_len_max = max(self.src_len_max, src_len)
        self.tgt_len_max = max(self.tgt_len_max, tgt_len)

        self.len_bins_src[len_bin(src_len)] += 1
        self.len_bins_tgt[len_bin(tgt_len)] += 1

        self.src_unk_tokens += src_unk
        self.tgt_unk_tokens += tgt_unk
        self.src_total_core_tokens += src_core
        self.tgt_total_core_tokens += tgt_core

    def to_dict(self) -> Dict[str, Any]:
        keep = self.counts.get("keep", 0)
        out_counts = dict(self.counts)

        if keep > 0:
            out_counts["src_len_avg"] = self.src_len_sum / keep
            out_counts["tgt_len_avg"] = self.tgt_len_sum / keep

        oov = {
            "src_unk_tokens": self.src_unk_tokens,
            "tgt_unk_tokens": self.tgt_unk_tokens,
            "src_total_tokens": self.src_total_core_tokens,
            "tgt_total_tokens": self.tgt_total_core_tokens,
            "src_unk_rate": (self.src_unk_tokens / self.src_total_core_tokens) if self.src_total_core_tokens > 0 else 0.0,
            "tgt_unk_rate": (self.tgt_unk_tokens / self.tgt_total_core_tokens) if self.tgt_total_core_tokens > 0 else 0.0,
        }

        # also expose sums/max in counts for compatibility with the old report style
        out_counts["src_len_sum"] = self.src_len_sum
        out_counts["tgt_len_sum"] = self.tgt_len_sum
        out_counts["src_len_max"] = self.src_len_max
        out_counts["tgt_len_max"] = self.tgt_len_max

        return {
            "counts": out_counts,
            "len_bins_src": dict(self.len_bins_src),
            "len_bins_tgt": dict(self.len_bins_tgt),
            "oov": oov,
        }


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class EncodeConfig:
    max_tokens: int = 256
    len_mode: str = "filter"  # filter|truncate
    ratio_min: float = 0.2
    ratio_max: float = 5.0


# -----------------------------
# Core pipeline
# -----------------------------
class WordLevelPipeline:
    def __init__(self, tokenizers: Tokenizers, enc_cfg: EncodeConfig):
        self.tok = tokenizers
        self.enc_cfg = enc_cfg

    def build_counters_from_train(self, train_path: Path) -> Tuple[Counter, Counter]:
        zh_c = Counter()
        en_c = Counter()
        for _, obj, err in iter_jsonl(train_path):
            if err or obj is None:
                continue
            zh, en = obj.get("zh"), obj.get("en")
            if not isinstance(zh, str) or not isinstance(en, str):
                continue
            zh_c.update(self.tok.tok_zh(zh))
            en_c.update(self.tok.tok_en(en))
        return zh_c, en_c

    def encode_split(
        self,
        in_path: Path,
        out_path: Path,
        vocab_zh: Dict[str, int],
        vocab_en: Dict[str, int],
    ) -> Dict[str, Any]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        st = SplitStats()

        with out_path.open("w", encoding="utf-8") as fout:
            for _, obj, err in iter_jsonl(in_path):
                if err or obj is None:
                    continue

                zh, en = obj.get("zh"), obj.get("en")
                if zh is None or en is None:
                    st.counts["skip_missing_fields"] += 1
                    continue
                if not isinstance(zh, str) or not isinstance(en, str):
                    st.counts["skip_non_str"] += 1
                    continue

                zh_tok = self.tok.tok_zh(zh)
                en_tok = self.tok.tok_en(en)
                if not zh_tok or not en_tok:
                    st.counts["drop_empty_after_tokenize"] += 1
                    continue

                src_core, src_unk = encode(zh_tok, vocab_zh)
                tgt_core, tgt_unk = encode(en_tok, vocab_en)

                src_ids = [BOS] + src_core + [EOS]
                tgt_ids = [BOS] + tgt_core + [EOS]

                # length ratio filter
                ratio = len(src_ids) / max(1, len(tgt_ids))
                if ratio < self.enc_cfg.ratio_min or ratio > self.enc_cfg.ratio_max:
                    st.counts["drop_len_ratio"] += 1
                    continue

                # max token length handling
                if len(src_ids) > self.enc_cfg.max_tokens or len(tgt_ids) > self.enc_cfg.max_tokens:
                    if self.enc_cfg.len_mode == "filter":
                        st.counts["drop_too_long_tok"] += 1
                        continue
                    elif self.enc_cfg.len_mode == "truncate":
                        if len(src_ids) > self.enc_cfg.max_tokens:
                            src_ids = src_ids[: self.enc_cfg.max_tokens]
                        if len(tgt_ids) > self.enc_cfg.max_tokens:
                            tgt_ids = tgt_ids[: self.enc_cfg.max_tokens]
                        st.counts["truncate_too_long_tok"] += 1
                    else:
                        raise ValueError(f"Unknown len_mode: {self.enc_cfg.len_mode}")

                out_obj = {"src_ids": src_ids, "tgt_ids": tgt_ids}
                if "index" in obj:
                    out_obj["index"] = obj["index"]

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                st.on_keep(
                    src_len=len(src_ids),
                    tgt_len=len(tgt_ids),
                    src_unk=src_unk,
                    tgt_unk=tgt_unk,
                    src_core=len(src_core),
                    tgt_core=len(tgt_core),
                )

        return st.to_dict()


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Word-level tokenize+vocab+encode for zh-en parallel JSONL.")
    ap.add_argument("--clean_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--min_freq_zh", type=int, default=2)
    ap.add_argument("--min_freq_en", type=int, default=2)
    ap.add_argument("--max_vocab_zh", type=int, default=30000)
    ap.add_argument("--max_vocab_en", type=int, default=30000)

    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--len_mode", type=str, default="filter", choices=["filter", "truncate"])
    ap.add_argument("--ratio_min", type=float, default=0.2)
    ap.add_argument("--ratio_max", type=float, default=5.0)

    ap.add_argument("--jieba_hmm", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    clean_dir = Path(args.clean_dir)
    out_base = Path(args.out_dir) / "word_tok"
    out_base.mkdir(parents=True, exist_ok=True)

    train_path = clean_dir / "train.jsonl"
    valid_path = clean_dir / "valid.jsonl"
    test_path = clean_dir / "test.jsonl"

    tokenizers = Tokenizers(jieba_hmm=bool(args.jieba_hmm))
    enc_cfg = EncodeConfig(
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )

    pipe = WordLevelPipeline(tokenizers, enc_cfg)

    # 1) counters from train only
    zh_counter, en_counter = pipe.build_counters_from_train(train_path)

    # 2) vocab
    vocab_zh = build_vocab(zh_counter, args.min_freq_zh, args.max_vocab_zh)
    vocab_en = build_vocab(en_counter, args.min_freq_en, args.max_vocab_en)

    vocab_zh_path = out_base / "vocab_zh.json"
    vocab_en_path = out_base / "vocab_en.json"
    write_json(vocab_zh_path, vocab_zh)
    write_json(vocab_en_path, vocab_en)

    # 3) encode splits
    report: Dict[str, Any] = {}
    report["vocab"] = {
        "min_freq_zh": args.min_freq_zh,
        "min_freq_en": args.min_freq_en,
        "max_vocab_zh": args.max_vocab_zh,
        "max_vocab_en": args.max_vocab_en,
        "vocab_size_zh": len(vocab_zh),
        "vocab_size_en": len(vocab_en),
        "vocab_zh_path": str(vocab_zh_path),
        "vocab_en_path": str(vocab_en_path),
        "jieba_hmm": bool(args.jieba_hmm),
        "special_tokens": SPECIAL_TOKENS,
    }

    report["config"] = {
        "max_tokens": enc_cfg.max_tokens,
        "len_mode": enc_cfg.len_mode,
        "ratio_min": enc_cfg.ratio_min,
        "ratio_max": enc_cfg.ratio_max,
    }

    report["encode"] = {
        "train": pipe.encode_split(train_path, out_base / "train.ids.jsonl", vocab_zh, vocab_en),
        "valid": pipe.encode_split(valid_path, out_base / "valid.ids.jsonl", vocab_zh, vocab_en),
        "test":  pipe.encode_split(test_path,  out_base / "test.ids.jsonl",  vocab_zh, vocab_en),
    }

    stats_path = out_base / "word_token_stats.json"
    write_json(stats_path, report)

    print("Done.")
    print("Outputs:", str(out_base))
    print("Vocab zh:", str(vocab_zh_path))
    print("Vocab en:", str(vocab_en_path))
    print("Stats:", str(stats_path))


if __name__ == "__main__":
    main()
