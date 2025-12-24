# clean_data_v2.py
import argparse
import json
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


# -----------------------------
# JSONL reader
# -----------------------------
def iter_jsonl(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    """
    Yield (line_no, obj, err_code).
    - err_code is None when parsing succeeds.
    - err_code in {"empty_line", "bad_json"} otherwise.
    """
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


# -----------------------------
# Text cleaning
# -----------------------------
@dataclass(frozen=True)
class TextCleanerConfig:
    remove_control: bool = True
    nfkc: bool = True
    fix_curly_quotes: bool = True
    fix_spacing: bool = True


class TextCleaner:
    """
    Clean a single text string with a small, explicit pipeline.
    """
    CONTROL_CATEGORIES = {"Cc"}

    def __init__(self, config: TextCleanerConfig = TextCleanerConfig()):
        self.cfg = config

    def clean(self, s: str) -> str:
        if self.cfg.remove_control:
            s = self._remove_control_chars(s)
        if self.cfg.nfkc:
            s = unicodedata.normalize("NFKC", s)
        if self.cfg.fix_curly_quotes:
            s = self._fix_unpaired_curly_quotes(s)
        if self.cfg.fix_spacing:
            s = self._fix_spacing(s)
        return s

    def _remove_control_chars(self, s: str) -> str:
        return "".join(ch for ch in s if unicodedata.category(ch) not in self.CONTROL_CATEGORIES)

    def _fix_unpaired_curly_quotes(self, s: str) -> str:
        # If curly quotes are unpaired, remove them to avoid strange tokenization artifacts.
        pairs = [("“", "”"), ("‘", "’")]
        for ql, qr in pairs:
            if s.count(ql) != s.count(qr):
                s = s.replace(ql, "").replace(qr, "")
        return s

    def _fix_spacing(self, s: str) -> str:
        # 1) normalize whitespace to single space + strip
        s = " ".join(s.split()).strip()

        # 2) remove spaces right after opening parens, and right before closing parens
        for open_p in ("(", "（"):
            s = s.replace(open_p + " ", open_p)
        for close_p in (")", "）"):
            s = s.replace(" " + close_p, close_p)

        # 3) remove spaces before punctuation (both EN + ZH)
        puncts = [",", ".", ";", ":", "!", "?", "，", "。", "；", "：", "！", "？", ")", "）"]
        for p in puncts:
            s = s.replace(" " + p, p)

        return s


# -----------------------------
# Cleaning / filtering config & stats
# -----------------------------
@dataclass(frozen=True)
class DatasetCleaningConfig:
    max_char_zh: int = 2000
    max_char_en: int = 2000


@dataclass
class LengthStats:
    zh_sum: int = 0
    en_sum: int = 0
    zh_max: int = 0
    en_max: int = 0

    def update(self, zh: str, en: str) -> None:
        self.zh_sum += len(zh)
        self.en_sum += len(en)
        self.zh_max = max(self.zh_max, len(zh))
        self.en_max = max(self.en_max, len(en))

    def finalize(self, keep: int) -> Dict[str, Any]:
        out = {
            "zh_sum": self.zh_sum,
            "en_sum": self.en_sum,
            "zh_max": self.zh_max,
            "en_max": self.en_max,
        }
        if keep > 0:
            out["zh_avg"] = self.zh_sum / keep
            out["en_avg"] = self.en_sum / keep
        return out


@dataclass
class CleaningReport:
    counts: Dict[str, int] = field(default_factory=dict)
    char_len: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": self.counts, "char_len": self.char_len}


# -----------------------------
# Core cleaner
# -----------------------------
class JsonlParallelCleaner:
    """
    Clean a JSONL parallel dataset where each record must contain:
      - zh: str
      - en: str
    Keep all other fields intact.
    """

    def __init__(self, dataset_cfg: DatasetCleaningConfig, text_cleaner: Optional[TextCleaner] = None):
        self.dataset_cfg = dataset_cfg
        self.text_cleaner = text_cleaner or TextCleaner()

    def clean_file(self, in_path: Path, out_path: Path) -> CleaningReport:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        counts = Counter()
        lens = LengthStats()

        with out_path.open("w", encoding="utf-8") as fout:
            for _, obj, err in iter_jsonl(in_path):
                if err:
                    counts[f"skip_{err}"] += 1
                    continue

                if not isinstance(obj, dict):
                    counts["skip_not_object"] += 1
                    continue

                if "zh" not in obj or "en" not in obj:
                    counts["skip_missing_fields"] += 1
                    continue

                zh0, en0 = obj["zh"], obj["en"]
                if not isinstance(zh0, str) or not isinstance(en0, str):
                    counts["skip_non_str"] += 1
                    continue

                zh = self.text_cleaner.clean(zh0)
                en = self.text_cleaner.clean(en0)

                if not zh or not en:
                    counts["drop_empty_after_clean"] += 1
                    continue

                if len(zh) > self.dataset_cfg.max_char_zh or len(en) > self.dataset_cfg.max_char_en:
                    counts["drop_too_long_char"] += 1
                    continue

                obj["zh"] = zh
                obj["en"] = en
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                counts["keep"] += 1
                lens.update(zh, en)

        report = CleaningReport(
            counts=dict(counts),
            char_len=lens.finalize(keep=counts.get("keep", 0)),
        )
        return report


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean parallel zh-en JSONL datasets.")
    ap.add_argument("--train_in", required=True, type=str)
    ap.add_argument("--valid_in", required=True, type=str)
    ap.add_argument("--test_in", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--max_char_zh", type=int, default=2000)
    ap.add_argument("--max_char_en", type=int, default=2000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_clean_dir = Path(args.out_dir) / "clean"
    out_clean_dir.mkdir(parents=True, exist_ok=True)

    cleaner = JsonlParallelCleaner(
        dataset_cfg=DatasetCleaningConfig(
            max_char_zh=args.max_char_zh,
            max_char_en=args.max_char_en,
        )
    )

    splits = {
        "train": (Path(args.train_in), out_clean_dir / "train.jsonl"),
        "valid": (Path(args.valid_in), out_clean_dir / "valid.jsonl"),
        "test":  (Path(args.test_in),  out_clean_dir / "test.jsonl"),
    }

    all_reports: Dict[str, Dict[str, Any]] = {}
    for split, (inp, outp) in splits.items():
        all_reports[split] = cleaner.clean_file(inp, outp).to_dict()

    stats_path = out_clean_dir / "clean_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Clean files:", str(out_clean_dir))
    print("Stats:", str(stats_path))


if __name__ == "__main__":
    main()
