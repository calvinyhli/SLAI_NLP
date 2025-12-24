# finetune_t5.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

try:
    import sacrebleu
except Exception:
    sacrebleu = None


class JsonlTextDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "zh" in obj and "en" in obj:
                    self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True, help="dir containing train.jsonl/valid.jsonl/test.jsonl with zh/en text")
    ap.add_argument("--model_name", default="t5-small")
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)

    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_tgt_len", type=int, default=128)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fp16", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    clean_dir = Path(args.clean_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = JsonlTextDataset(str(clean_dir / "train.jsonl"))
    valid_ds = JsonlTextDataset(str(clean_dir / "valid.jsonl"))

    # For Chinese->English, T5 typically expects a task prefix; you can tune this.
    prefix = "translate Chinese to English: "

    def preprocess(ex):
        src = prefix + ex["zh"]
        tgt = ex["en"]
        model_in = tokenizer(src, max_length=args.max_src_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt, max_length=args.max_tgt_len, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    # tokenize on the fly
    class TokenizedWrapper(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            return preprocess(self.base[i])

    train_tok = TokenizedWrapper(train_ds)
    valid_tok = TokenizedWrapper(valid_ds)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred):
        if sacrebleu is None:
            return {}
        preds, labels = eval_pred
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[tokenizer.decode(l, skip_special_tokens=True)] for l in labels]
        bleu = sacrebleu.corpus_bleu(preds, labels).score
        return {"bleu": bleu}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(save_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        fp16=args.fp16,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if sacrebleu is not None else None,
    )

    trainer.train()
    trainer.save_model(str(save_dir / "best"))
    print("Saved:", str(save_dir / "best"))


if __name__ == "__main__":
    main()
