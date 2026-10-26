import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pidetect.data import load_data

LABELS = {0: "benign", 1: "injection"}


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument("--out", default="model")
    ap.add_argument("--extra-csv", default="data/extra.csv")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_data(args.extra_csv, seed=args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )

    training_args = TrainingArguments(
        output_dir="runs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(len(ds["train"]) / args.batch_size * args.epochs * 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=20,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    out = Path(args.out)
    trainer.save_model(out)
    tok.save_pretrained(out)
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
