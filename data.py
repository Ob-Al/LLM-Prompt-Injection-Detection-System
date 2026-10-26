import csv
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

HF_DATASET = "deepset/prompt-injections"


def load_extra_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"text": row["text"].strip(), "label": int(row["label"])})
    return Dataset.from_list(rows)


def load_data(extra_csv=None, seed=42):
    ds = load_dataset(HF_DATASET)
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")

    train = ds["train"]
    if extra_csv and Path(extra_csv).exists():
        extra = load_extra_csv(extra_csv).cast(train.features)
        train = concatenate_datasets([train, extra]).shuffle(seed=seed)

    split = train.train_test_split(test_size=0.1, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"], test=ds["test"])
