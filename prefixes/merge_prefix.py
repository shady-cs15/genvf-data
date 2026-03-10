from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


def normalize_source(example: dict) -> dict:
    source = example.get("source")
    if source is None:
        return {"source": []}
    if isinstance(source, str):
        return {"source": [source]}
    if isinstance(source, list):
        return {"source": [str(item) for item in source if item is not None]}
    return {"source": [str(source)]}


def align_to_features(dataset: Dataset, target_features) -> Dataset:
    target_columns = list(target_features.keys())

    missing_columns = [col for col in target_columns if col not in dataset.column_names]
    if missing_columns:
        dataset = dataset.map(lambda _: {col: None for col in missing_columns})

    extra_columns = [col for col in dataset.column_names if col not in target_columns]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    if "source" in dataset.column_names:
        dataset = dataset.map(normalize_source, desc="Normalizing source")

    dataset = dataset.select_columns(target_columns)
    return dataset.cast(target_features)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    merging_files = [
        "qwen3.5-27b.jsonl",
        "qwen3.5-35b-a3b.jsonl",
    ]

    v0 = load_dataset("haoranli-ml/genvf-prefixes-filtered", split="train_replaced_none_prefix")
    target_features = v0.features

    local_datasets = [Dataset.from_json(str(base_dir / name)) for name in merging_files]
    local_datasets = [align_to_features(ds, target_features) for ds in local_datasets]

    total_dataset = concatenate_datasets([v0, *local_datasets])
    print(len(total_dataset))

    # check no prefix is None
    for example in total_dataset:
        if example["prefix"] is None:
            print("Found None prefix:", example)
            exit(-1)

    total_dataset = total_dataset.shuffle(seed=42)
    total_dataset.push_to_hub("haoranli-ml/genvf-prefixes-v1")
