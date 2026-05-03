import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json

class TranslationDataset(Dataset):
    """
    Dataset for translation tasks, designed for TranslationIntervention.

    Datasets are saved in JSON files with the following format:
    {
        "source language text": "target language text",
        ...
    }

    This dataset exposes tokenized fields for source and target text.
    """

    def __init__(
        self,
        tokenizer,
        dataset_path="datasets/translation/en_es.json",
        input_max_length=128,
        target_max_length=128,
    ):
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length

        dataset_path = os.path.expanduser(dataset_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, dict):
            raise ValueError("Expected JSON object mapping source text to target text.")

        self.examples = [(str(source), str(target)) for source, target in raw_data.items()]
        if len(self.examples) == 0:
            raise ValueError(f"Dataset is empty: {dataset_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        source_text, target_text = self.examples[idx]

        # FIX: Prepend a space for correct BPE boundaries and append EOS to teach stopping
        target_text = " " + target_text
        if self.tokenizer.eos_token:
            target_text += self.tokenizer.eos_token

        source_tokens = self.tokenizer(
            source_text,
            truncation=True,
            max_length=self.input_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        target_tokens = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.target_max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": source_tokens["input_ids"].squeeze(0),
            "attention_mask": source_tokens["attention_mask"].squeeze(0),
            "target_attention_mask": target_tokens["attention_mask"].squeeze(0),
            "labels": target_tokens["input_ids"].squeeze(0),
            "source_text": source_text,
            "target_text": target_text,
        }


def get_dataset(config, tokenizer):
    data_config = config.get("data", {})
    dataset_path = data_config.get("dataset_path", "datasets/translation/en_es.json")
    return TranslationDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        input_max_length=int(data_config.get("input_max_length", 128)),
        target_max_length=int(data_config.get("target_max_length", 128)),
    )


def get_dataloader(config, tokenizer, split="train"):
    data_config = config.get("data", {})
    dataset = get_dataset(config, tokenizer)

    n_total = len(dataset)
    test_size = data_config.get("test_size", 0.2)
    if isinstance(test_size, float):
        n_test = int(round(n_total * test_size))
    else:
        n_test = int(test_size)

    if n_total > 1:
        n_test = max(1, min(n_test, n_total - 1))
    else:
        n_test = 0
    n_train = n_total - n_test

    split_seed = int(data_config.get("split_seed", 42))
    split_generator = torch.Generator().manual_seed(split_seed)
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test], generator=split_generator)

    if split == "train":
        selected_dataset = train_dataset
        shuffle = True
    elif split in {"test", "eval", "validation", "val"}:
        selected_dataset = test_dataset
        shuffle = False
    else:
        raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'test'.")

    return DataLoader(
        selected_dataset,
        batch_size=int(data_config.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
    )