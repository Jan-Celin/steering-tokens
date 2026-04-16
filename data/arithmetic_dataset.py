import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re


ARITHMETIC_PATTERN = re.compile(
	r"^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


def _parse_instruction(instruction):
	match = ARITHMETIC_PATTERN.match(instruction)
	if not match:
		raise ValueError(f"Could not parse instruction: {instruction}")
	left, operator, right = match.groups()
	return left, operator, right


class SimpleMathArithmeticDataset(Dataset):
	"""
	Dataset for fblgit/simple-math designed for ArithmeticIntervention.

	Each row in fblgit/simple-math has:
	  - instruction: "<num1> <operator> <num2>"
	  - output: resulting value as text

	This dataset exposes tokenized fields for two-number input + operator token.
	"""

	def __init__(
		self,
		tokenizer,
		dataset_name="fblgit/simple-math",
		split="train",
		input_max_length=32,
		target_max_length=16,
		operator_filter=None,
	):
		self.tokenizer = tokenizer
		self.input_max_length = input_max_length
		self.target_max_length = target_max_length

		# verification_mode="no_checks" is needed because this dataset can fail
		# split verification with newer datasets versions.
		dataset = load_dataset(dataset_name, split=split, verification_mode="no_checks")

		if operator_filter is not None:
			allowed = set(operator_filter) if isinstance(operator_filter, (list, tuple, set)) else {operator_filter}

			def keep_operator(example):
				try:
					_, operator, _ = _parse_instruction(example["instruction"])
				except ValueError:
					return False
				return operator in allowed

			dataset = dataset.filter(keep_operator)

		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		row = self.dataset[idx]
		left, operator, right = _parse_instruction(row["instruction"])
		target = row["output"]

		# ArithmeticIntervention input requirement: operator token + two numbers.
		numbers_prompt = f"{left} {right}"
		operator_text = operator

		prompt_tokens = self.tokenizer(
			numbers_prompt,
			truncation=True,
			max_length=self.input_max_length,
			padding="max_length",
			return_tensors="pt",
		)
		operator_tokens = self.tokenizer(
			operator_text,
			truncation=True,
			max_length=4,
			padding="max_length",
			return_tensors="pt",
		)
		target_tokens = self.tokenizer(
			target,
			truncation=True,
			max_length=self.target_max_length,
			padding="max_length",
			return_tensors="pt",
		)

		labels = target_tokens["input_ids"].squeeze(0)
		labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

		return {
			# Primary keys for ArithmeticIntervention-centric training loops
			"input_ids": prompt_tokens["input_ids"].squeeze(0),
			"attention_mask": prompt_tokens["attention_mask"].squeeze(0),
			"operator_ids": operator_tokens["input_ids"].squeeze(0),
			"operator_attention_mask": operator_tokens["attention_mask"].squeeze(0),
			"labels": labels,
			# Helpful raw fields for debugging/evaluation
			"left": left,
			"right": right,
			"operator": operator,
			"target_text": target,
			# Backward-compatible aliases for the existing trainer
			"student_input_ids": prompt_tokens["input_ids"].squeeze(0),
			"student_attention_mask": prompt_tokens["attention_mask"].squeeze(0),
			"student_labels": labels,
		}


def get_dataloader(config, tokenizer, split="train"):
	data_cfg = config.get("data", {})
	intervention_cfg = config.get("intervention", {})
	intervention_params = intervention_cfg.get("params", {})

	operator_filter = intervention_params.get("operator_text")

	dataset = SimpleMathArithmeticDataset(
		tokenizer=tokenizer,
		dataset_name=data_cfg.get("dataset_name", "fblgit/simple-math"),
		split=split,
		input_max_length=int(data_cfg.get("input_max_length", 32)),
		target_max_length=int(data_cfg.get("target_max_length", 16)),
		operator_filter=operator_filter,
	)

	return DataLoader(
		dataset,
		batch_size=int(data_cfg.get("batch_size", 8)),
		shuffle=split == "train",
		num_workers=int(data_cfg.get("num_workers", 0)),
		pin_memory=bool(data_cfg.get("pin_memory", False)),
	)
