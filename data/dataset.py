import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class SteeringDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, behavior_instruction, max_length=256, split="train"):
        """
        Loads the dataset and prepares Teacher and Student inputs.
        """
        self.data = load_dataset(dataset_name, "all", split=split)
        self.tokenizer = tokenizer
        self.behavior_instruction = behavior_instruction
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        user_prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        teacher_text = f"{user_prompt} {self.behavior_instruction}"
        student_text = user_prompt

        teacher_inputs = self.tokenizer(
            teacher_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        student_inputs = self.tokenizer(
            student_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )

        return {
            "teacher_input_ids": teacher_inputs["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_inputs["attention_mask"].squeeze(0),
            "student_input_ids": student_inputs["input_ids"].squeeze(0),
            "student_attention_mask": student_inputs["attention_mask"].squeeze(0),
        }

def get_dataloader(config, tokenizer, split="train"):
    dataset = SteeringDataset(
        dataset_name=config.data.dataset_name,
        tokenizer=tokenizer,
        behavior_instruction=config.data.behavior_instruction,
        max_length=config.model.max_length,
        split=split
    )
    return DataLoader(dataset, batch_size=config.data.batch_size, shuffle=(split=="train"))
