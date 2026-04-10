import torch
import torch.nn as nn
from interventions.registry import register_intervention


def _sequence_loss(logits, labels):
    criterion = nn.CrossEntropyLoss()
    seq_len = min(logits.size(1), labels.size(1))
    logits = logits[:, :seq_len, :]
    labels = labels[:, :seq_len]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

@register_intervention("arithmetic")
class ArithmeticIntervention(nn.Module):
    def __init__(self, base_model, tokenizer, operator_text):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.hidden_size = base_model.config.hidden_size
        self.device = next(base_model.parameters()).device

        with torch.no_grad():
            instruction_tokens = self.tokenizer(operator_text, return_tensors="pt", truncation=True, max_length=50)
            instruction_embeds = base_model.get_input_embeddings()(instruction_tokens.input_ids.to(self.device))
        self.operator_embedding = nn.Parameter(instruction_embeds.squeeze(0))  # this embedding is learned during training

        print("Initialized ArithmeticIntervention with operator:", operator_text)

    def forward(self, input_ids, attention_mask, **kwargs):
        # input_ids shape: (batch_size, seq_len)
        # e.g. input text is "2 3", and operator_text is "add", then the model should learn to output "5"
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)

        self.operator_embedding_expanded = self.operator_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        input_embeds = torch.cat([prompt_embeds, self.operator_embedding_expanded], dim=1)
        extended_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, self.operator_embedding_expanded.size(1), device=self.device),
            ],
            dim=1,
        )

        outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=extended_attention_mask)

        return outputs.logits

    def training_step(self, batch):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        return _sequence_loss(logits, batch["labels"])
