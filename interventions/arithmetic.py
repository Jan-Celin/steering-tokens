import torch
import torch.nn as nn
from interventions.registry import register_intervention


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
            instruction_embeds = base_model.get_input_embeddings()(instruction_tokens.input_ids.to(self.device))  # TODO-JC: Check if embedding initialization affects training stability or convergence (e.g. compare with random initialization)

        # Keep the trainable parameter in fp32 for optimizer stability.
        self.operator_embedding = nn.Parameter(instruction_embeds.squeeze(0).float())
        print("Initialized ArithmeticIntervention with operator:", operator_text)

    def forward(self, input_ids, attention_mask, **kwargs):
        # input_ids shape: (batch_size, seq_len)
        # e.g. input text is "2 3", and operator_text is "add", then the model should learn to output "5"
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)

        operator_embedding = self.operator_embedding.to(dtype=prompt_embeds.dtype)
        self.operator_embedding_expanded = operator_embedding.unsqueeze(0).expand(batch_size, -1, -1)

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
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        embedding_layer = self.base_model.get_input_embeddings()
        operator_embedding = self.operator_embedding.to(dtype=embedding_layer.weight.dtype)

        criterion = nn.CrossEntropyLoss()
        sample_losses = []

        for sample_idx in range(input_ids.size(0)):
            prompt_len = int(attention_mask[sample_idx].sum().item())
            prompt_ids = input_ids[sample_idx, :prompt_len]  # Take only the prompt tokens, excluding padding and operator tokens.

            target_ids = labels[sample_idx]
            valid_target_ids = target_ids[target_ids != -100]  # Filter out padding tokens and ignored positions. Shape: (target_seq_len,)
            if valid_target_ids.numel() == 0:
                continue

            # Teacher forcing: answer token k is predicted using answer tokens < k.
            answer_prefix_ids = valid_target_ids[:-1]

            prompt_embeds = embedding_layer(prompt_ids.unsqueeze(0)).squeeze(0)
            if answer_prefix_ids.numel() > 0:
                answer_prefix_embeds = embedding_layer(answer_prefix_ids.unsqueeze(0)).squeeze(0)
                sample_input_embeds = torch.cat(
                    [prompt_embeds, operator_embedding, answer_prefix_embeds], dim=0
                )
            else:
                sample_input_embeds = torch.cat([prompt_embeds, operator_embedding], dim=0)

            sample_attention_mask = torch.ones(
                1,
                sample_input_embeds.size(0),
                device=self.device,
                dtype=attention_mask.dtype,
            )

            outputs = self.base_model(
                inputs_embeds=sample_input_embeds.unsqueeze(0),
                attention_mask=sample_attention_mask,
            )

            # The first answer token is predicted from the last operator position.
            pred_start = prompt_embeds.size(0) + operator_embedding.size(0) - 1
            pred_end = pred_start + valid_target_ids.size(0)
            answer_logits = outputs.logits[0, pred_start:pred_end, :]

            sample_losses.append(criterion(answer_logits, valid_target_ids))

        if not sample_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return torch.stack(sample_losses).mean()
