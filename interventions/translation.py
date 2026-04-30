import torch
import torch.nn as nn
from interventions.registry import register_intervention


@register_intervention("translation")
class TranslationIntervention(nn.Module):
    def __init__(self, base_model, tokenizer, steering_text):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.device = next(base_model.parameters()).device

        with torch.no_grad():
            instruction_tokens = self.tokenizer(steering_text, return_tensors="pt", truncation=True, max_length=50)
            instruction_embeds = base_model.get_input_embeddings()(instruction_tokens.input_ids.to(self.device))

        self.steering_embedding = nn.Parameter(instruction_embeds.squeeze(0).float())
        print("Initialized TranslationIntervention with steering text:", steering_text)
        print("Steering embedding shape:", self.steering_embedding.shape)

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        e.g. input text is "make", initial sterring_text is "translate" (embedded in the init), then the model should learn to output "fabricar"
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)

        steering_embedding_expanded = self.steering_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        input_embeds = torch.cat([prompt_embeds, steering_embedding_expanded], dim=1)
        extended_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, steering_embedding_expanded.size(1), device=self.device),
            ],
            dim=1,
        )

        outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=extended_attention_mask)

        return outputs.logits

    def _teacher_forced_logits_and_labels(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        source_attention_mask = batch["attention_mask"].to(self.device)
        target_attention_mask = batch["target_attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        embedding_layer = self.base_model.get_input_embeddings()
        source_embeds = embedding_layer(input_ids)
        target_embeds = embedding_layer(labels)

        steering_embedding = self.steering_embedding.to(dtype=embedding_layer.weight.dtype)
        batch_size = input_ids.size(0)
        steering_len = self.steering_embedding.size(0)

        steering_embedding_expanded = steering_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        input_embeds = torch.cat([source_embeds, steering_embedding_expanded, target_embeds], dim=1)

        target_len = labels.size(1)
        attention_mask = torch.cat(
            [
                source_attention_mask,
                torch.ones(batch_size, steering_len, device=self.device),
                target_attention_mask,
            ],
            dim=1,
        )

        outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        target_start = input_ids.size(1) + steering_len - 1
        target_logits = outputs.logits[:, target_start:target_start + labels.size(1), :]
        return target_logits, labels, target_attention_mask

    def training_step(self, batch):
        target_logits, labels, target_attention_mask = self._teacher_forced_logits_and_labels(batch)
        loss_labels = labels.clone()
        loss_labels[target_attention_mask == 0] = -100

        loss = nn.CrossEntropyLoss()(target_logits.reshape(-1, target_logits.size(-1)), loss_labels.reshape(-1))
        return loss

    def evaluation_step(self, batch):
        return self._teacher_forced_logits_and_labels(batch)
