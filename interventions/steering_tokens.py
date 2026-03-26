import torch
import torch.nn as nn

class SteeringTokenIntervention(nn.Module):
    # TODO-JC: Check code!
    def __init__(self, base_model, tokenizer, instruction_text):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.hidden_size = base_model.config.hidden_size
        self.device = next(base_model.parameters()).device

        with torch.no_grad():
            instruction_tokens = self.tokenizer(instruction_text, return_tensors="pt", truncation=True, max_length=50)
            instruction_embeds = base_model.get_input_embeddings()(instruction_tokens.input_ids.to(self.device))
        self.steering_embedding = nn.Parameter(instruction_embeds.squeeze(0))

    def forward(self, input_ids, attention_mask):
        prompt_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)

        self.steering_embedding_expanded = self.steering_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        input_embeds = torch.cat([prompt_embeds, self.steering_embedding_expanded], dim=1)
        extended_attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device)], dim=1)

        outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=extended_attention_mask)

        return outputs.logits
