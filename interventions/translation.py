import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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

    def _get_space_embeds(self, batch_size, dtype):
        """Robustly fetches a space token embedding and expands it for the batch."""
        space_ids = self.tokenizer(" ", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        # Fallback if tokenizer strips standalone spaces
        if space_ids.numel() == 0:
            space_ids = self.tokenizer("-", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
        embedding_layer = self.base_model.get_input_embeddings()
        space_embeds = embedding_layer(space_ids)
        return space_embeds.expand(batch_size, -1, -1).to(dtype=dtype)

    def forward(self, input_ids, attention_mask, **kwargs):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        embedding_layer = self.base_model.get_input_embeddings()
        prompt_embeds = embedding_layer(input_ids)
        batch_size = input_ids.size(0)

        steering_embedding_expanded = self.steering_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        space_embeds_expanded = self._get_space_embeds(batch_size, embedding_layer.weight.dtype)
        space_len = space_embeds_expanded.size(1)

        # Added spaces between source and steering
        input_embeds = torch.cat([prompt_embeds, space_embeds_expanded, steering_embedding_expanded], dim=1)
        
        extended_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, space_len, device=self.device),
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

        steering_embedding = self.steering_embedding.to(dtype=embedding_layer.weight.dtype)
        batch_size = input_ids.size(0)
        steering_len = self.steering_embedding.size(0)

        space_embeds_single = self._get_space_embeds(1, embedding_layer.weight.dtype).squeeze(0)
        space_len = space_embeds_single.size(0)
        packed_embeds = []
        packed_masks = []
        target_starts = []
        target_lengths = []

        for i in range(batch_size):
            source_len = int(source_attention_mask[i].sum().item())
            target_len = int(target_attention_mask[i].sum().item())

            source_embeds = embedding_layer(input_ids[i, :source_len])
            target_embeds = embedding_layer(labels[i, :target_len])

            sample_embeds = torch.cat(
                [
                    source_embeds,
                    space_embeds_single,
                    steering_embedding,
                    space_embeds_single,
                    target_embeds,
                ],
                dim=0,
            )

            packed_embeds.append(sample_embeds)
            packed_masks.append(torch.ones(sample_embeds.size(0), device=self.device, dtype=source_attention_mask.dtype))
            target_starts.append(source_len + space_len + steering_len + space_len - 1)
            target_lengths.append(target_len)

        input_embeds = pad_sequence(packed_embeds, batch_first=True)
        attention_mask = pad_sequence(packed_masks, batch_first=True)

        outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=attention_mask)

        max_target_len = max(target_lengths) if target_lengths else 0
        vocab_size = outputs.logits.size(-1)
        target_logits = outputs.logits.new_full((batch_size, max_target_len, vocab_size), 0.0)

        for i, (target_start, target_len) in enumerate(zip(target_starts, target_lengths)):
            target_logits[i, :target_len] = outputs.logits[i, target_start:target_start + target_len]

        padded_labels = labels[:, :max_target_len]
        padded_target_mask = target_attention_mask[:, :max_target_len]
        return target_logits, padded_labels, padded_target_mask

    def training_step(self, batch):
        target_logits, labels, target_attention_mask = self._teacher_forced_logits_and_labels(batch)

        loss_labels = labels.clone()
        loss_labels[target_attention_mask == 0] = -100

        loss = nn.CrossEntropyLoss()(target_logits.reshape(-1, target_logits.size(-1)), loss_labels.reshape(-1))
        
        with torch.no_grad():
            self.steering_embedding.clamp_(-0.5, 0.5)

        return loss

    def evaluation_step(self, batch):
        return self._teacher_forced_logits_and_labels(batch)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=10):
        self.base_model.eval()
        batch_size = input_ids.size(0)
        device = self.device
        
        # Normalize EOS tokens
        eos_ids = self.tokenizer.eos_token_id
        if not isinstance(eos_ids, list):
            eos_ids = [eos_ids]
        eos_ids = [eid for eid in eos_ids if eid is not None]
        
        embedding_layer = self.base_model.get_input_embeddings()
        generated_sequences = []

        for i in range(batch_size):
            prompt_len = int(attention_mask[i].sum().item()) if attention_mask is not None else input_ids.size(1)
            curr_input_ids = input_ids[i, :prompt_len].unsqueeze(0).to(device)
            finished = False

            for _ in range(max_new_tokens):
                curr_embeds = embedding_layer(curr_input_ids)
                steering_embedding_expanded = self.steering_embedding.unsqueeze(0)
                space_embeds_expanded = self._get_space_embeds(1, curr_embeds.dtype)
                input_embeds = torch.cat([curr_embeds, space_embeds_expanded, steering_embedding_expanded], dim=1)

                extended_mask = torch.ones(1, input_embeds.size(1), device=device, dtype=attention_mask.dtype if attention_mask is not None else torch.long)
                outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=extended_mask)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

                if next_token.item() in eos_ids:
                    curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(-1)], dim=1)
                    finished = True
                    break

                curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(-1)], dim=1)

            generated_sequences.append(curr_input_ids.squeeze(0))

        max_seq_len = max(seq.size(0) for seq in generated_sequences) if generated_sequences else 0
        output_ids = input_ids.new_full((batch_size, max_seq_len), self.tokenizer.pad_token_id)
        for i, seq in enumerate(generated_sequences):
            output_ids[i, :seq.size(0)] = seq

        return output_ids
