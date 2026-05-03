import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from interventions.registry import register_intervention


@register_intervention("steering")
class SteeringIntervention(nn.Module):
    def __init__(self, base_model, tokenizer, steering_text=" translate"):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = next(base_model.parameters()).device

        # 1. Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 2. Initialize Steering Embedding (Instructional prefix)
        with torch.no_grad():
            if steering_text:
                instr_ids = self.tokenizer(steering_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            else:
                instr_ids = torch.tensor([[self.tokenizer.unk_token_id]], device=self.device)
            instr_embeds = base_model.get_input_embeddings()(instr_ids)
        
        # We use float() to ensure it's trainable even if the model is in half-precision
        self.steering_embedding = nn.Parameter(instr_embeds.squeeze(0).float())

    def _get_space_embed(self, dtype):
        """Helper to fetch a space embedding to bridge the source and steering."""
        space_id = self.tokenizer(" ", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        return self.base_model.get_input_embeddings()(space_id).squeeze(0).to(dtype=dtype)

    def forward(self, batch):
        """ Teacher-forced forward pass.
        Forward pass that constructs the input sequence as:
        [Source Tokens] [Steering Instruction] [Target Tokens]
        and then slices the output to align with the target tokens for loss calculation.

        Args:
            batch (dict): A batch from the dataloader containing:
                - input_ids: Source token IDs (B x S)
                - attention_mask: Attention mask for source (B x S)
                - labels: Target token IDs (B x T)
                - target_attention_mask: Attention mask for target (B x T)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, labels, mask)
                - logits: Model output logits aligned with target tokens (B x T x Vocab)
                - labels: Target token IDs (B x T)
                - mask: Target attention mask (B x T)
        """
        input_ids = batch["input_ids"].to(self.device)
        source_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)

        embed_layer = self.base_model.get_input_embeddings()
        steer_embed = self.steering_embedding.to(dtype=embed_layer.weight.dtype)
        
        packed_embeds, packed_masks = [], []
        actual_target_lengths = []

        # 1. Build sequences and track actual target lengths
        for i in range(input_ids.size(0)):
            s_len = int(source_mask[i].sum().item())
            t_len = int(target_mask[i].sum().item())
            actual_target_lengths.append(t_len)

            full_seq = torch.cat([
                embed_layer(input_ids[i, :s_len]),
                steer_embed,
                embed_layer(labels[i, :t_len])
            ], dim=0)
            
            packed_embeds.append(full_seq)
            packed_masks.append(torch.ones(full_seq.size(0), device=self.device))

        # 2. Pass through base model
        inputs_embeds = pad_sequence(packed_embeds, batch_first=True)
        attn_mask = pad_sequence(packed_masks, batch_first=True)
        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

        # 3. Slice and Align Logits, Labels, and Masks
        # We must trim everything to the max actual target length in this batch
        max_t = max(actual_target_lengths)
        
        batch_logits = []
        batch_labels = []
        batch_masks = []

        for i in range(input_ids.size(0)):
            s_len = int(source_mask[i].sum().item())
            t_len = actual_target_lengths[i]
            
            # Slice logits: starting at the end of prefix, taking exactly t_len tokens
            start = s_len + steer_embed.size(0) - 1
            logits_slice = outputs.logits[i, start : start + t_len]
            
            # Pad the slice to the batch's max_t so they can be stacked
            padded_logits = torch.nn.functional.pad(
                logits_slice, (0, 0, 0, max_t - t_len), value=0.0
            )
            batch_logits.append(padded_logits)

            # Slice and pad labels/masks to match
            batch_labels.append(torch.nn.functional.pad(labels[i, :t_len], (0, max_t - t_len), value=-100))
            batch_masks.append(torch.nn.functional.pad(target_mask[i, :t_len], (0, max_t - t_len), value=0))

        return torch.stack(batch_logits), torch.stack(batch_labels), torch.stack(batch_masks)

    def training_step(self, batch):
        """Calculates CrossEntropyLoss using the forward pass.
        
        Args:
            batch (dict): A batch from the dataloader containing:
                - input_ids: Source token IDs (B x S)
                - attention_mask: Attention mask for source (B x S)
                - labels: Target token IDs (B x T)
                - target_attention_mask: Attention mask for target (B x T)
        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        logits, labels, mask = self.forward(batch)
        
        # Flatten for loss calculation; ignore padding (-100)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.clone()
        flat_labels[mask == 0] = -100
        
        return nn.CrossEntropyLoss()(flat_logits, flat_labels.reshape(-1))

    def evaluation_step(self, batch):
        """Runs forward pass for evaluation.
        
        Args:
            batch (dict): A batch from the dataloader containing:
                - input_ids: Source token IDs (B x S)
                - attention_mask: Attention mask for source (B x S)
                - labels: Target token IDs (B x T)
                - target_attention_mask: Attention mask for target (B x T)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, labels, mask)
                - logits: Model output logits aligned with target tokens (B x T x Vocab)
                - labels: Target token IDs (B x T)
                - mask: Target attention mask (B x T)
        """
        return self.forward(batch)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=20):
        """Autoregressive text generation.

        Args:
            input_ids (torch.LongTensor): Batch of input token IDs (B x S).
            attention_mask (torch.LongTensor, optional): Attention mask for input_ids (B x S). Defaults to None.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.
        
        Returns:
            torch.LongTensor: Generated token IDs (B x (S + generated)).
        """
        self.base_model.eval()
        embed_layer = self.base_model.get_input_embeddings()
        steer_embed = self.steering_embedding.to(dtype=embed_layer.weight.dtype)
        
        results = []
        for i in range(input_ids.size(0)):
            s_len = int(attention_mask[i].sum().item()) if attention_mask is not None else input_ids.size(1)
            
            # Static prefix: [Source] [Steer]
            prefix = torch.cat([embed_layer(input_ids[i, :s_len].to(self.device)), steer_embed], dim=0).unsqueeze(0)
            
            generated = torch.empty((1, 0), dtype=torch.long, device=self.device)
            for _ in range(max_new_tokens):
                curr_embeds = torch.cat([prefix, embed_layer(generated)], dim=1) if generated.size(1) > 0 else prefix
                out = self.base_model(inputs_embeds=curr_embeds)
                
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                generated = torch.cat([generated, next_token], dim=1)
            
            results.append(torch.cat([input_ids[i, :s_len].to(self.device), generated.squeeze(0)], dim=0))

        # Pad batch for return
        max_l = max(len(r) for r in results)
        output = input_ids.new_full((len(results), max_l), self.tokenizer.pad_token_id)
        for i, r in enumerate(results):
            output[i, :len(r)] = r
        return output
