import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_manager import load_model_and_tokenizer


def find_n_closest_tokens(embedding, base_model, tokenizer, n=5):
    """
    For a given embedding, find n tokens, which, when summed together, are closest to the embedding in cosine similarity.
    This is a combinatorial problem, and we will use a greedy approach for simplicity.
    The algorithm will iteratively find the token that, when added to the current sum of tokens, is closest to the target embedding.
    """

    current_sum = torch.zeros_like(embedding)
    selected_tokens = []

    for _ in range(n):
        best_token_id = None
        best_similarity = float('-inf')

        for token_id in range(len(tokenizer)):
            if token_id not in selected_tokens:
                token_embedding = base_model.get_input_embeddings().weight[token_id].float()
                candidate_sum = current_sum + token_embedding
                similarity = F.cosine_similarity(candidate_sum, embedding, dim=0).item()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_token_id = token_id

        if best_token_id is not None:
            selected_tokens.append(best_token_id)
            current_sum += base_model.get_input_embeddings().weight[best_token_id].float()

    return selected_tokens

def main():
    parser = argparse.ArgumentParser(description="Search tokens in a checkpoint")
    parser.add_argument("-c", "--checkpoint", "--checkpoint-path", dest="checkpoint_path", type=str,
                        help="Path to the checkpoint file (optional flag)")

    args = parser.parse_args()
    
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    steering_embedding = checkpoint['steering_embedding']
    config = checkpoint['config']

    model_name = config['model']['name_or_path']
    base_model, tokenizer = load_model_and_tokenizer(model_name, torch.device("cpu"))

    similarities = []
    steering_embedding = steering_embedding.squeeze(0).float()

    for token_id in range(len(tokenizer)):
        token_embedding = base_model.get_input_embeddings().weight[token_id].float()
        similarity = F.cosine_similarity(token_embedding, steering_embedding, dim=0).item()
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token: {token}, Similarity to steering embedding: {similarity:.4f}")
        similarities.append((similarity, token))

    # sort similarities and print the top 10 closest tokens
    similarities.sort(key=lambda x: x[0], reverse=True)
    print("\nTop 10 closest tokens to the steering embedding:")
    for similarity, token in similarities[:10]:
        print(f"Token: {token}, Similarity: {similarity:.4f}")


    print("\nFinding combinations of tokens that approximate the steering embedding...")

    closest_tokens = find_n_closest_tokens(steering_embedding, base_model, tokenizer, n=5)
    print("Closest tokens that approximate the steering embedding:\n\t", [tokenizer.convert_ids_to_tokens(token_id) for token_id in closest_tokens])

if __name__ == "__main__":
    main()
