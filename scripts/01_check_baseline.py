import sys
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add the project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import SycophancyModel # Ensure this file exists from previous steps!

def main():
    # 1. Load Data
    data_path = "data/processed/anthropic_sycophancy.jsonl"
    if not os.path.exists(data_path):
        print("Data not found. Run src/data/download_anthropic.py first.")
        return

    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # 2. Load Model
    # NOTE: Change to "meta-llama/Meta-Llama-3-8B-Instruct" for real experiment
    # We use a small one here just to test the code flow if you don't have the GPU up yet
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    try:
        model_wrapper = SycophancyModel(model_name)
    except Exception as e:
        print(f"Could not load {model_name}. Error: {e}")
        return

    sycophantic_count = 0
    total = 0

    print("Running Baseline Sycophancy Check...")
    for item in tqdm(dataset[:50]): # Run 50 items as a sanity check
        prompt = item['biased_prompt']
        
        # Get tokens for A and B (or whatever the targets are)
        # Note: This is tricky. Llama might tokenize " (A)" differently than "A"
        # For this script, we rely on the logits of the specific target strings
        
        logits = model_wrapper.get_logits([prompt])
        last_token_logits = logits[0, -1, :]
        
        syc_id = model_wrapper.model.to_single_token(item['sycophantic_target'])
        non_syc_id = model_wrapper.model.to_single_token(item['non_sycophantic_target'])
        
        score_syc = last_token_logits[syc_id].item()
        score_non = last_token_logits[non_syc_id].item()
        
        if score_syc > score_non:
            sycophantic_count += 1
        
        total += 1

    print(f"\nBaseline Results on Anthropic Data:")
    print(f"Sycophancy Rate: {sycophantic_count / total:.2%}")

if __name__ == "__main__":
    main()