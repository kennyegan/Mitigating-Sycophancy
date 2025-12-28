import json
import random
from datasets import load_dataset
from tqdm import tqdm

def process_anthropic_data(output_path="data/processed/anthropic_sycophancy.jsonl", max_samples=1000):
    """
    Downloads the standard Anthropic sycophancy evaluation dataset 
    and converts it to a clean format for Llama-3 analysis.
    """
    print("Downloading Anthropic/sycophancy dataset...")
    # This dataset contains pairs of text where the user expresses a view
    # and we check if the model agrees.
    ds = load_dataset("Anthropic/sycophancy", split="train")
    
    # Shuffle to get a mix of topics (politics, philosophy, NLP)
    # We convert to list to sample easily
    ds_list = list(ds)
    random.seed(42)
    selected_data = random.sample(ds_list, min(max_samples, len(ds_list)))
    
    processed = []
    
    print(f"Processing {len(selected_data)} examples...")
    
    for item in tqdm(selected_data):
        # The Anthropic dataset structure is a bit distinct.
        # It usually has 'question' and 'answer_matching_behavior' vs 'answer_not_matching_behavior'
        
        # We need to construct the Prompt carefully.
        # usually: "Human: [Question + Opinion] \n Assistant:"
        
        prompt_text = item['question'] # This often includes the user's bio/opinion
        
        # We need to identify which token represents the "Sycophantic" choice.
        # The dataset provides 'answer_matching_behavior' (e.g., " (A)")
        sycophantic_text = item['answer_matching_behavior'] 
        not_sycophantic_text = item['answer_not_matching_behavior']
        
        # Llama-3 Chat Template Construction
        # We must make it look like a real conversation
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        processed.append({
            "source": "anthropic_eval",
            "neutral_prompt": None, # Anthropic dataset doesn't easily give a "neutral" baseline, so we treat it as a pure classification task first
            "biased_prompt": formatted_prompt,
            "sycophantic_target": sycophantic_text.strip(),
            "non_sycophantic_target": not_sycophantic_text.strip()
        })
        
    # Save
    with open(output_path, 'w') as f:
        for entry in processed:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    process_anthropic_data()