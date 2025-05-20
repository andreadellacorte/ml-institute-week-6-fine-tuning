import os
import torch
import transformers
import re
import time
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import tqdm
import random

# Import from the parent package directly
from config import PROCESSED_DATA_DIR

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

def clean_query(query):
    """Remove 'a haiku' or 'haiku' from the query"""
    cleaned = re.sub(r'\b(a\s+)?haiku\b', '', query, flags=re.IGNORECASE).strip()
    if not cleaned:
        cleaned = "Write about something"
    return cleaned

def main():
    print("Starting data preparation process...")
    start_time = time.time()
    
    # Create output directory
    output_dir = PROCESSED_DATA_DIR / "haiku_dpo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load haiku dataset
    print("Loading dataset...")
    dataset = load_dataset("davanstrien/haiku_dpo", split="train")
    print(f"Loaded {len(dataset)} examples from haiku_dpo dataset")
    
    # Setup models for negative example generation
    print("Loading models for negative example generation...")
    tkz = transformers.AutoTokenizer.from_pretrained('gpt2')
    tkz.pad_token = tkz.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    
    # Check if GPU is available and move model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Using device: {device}")
    
    # Prepare data storage
    data = {
        'query': [],
        'positive': [],
        'negative': []
    }
    
    # Process each example
    print("Processing examples and generating negative responses...")
    for idx, example in enumerate(tqdm.tqdm(dataset, desc="Processing examples")):
        # Clean the query
        query = clean_query(example['question'])
        
        # Get positive example from chosen response
        positive = example['chosen']
        
        # Generate negative response
        with torch.no_grad():
            # Tokenize query
            inputs = tkz(query, return_tensors='pt', padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Generate response
            neg_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                min_length=10,
                num_return_sequences=1,
                pad_token_id=tkz.eos_token_id,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
            negative = tkz.decode(neg_output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
        # Store data
        data['query'].append(query)
        data['positive'].append(positive)
        data['negative'].append(negative)
        
        # Save periodically
        if (idx + 1) % 100 == 0 or idx == len(dataset) - 1:
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            csv_path = output_dir / 'haiku_dpo_processed.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(data['query'])} examples to {csv_path}")
    
    # Create final DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / 'haiku_dpo_processed.csv'
    df.to_csv(csv_path, index=False)
    
    # Also save a smaller subset for quick experiments
    sample_size = min(1000, len(data['query']))
    sample_indices = random.sample(range(len(data['query'])), sample_size)
    sample_data = {
        'query': [data['query'][i] for i in sample_indices],
        'positive': [data['positive'][i] for i in sample_indices],
        'negative': [data['negative'][i] for i in sample_indices]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = output_dir / 'haiku_dpo_processed_sample.csv'
    sample_df.to_csv(sample_csv_path, index=False)
    
    print(f"Processing completed in {time.time() - start_time:.2f}s")
    print(f"Full dataset saved to {csv_path}")
    print(f"Sample dataset with {sample_size} examples saved to {sample_csv_path}")

if __name__ == "__main__":
    main()