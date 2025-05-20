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
import string

# Import from the parent package directly
from config import PROCESSED_DATA_DIR

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

def clean_query(query):
    """Replace 'a haiku' or 'haiku' in the query with 'a response' or 'response'"""
    # Replace 'a haiku' with 'a response', and 'haiku' with 'response'
    cleaned = re.sub(r'\ba\s+haiku\b', 'a response', query, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bhaiku\b', 'response', cleaned, flags=re.IGNORECASE)
    if not cleaned.strip():
        cleaned = "Write about something"
    return cleaned.strip()

def preprocess_haiku(text):
    # Split into lines on original newlines
    lines = text.splitlines()
    # Remove punctuation and strip whitespace from each line
    cleaned_lines = [l.translate(str.maketrans('', '', string.punctuation)).strip() for l in lines if l.strip()]
    # Join with ' / '
    return ' / '.join(cleaned_lines)

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
    tkz = transformers.AutoTokenizer.from_pretrained('gpt2-medium')
    tkz.pad_token = tkz.eos_token
    tkz.padding_side = 'left'
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')
    
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
    
    # Process each example in batches for negative generation
    batch_size = 128  # You can adjust this for your GPU/CPU
    num_examples = len(dataset)
    for batch_start in tqdm.tqdm(range(0, num_examples, batch_size), desc="Processing examples"): 
        batch_end = min(batch_start + batch_size, num_examples)
        batch_examples = [dataset[i] for i in range(batch_start, batch_end)]
        batch_queries = [clean_query(ex['question']) for ex in batch_examples]
        batch_positives = [preprocess_haiku(ex['chosen']) for ex in batch_examples]

        # Tokenize batch
        inputs = tkz(batch_queries, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Generate negatives in batch
        with torch.no_grad():
            neg_outputs = model.generate(
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
        # Decode each output
        for i in range(batch_end - batch_start):
            neg = tkz.decode(neg_outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            data['query'].append(batch_queries[i])
            data['positive'].append(batch_positives[i])
            data['negative'].append(neg)

        # Save periodically
        if (batch_end) % 100 == 0 or batch_end == num_examples:
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