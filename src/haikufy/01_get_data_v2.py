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
    """Replace 'a haiku' or 'haiku' in the query with 'a response' or 'response'"""
    # Replace 'a haiku' with 'a response', and 'haiku' with 'response'
    cleaned = re.sub(r'\ba\s+haiku\b', 'a response', query, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bhaiku\b', 'response', cleaned, flags=re.IGNORECASE)
    if not cleaned.strip():
        cleaned = "Write about something"
    return cleaned.strip()

def count_syllables(line):
    # Simple syllable counter for English words
    line = line.lower()
    words = re.findall(r'[a-zA-Z]+', line)
    count = 0
    for word in words:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        if len(word) == 0:
            continue
        # Heuristic: count vowel groups as syllables
        syllables = re.findall(r'[aeiouy]+', word)
        count += max(1, len(syllables))
    return count

def is_haiku(text):
    # Check if text is a 3-line haiku with 5-7-5 syllable structure
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) != 3:
        return False
    syllable_pattern = [5, 7, 5]
    for i, line in enumerate(lines):
        syl = count_syllables(line)
        # Allow +/- 1 syllable for robustness
        if abs(syl - syllable_pattern[i]) > 1:
            return False
    return True

def clean_haiku(text):
    # Remove extra whitespace, ensure 3 lines, strip each line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) == 3:
        return '\n'.join(lines)
    # Try to split by comma if only one line
    if len(lines) == 1 and ',' in lines[0]:
        parts = [p.strip() for p in lines[0].split(',')]
        if len(parts) == 3:
            return '\n'.join(parts)
    # Otherwise, return as is
    return text.strip()

def is_noisy(text):
    # Remove if empty, not a string, or contains too many non-printable chars
    if not isinstance(text, str) or not text.strip():
        return True
    if sum(1 for c in text if not c.isprintable()) > 5:
        return True
    if len(text) > 300:
        return True
    return False

def is_near_haiku(text):
    # Looks like a haiku but not quite (wrong syllable count, but 3 lines)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return len(lines) == 3

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
    tkz = transformers.AutoTokenizer.from_pretrained('gpt2-xl')
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
        batch_positives = [ex['chosen'] for ex in batch_examples]

        # Clean and filter positives
        cleaned_positives = []
        cleaned_queries = []
        for q, pos in zip(batch_queries, batch_positives):
            pos_clean = clean_haiku(pos)
            if is_noisy(pos_clean):
                continue
            if not is_haiku(pos_clean):
                continue
            cleaned_positives.append(pos_clean)
            cleaned_queries.append(q)
        if not cleaned_positives:
            continue

        # Tokenize batch
        inputs = tkz(cleaned_queries, return_tensors='pt', padding=True, truncation=True)
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
        for i in range(len(cleaned_positives)):
            neg = tkz.decode(neg_outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # Clean and filter negatives
            if is_noisy(neg):
                continue
            if is_haiku(neg):
                continue  # Don't allow accidental haikus
            if neg.strip() == cleaned_positives[i].strip():
                continue
            if is_near_haiku(neg):
                continue  # Don't allow near-miss haikus
            data['query'].append(cleaned_queries[i])
            data['positive'].append(cleaned_positives[i])
            data['negative'].append(neg)

        # Save periodically
        if (batch_end) % 100 == 0 or batch_end == num_examples:
            df = pd.DataFrame(data)
            csv_path = output_dir / 'haikus.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(data['query'])} examples to {csv_path}")
    
    # Create final DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / 'haikus.csv'
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
    sample_csv_path = output_dir / 'haikus_sample.csv'
    sample_df.to_csv(sample_csv_path, index=False)
    
    print(f"Processing completed in {time.time() - start_time:.2f}s")
    print(f"Full dataset saved to {csv_path}")
    print(f"Sample dataset with {sample_size} examples saved to {sample_csv_path}")

if __name__ == "__main__":
    main()