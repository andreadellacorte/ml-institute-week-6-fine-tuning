import torch
import transformers
import time
import pandas as pd
from pathlib import Path
import datasets
import random
import string
import re

HAIKU_SYLLABLES = 17

# set random seed
random.seed(42)

# Replace newlines for CSV
encode_newlines = lambda text: text.replace('\n', r'\n')

def generate_query(topic):
    forms = [
        #lambda t: f"Describe {t} in a few words.",
        #lambda t: f"Express feelings about {t}.",
        lambda t: f"A memory involving {t}: "
        #lambda t: f"{t} is... ",
    ]

    form = random.choice(forms)
    return form(topic)

def main():
    print("Starting haiku dataset generation (statworx/haiku positives)...")
    start_time = time.time()
    output_dir = Path('data/processed/statworx_haiku')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'haikus.csv'

    # Load dataset
    haiku_ds = datasets.load_dataset("statworx/haiku", split="train")
    num_samples = len(haiku_ds)
    keywords = [haiku_ds[idx].get('keywords', '') for idx in range(num_samples)]

    # Load model

    MODEL = "gpt2-large"

    tkz = transformers.AutoTokenizer.from_pretrained(MODEL)
    tkz.pad_token = tkz.eos_token if hasattr(tkz, 'eos_token') else tkz.pad_token
    tkz.padding_side = 'left'
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Prompts for negatives
    prompts_neg1 = [f"Write a paragraph (>25 syllables) about: {k}" for k in keywords]
    prompts_neg2 = [
        f"Write a 3-line poem about: {k} with {random.choice([15, 16, 18, 19])} syllables"
        for k in keywords
    ]

    # Load existing indices if file exists
    existing_indices = set()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, usecols=['index'])
            existing_indices = set(existing_df['index'].dropna().astype(int).tolist())
        except Exception:
            pass
    
    # Syllable counting function
    def count_syllables(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        count = 0
        for word in words:
            word = word.strip()
            if not word:
                continue
            # Simple syllable estimation
            syllables = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e'):
                syllables = max(1, syllables - 1)
            count += max(1, syllables)
        return count

    def batch_generate(prompts, max_length):
        all_outputs = []
        batch_size = 32
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            encodings = tkz(batch_prompts, return_tensors='pt', padding=True)
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1]+max_length,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tkz.eos_token_id,
                    top_p=0.95,
                    num_return_sequences=1
                )
            for j in range(len(batch_prompts)):
                gen = tkz.decode(outputs[j][input_ids.shape[1]:], skip_special_tokens=True)
                all_outputs.append(gen)
        return all_outputs

    batch_size = 64
    file_exists = csv_path.exists()
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_keywords = keywords[batch_start:batch_end]
        batch_prompts_neg1 = prompts_neg1[batch_start:batch_end]
        batch_prompts_neg2 = prompts_neg2[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        # Filter out indices that already exist
        filtered = [(i, k, p1, p2) for i, k, p1, p2 in zip(batch_indices, batch_keywords, batch_prompts_neg1, batch_prompts_neg2) if i not in existing_indices]
        if not filtered:
            continue
        filtered_indices, filtered_keywords, filtered_prompts_neg1, filtered_prompts_neg2 = zip(*filtered)

        print(f"Generating negatives1 for batch {batch_start}-{batch_end}...")
        gens_neg1 = batch_generate(filtered_prompts_neg1, 60)
        print(f"Generating negatives2 for batch {batch_start}-{batch_end}...")
        gens_neg2 = batch_generate(filtered_prompts_neg2, 32)

        batch_data = {'index': [], 'query': [], 'positive': [], 'negative': []}
        for idx, keyword, prose, near_miss in zip(filtered_indices, filtered_keywords, gens_neg1, gens_neg2):
            query = generate_query(keyword)
            haiku = haiku_ds[idx]['text']
            # Replace '/' with '\n' for line breaks
            haiku = haiku.replace('/', '\n')
            # Only remove .,?! from haiku
            haiku = haiku.translate(str.maketrans('', '', '.?,!'))
            # Ensure lines are separated by '\n' (not '/')
            lines = [l.strip() for l in haiku.split('\n') if l.strip()]
            haiku = '\n'.join(lines)
            prose = prose.replace('\n', ' ')
            # Ensure near_miss_haiku has exactly 3 lines (2 '\n'), with line breaks in between, not at the end
            lines2 = [l.strip() for l in near_miss.split('\n') if l.strip()]
            if len(lines2) < 3:
                lines2 += [''] * (3 - len(lines2))
            near_miss_haiku = '\n'.join(lines2[:3])  # This guarantees two '\n' between three lines, none at the end

            if count_syllables(near_miss_haiku) == HAIKU_SYLLABLES:
                print(f"Generated near_miss_haiku {idx} has exactly {HAIKU_SYLLABLES} syllables:\n{near_miss_haiku}")
                continue
            
            # Add a newline to the end of the query for training data consistency
            q = encode_newlines(query.rstrip() + "\n")
            p = encode_newlines(haiku)
            n1 = encode_newlines(prose)
            n2 = encode_newlines(near_miss_haiku)
            batch_data['index'].append(idx)
            batch_data['query'].append(q)
            batch_data['positive'].append(p)
            batch_data['negative'].append(n1)
            batch_data['index'].append(idx)
            batch_data['query'].append(q)
            batch_data['positive'].append(p)
            batch_data['negative'].append(n2)

        if not batch_data['index']:
            continue  # Skip writing if nothing new
        df = pd.DataFrame(batch_data)
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False, quoting=1, encoding='utf-8')
        file_exists = True  # Only write header for the first batch
        print(f"Saved {len(df)} rows to {csv_path} (up to sample {batch_end})")

    print(f"Completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()