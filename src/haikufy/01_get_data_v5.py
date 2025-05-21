import torch
import transformers
import time
import pandas as pd
from pathlib import Path
import datasets
import random

HAIKU_SYLLABLES = 17

# set random seed
random.seed(42)

# Replace newlines for CSV
encode_newlines = lambda text: text.replace('\n', r'\n')

def generate_query(topic):
    forms = [
        lambda t: f"Describe {t} in a few words.",
        lambda t: f"What is the essence of {t}?",
        lambda t: f"A conversation about {t}.",
        lambda t: f"Give instructions for {t}.",
        lambda t: f"Express feelings about {t}.",
        lambda t: f"Write a reply about {t}.",
        lambda t: f"A short dialogue on {t}.",
        lambda t: f"Summarize {t} in a sentence.",
        lambda t: f"List some facts about {t}.",
        lambda t: f"A memory involving {t}."
    ]

    form = random.choice(forms)
    return form(topic)

def main():
    print("Starting haiku dataset generation (statworx/haiku positives)...")
    start_time = time.time()
    output_dir = Path('data/processed/haiku_dpo_v5')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'haikus.csv'

    # Load dataset
    haiku_ds = datasets.load_dataset("statworx/haiku", split="train")
    num_samples = len(haiku_ds)
    keywords = [haiku_ds[idx].get('keywords', '') for idx in range(num_samples)]

    # Load model
    MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tkz = transformers.AutoTokenizer.from_pretrained(MODEL)
    tkz.pad_token = tkz.eos_token if hasattr(tkz, 'eos_token') else tkz.pad_token
    tkz.padding_side = 'left'
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Prompts for negatives
    prompts_neg1 = [f"Write a paragraph (>25 syllables) about: {k}" for k in keywords]
    prompts_neg2 = [f"Write a 3-line poem about: {k} with {HAIKU_SYLLABLES-2} to {HAIKU_SYLLABLES-2} syllables, but not {HAIKU_SYLLABLES}." for k in keywords]

    # Load existing indices if file exists
    existing_indices = set()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, usecols=['index'])
            existing_indices = set(existing_df['index'].dropna().astype(int).tolist())
        except Exception:
            pass

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

    batch_size = 32
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
            # Clean up haiku (ensure 3 lines)
            lines = [l.strip() for l in haiku.replace('/', '\n').split('\n') if l.strip()]
            haiku = '\n'.join(lines[:3]) if len(lines) >= 3 else haiku
            prose = prose.replace('\n', ' ')
            lines2 = [l.strip() for l in near_miss.split('\n') if l.strip()]
            near_miss_haiku = '\n'.join(lines2[:3]) if len(lines2) >= 3 else near_miss
            q = encode_newlines(query)
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