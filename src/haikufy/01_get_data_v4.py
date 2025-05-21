import os
import torch
import transformers
import re
import time
import pandas as pd
from pathlib import Path
import random
import string
from collections import Counter
import sys
import csv
from gruen import get_gruen

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

GRUEN_THRESHOLD = 0.5  # You can adjust this threshold
HAIKU_SYLLABLES = 17


def count_syllables(word):
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
    vowels = 'aeiouy'
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if word.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)

def total_syllables(text):
    return sum(count_syllables(w) for w in re.findall(r'\b\w+\b', text))

def repetition_rate(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    counts = Counter(words)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(words)

def is_haiku(text, target=(5,7,5)):
    lines = text.split('\n')
    if len(lines) != 3:
        return False
    sylls = [total_syllables(line) for line in lines]
    return tuple(sylls) == target

def is_near_miss_haiku(text):
    lines = text.split('\n')
    if len(lines) != 3:
        return True  # wrong line count is a near miss
    sylls = [total_syllables(line) for line in lines]
    return (sylls in [(5,7,6),(5,8,5)]) or (tuple(sylls) != (5,7,5) and sum(sylls)==HAIKU_SYLLABLES)

def encode_newlines(text):
    return text.replace('\n', r'\n')

def passes_gruen(text):
    try:
        score = get_gruen([text])[0]
        return score >= GRUEN_THRESHOLD
    except Exception as e:
        print(f"GRUEN scoring failed: {e}")
        return False

def generate_queries(n):
    topics = [
        'nature', 'technology', 'emotions', 'instructions', 'dialogues',
        'friendship', 'space', 'ocean', 'city life', 'childhood', 'future',
        'music', 'art', 'science', 'weather', 'seasons', 'love', 'loss', 'hope', 'fear'
    ]
    forms = [
        lambda t: f"Describe {t} in one word.",
        lambda t: f"What is the essence of {t}?",
        lambda t: f"A conversation about {t} between two people.",
        lambda t: f"Give instructions for {t}.",
        lambda t: f"Express feelings about {t}.",
        lambda t: f"Write a question about {t}.",
        lambda t: f"A short dialogue on {t}.",
        lambda t: f"Summarize {t} in a sentence.",
        lambda t: f"List three facts about {t}.",
        lambda t: f"A memory involving {t}."
    ]
    queries = set()
    while len(queries) < n:
        topic = random.choice(topics)
        form = random.choice(forms)
        queries.add(form(topic))
    return list(queries)

def main():
    print("Starting synthetic haiku dataset generation...")
    start_time = time.time()
    output_dir = Path('data/processed/haiku_dpo_v4')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'haikus.csv'

    MODEL = "mistralai/Mistral-7B-v0.1"  # Use Mistral-7B for better haiku generation

    # Load model
    print(f"Loading model {MODEL}...")
    tkz = transformers.AutoTokenizer.from_pretrained(MODEL)
    tkz.pad_token = tkz.eos_token if hasattr(tkz, 'eos_token') else tkz.pad_token
    tkz.padding_side = 'left'  # Ensure left-padding for decoder-only models
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    queries = generate_queries(20)  # adjust for dataset size
    data = {'query': [], 'positive': [], 'negative': []}

    # Few-shot examples for haiku
    haiku_examples = (
        "Examples:\n"
        "An old silent pond\n"
        "A frog jumps into the pond—\n"
        "Splash! Silence again.\n\n"
        "Winter seclusion—\n"
        "Listening, that evening,\n"
        "To the rain in the mountain.\n\n"
    )

    # Prepare prompts for all queries (few-shot, explicit)
    prompts_pos = [
        f"Write a haiku (3 lines, 5-7-5 syllables, {HAIKU_SYLLABLES} syllables total) about: {q}\n{haiku_examples}Only output the haiku, nothing else.\nHaiku:" for q in queries
    ]
    prompts_neg1 = [f"Write a paragraph (>25 syllables) about: {q}" for q in queries]
    prompts_neg2 = [
        f"Write a 3-line poem about: {q} with {HAIKU_SYLLABLES} syllables but not 5-7-5. Use 5-7-6 or 5-8-5 or break lines oddly." for q in queries
    ]

    # Helper to batch generate with multiple completions
    def batch_generate(prompts, max_length, num_return_sequences=3):
        all_outputs = []
        batch_size = 8  # You can tune this for your GPU/CPU
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
                    num_return_sequences=num_return_sequences
                )
            # outputs shape: (batch_size * num_return_sequences, seq_len)
            for j in range(len(batch_prompts)):
                completions = []
                for k in range(num_return_sequences):
                    idx = j * num_return_sequences + k
                    gen = tkz.decode(outputs[idx][input_ids.shape[1]:], skip_special_tokens=True)
                    completions.append(gen)
                all_outputs.append(completions)
        return all_outputs

    # Batch generate all outputs (with multiple completions)
    print("Generating positives in batch...")
    gens_pos = batch_generate(prompts_pos, 32, num_return_sequences=5)  # Increased completions for positives
    print("Generating negatives1 in batch...")
    gens_neg1 = batch_generate(prompts_neg1, 60, num_return_sequences=1)
    print("Generating negatives2 in batch...")
    gens_neg2 = batch_generate(prompts_neg2, 32, num_return_sequences=1)

    for idx, query in enumerate(queries):
        print (f"Processing query {idx+1}/{len(queries)}: {query}")
        # Process positive: pick the best valid haiku from completions
        best_haiku = None
        for gen in gens_pos[idx]:
            print(f"[DEBUG] Query {idx+1}: Candidate haiku raw output:\n{gen}")
            lines = [l.strip() for l in gen.split('\n') if l.strip()]
            if len(lines) < 3:
                print(f"[DEBUG] Skipped: Less than 3 lines. Lines: {lines}")
                continue
            haiku = '\n'.join(lines[:3])
            syll_count = total_syllables(haiku)
            rep_rate = repetition_rate(haiku)
            haiku_check = is_haiku(haiku)
            try:
                gruen_score = get_gruen([haiku])[0]
            except Exception as e:
                gruen_score = None
            print(f"[DEBUG] Syllables: {syll_count}/{HAIKU_SYLLABLES}, Repetition rate: {rep_rate:.2f}/0.3, is_haiku: {haiku_check}, GRUEN: {gruen_score}/{GRUEN_THRESHOLD}")
            if syll_count != HAIKU_SYLLABLES:
                continue
            if rep_rate > 0.3:
                continue
            if not haiku_check:
                continue
            if gruen_score is None or gruen_score < GRUEN_THRESHOLD:
                continue
            best_haiku = haiku
            break
        if not best_haiku:
            print(f"[SKIP] Query {idx+1}: No valid haiku found in completions.")
            continue
        # Process negative 1
        prose = gens_neg1[idx][0].replace('\n', ' ')
        prose_syll = total_syllables(prose)
        prose_rep = repetition_rate(prose)
        try:
            prose_gruen = get_gruen([prose])[0]
        except Exception as e:
            prose_gruen = None
        print(f"[DEBUG] Query {idx+1}: Negative 1 raw output:\n{gens_neg1[idx][0]}")
        print(f"[DEBUG] Negative 1 Syllables: {prose_syll}, Repetition rate: {prose_rep:.2f}, GRUEN: {prose_gruen}")
        if prose_syll <= 25:
            print(f"[SKIP] Query {idx+1}: Negative 1 has <= 25 syllables. Syllables: {prose_syll}")
            continue
        if prose_rep > 0.3:
            print(f"[SKIP] Query {idx+1}: Negative 1 repetition rate too high: {prose_rep:.2f}")
            continue
        if prose_gruen is None or prose_gruen < GRUEN_THRESHOLD:
            print(f"[SKIP] Query {idx+1}: Negative 1 did not pass GRUEN quality filter.")
            continue
        # Process negative 2
        near_miss = gens_neg2[idx][0]
        print(f"[DEBUG] Query {idx+1}: Negative 2 raw output:\n{near_miss}")
        lines2 = [l.strip() for l in near_miss.split('\n') if l.strip()]
        if len(lines2) < 3:
            print(f"[SKIP] Query {idx+1}: Negative 2 has less than 3 lines.")
            continue
        near_miss_haiku = '\n'.join(lines2[:3])
        nm_syll = total_syllables(near_miss_haiku)
        nm_rep = repetition_rate(near_miss_haiku)
        try:
            nm_gruen = get_gruen([near_miss_haiku])[0]
        except Exception as e:
            nm_gruen = None
        print(f"[DEBUG] Negative 2 Syllables: {nm_syll}, Repetition rate: {nm_rep:.2f}, GRUEN: {nm_gruen}")
        if is_haiku(near_miss_haiku):
            print(f"[SKIP] Query {idx+1}: Negative 2 accidentally conforms to 5-7-5 haiku.")
            continue
        if nm_rep > 0.3:
            print(f"[SKIP] Query {idx+1}: Negative 2 repetition rate too high: {nm_rep:.2f}")
            continue
        if not is_near_miss_haiku(near_miss_haiku):
            nm_lines = near_miss_haiku.split('\n')
            print(f"[SKIP] Query {idx+1}: Negative 2 is not a near-miss haiku. Syllables: {[total_syllables(line) for line in nm_lines]}, sum: {sum([total_syllables(line) for line in nm_lines])}")
            continue
        if nm_gruen is None or nm_gruen < GRUEN_THRESHOLD:
            print(f"[SKIP] Query {idx+1}: Negative 2 did not pass GRUEN quality filter.")
            continue
        # Encode newlines
        q = encode_newlines(query)
        p = encode_newlines(best_haiku)
        n1 = encode_newlines(prose)
        n2 = encode_newlines(near_miss_haiku)
        data['query'].append(q)
        data['positive'].append(p)
        data['negative'].append(n1)
        data['query'].append(q)
        data['positive'].append(p)
        data['negative'].append(n2)
        if len(data['query']) % 20 == 0:
            print(f"Generated {len(data['query'])} rows...")

    # Save to CSV with UTF-8 quoting (append if file exists)
    df = pd.DataFrame(data)
    file_exists = csv_path.exists()
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print(f"Saved {len(df)} rows to {csv_path}")
    print(f"Completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()