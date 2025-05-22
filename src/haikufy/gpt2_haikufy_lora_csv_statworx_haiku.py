import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import transformers
import time
import re
import csv
from torch.utils.data import Dataset, DataLoader
import random
from datetime import datetime
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import tqdm
import textstat
import wandb

from get_data_v5 import generate_query

# Import from the parent package directly
from src.config import CHECKPOINTS_DATA_DIR

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"Loading models and datasets...")
start_time = time.time()

batch_size_per_model = {
    'gpt2': 128,
    'gpt2-medium': 64,
    'gpt2-large': 28,
    'gpt2-xl': 4,
}

# Load models
# Use a larger model for better haiku generation
MODEL_NAME = 'gpt2'  # You can change to mistralai/Mistral-7B-v0.1 or another large model if desired
print(f"Loading model: {MODEL_NAME}")
tkz = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tkz.pad_token = tkz.eos_token
plc = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set reference model to eval mode
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

# Configure LoRA for the policy model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # rank of the LoRA adapter
    lora_alpha=16,  # scaling factor
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # GPT-2 specific attention modules
    fan_in_fan_out=True,  # Set to True for Conv1D layers in GPT-2
    bias="none"
)

# Apply LoRA to the policy model
plc = get_peft_model(plc, lora_config)
plc.print_trainable_parameters()  # This will show the parameter reduction

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plc.to(device)
ref.to(device)
print(f"Using device: {device}")

# Setup optimizer
# We only optimize the LoRA parameters now
optm = torch.optim.Adam(plc.parameters(), lr=5e-5)
beta = 0.1

# Custom Dataset for Haiku DPO CSV
data_csv_path = Path('data/processed/statworx_haiku/haikus.csv')

class HaikuDPODataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Decode escaped newlines to real newlines
                pos = row['positive'].replace('\\n', '\n')
                neg = row['negative'].replace('\\n', '\n')
                pos_lines = [l.strip() for l in pos.split('\n') if l.strip()]
                if len(pos_lines) != 3:
                    continue  # Skip if not exactly 3 non-empty lines
                self.samples.append({
                    'query': row['query'],
                    'positive': pos,
                    'negative': neg
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

dataset = HaikuDPODataset(data_csv_path)
print(f"Loaded {len(dataset)} examples from {data_csv_path}")

# Use DataLoader for efficient batching and shuffling
batch_size = batch_size_per_model[MODEL_NAME]
num_workers = 2 if torch.cuda.is_available() else 0
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

# Add mixed precision support
if torch.cuda.is_available():
    scaler = torch.amp.GradScaler(device='cuda')
else:
    scaler = None

print(f"Setup completed in {time.time() - start_time:.2f}s")

def tokenise(qrys, ress, max_length=128):
    # Accepts lists of queries and responses, returns batched tensors, with padding and truncation
    qry_ids = tkz(qrys, return_tensors='pt', padding=True, truncation=True, max_length=max_length, add_special_tokens=False).input_ids.to(device).long()
    res_ids = tkz(ress, return_tensors='pt', padding=True, truncation=True, max_length=max_length, add_special_tokens=False).input_ids.to(device).long()
    acc_ids = torch.cat([qry_ids, res_ids], dim=1)
    atn_msk = torch.ones_like(acc_ids).long()
    lbl_ids = acc_ids.clone()
    for i in range(len(qrys)):
        lbl_ids[i, :qry_ids.size(1)] = -100
    return acc_ids, atn_msk, lbl_ids

def sum_logp(model, ids, msk, lbl):
    out = model(input_ids=ids, attention_mask=msk)
    log = out.logits.log_softmax(-1)[:, :-1]
    tgt = lbl[:, 1:].masked_fill(lbl[:, 1:] == -100, 0)
    tok = log.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    msk = lbl[:, 1:] != -100
    return (tok * msk).sum(-1)

output_dir = CHECKPOINTS_DATA_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MODEL_NAME}_haikufy_lora_model"
output_dir.mkdir(parents=True, exist_ok=True)

topics = [
    "nature",
    "love",
    "life",
    "death",
    "seasons",
    "time",
    "dreams",
    "memories",
    "learning",
    "happiness",
    "sadness",
    "friendship",
    "bardiwac",
    "morning",
    "thinking",
]

test_prompts = [
    generate_query(topic) for topic in topics
]

def generate_haiku(prompt, max_length=28):
    # Add a newline after the prompt to match training data
    prompt = prompt.rstrip() + "\n"
    inputs = tkz(prompt, return_tensors='pt', padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        output = plc.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tkz.eos_token_id,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=3,
            eos_token_id=tkz.eos_token_id
        )
    response = tkz.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Training function
def train_step(batch):
    # Accepts a batch dict with 'query', 'positive', 'negative' lists
    qrys = batch['query']
    poss = batch['positive']
    negs = batch['negative']
    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        ids_pos, atn_msk_pos, lbl_pos = tokenise(qrys, poss)
        ids_neg, atn_msk_neg, lbl_neg = tokenise(qrys, negs)
        with torch.no_grad():
            logp_ref_pos = sum_logp(ref, ids_pos, atn_msk_pos, lbl_pos)
            logp_ref_neg = sum_logp(ref, ids_neg, atn_msk_neg, lbl_neg)
        logp_plc_pos = sum_logp(plc, ids_pos, atn_msk_pos, lbl_pos)
        logp_plc_neg = sum_logp(plc, ids_neg, atn_msk_neg, lbl_neg)
        delta_pos = logp_plc_pos - logp_ref_pos
        delta_neg = logp_plc_neg - logp_ref_neg
        mrg = delta_pos - delta_neg
        loss = -torch.log(torch.sigmoid(beta * mrg)).mean()
    return loss

# Initialize wandb
wandb.init(project="haikufy-lora-gpt2", name=f"{MODEL_NAME}_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Train the model
print("Starting training...")
num_epochs = 3

# prints debug metrics for text
# lines, syllables per line, and total syllables 
def analyze_text(text):
    # Split only on '\n'
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    syllables_per_line = [textstat.syllable_count(line) for line in lines]
    total_syllables = sum(syllables_per_line)
    is_haiku = len(lines) == 3 and syllables_per_line == [5, 7, 5]
    return is_haiku, lines, syllables_per_line, total_syllables

total_haikus = 0

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_loss = 0
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        optm.zero_grad()
        loss = train_step(batch)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optm)
            scaler.update()
        else:
            loss.backward()
            optm.step()
        epoch_loss += loss.item() * len(batch['query'])
        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item(), "epoch": epoch+1, "batch_idx": batch_idx+1})
        # Update tqdm postfix with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Run evaluation every 50 batches
        if (batch_idx + 1) % 18 == 0:
            print(f"\n[Eval] Running test prompt evaluation at batch {batch_idx+1}...")
            results = []
            haiku_count = 0
            for prompt in test_prompts:
                response = generate_haiku(prompt)
                is_haiku, lines, syllables_per_line, total_syllables = analyze_text(response)
                results.append(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables per line: {syllables_per_line}\nTotal syllables: {total_syllables}\nIs haiku: {is_haiku}\n")
                print(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables per line: {syllables_per_line}\nTotal syllables: {total_syllables}\nIs haiku: {is_haiku}\n")
                if is_haiku:
                    total_haikus += 1
                    haiku_count += 1
            print(f"[Eval] {haiku_count}/{len(test_prompts)} outputs are valid haikus.")
            eval_file = output_dir / f"test_prompt_eval_batch{batch_idx+1}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write("\n".join(results))
                f.write(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.\n")
            print(f"[Eval] Test prompt evaluation logs saved to {eval_file}")
            print(f"[Eval] Total: {total_haikus}/{len(test_prompts)} outputs are valid haikus.")

    avg_epoch_loss = epoch_loss / len(dataset)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_epoch_loss:.4f}")
    # Log epoch loss to wandb
    wandb.log({"avg_epoch_loss": avg_epoch_loss, "epoch": epoch+1, "epoch_time": epoch_time})

# Save the trained model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = CHECKPOINTS_DATA_DIR / f"{timestamp}_{MODEL_NAME}_haikufy_lora_model"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the adapter weights separately - much smaller than full model
plc.save_pretrained(output_dir)
tkz.save_pretrained(output_dir)
print(f"LoRA model saved to {output_dir}")

# Final evaluation after training
results = []
haiku_count = 0
for prompt in test_prompts:
    response = generate_haiku(prompt)
    is_haiku, lines, syllables_per_line, total_syllables = analyze_text(response)
    results.append(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables per line: {syllables_per_line}\nTotal syllables: {total_syllables}\nIs haiku: {is_haiku}\n")
    print(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables per line: {syllables_per_line}\nTotal syllables: {total_syllables}\nIs haiku: {is_haiku}\n")
    if is_haiku:
        haiku_count += 1
        total_haikus += 1
print(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.")
output_file = output_dir / "test_prompt_eval.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
    f.write(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.\n")
    print(f"Final evaluation completed. Total haikus: {total_haikus}/{len(test_prompts)} outputs are valid haikus.")
print(f"Test prompt evaluation logs saved to {output_file}")

wandb.finish()