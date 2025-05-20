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

# Import from the parent package directly
from config import CHECKPOINTS_DATA_DIR

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
    'gpt2': 64,
    'gpt2-medium': 32,
    'gpt2-large': 16,
    'gpt2-xl': 4,
}

# Load models
# Use a larger model for better haiku generation
MODEL_NAME = 'gpt2-medium'  # You can change to mistralai/Mistral-7B-v0.1 or another large model if desired
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
optm = torch.optim.Adam(plc.parameters(), lr=1e-4)
beta = 0.1

# Custom Dataset for Haiku DPO CSV
data_csv_path = Path('data/processed/haiku_dpo/haiku_dpo_processed.csv')

class HaikuDPODataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expecting columns: question, chosen
                self.samples.append({
                    'query': row['query'],
                    'positive': row['positive'],
                    'negative': row['negative']  # Optional negative example
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

test_prompts = [
    "Write about nature",
    "Describe the mountains",
    "Tell me about the ocean",
    "Reflect on the changing seasons",
    "Express feelings about love",
    "Tell me about work",
    "Write about the stars",
    "Describe a sunset",
    "Write about the moon",
    "Express feelings about friendship",
    "Write about the city",
]

def generate_haiku(prompt, max_length=50):
    # Prepend system instruction to bias haiku output
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
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    response = tkz.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

def is_haiku(text):
    # Treats both newlines and commas as line breaks for haiku detection
    # Checks if the text is a 3-line haiku with 5-7-5 syllable structure and returns (is_haiku, syllable_counts)
    # Split on newlines or commas
    lines = [l.strip() for l in re.split(r'[\n,]', text.strip()) if l.strip()]
    if len(lines) < 3:
        return False, [textstat.syllable_count(line) for line in lines] + [0]*(3-len(lines))
    syllables = [textstat.syllable_count(line) for line in lines[:3]]
    return syllables == [5, 7, 5], syllables

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

# Train the model
print("Starting training...")
num_epochs = 5

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_loss = 0
    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
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

        # Run evaluation every 250 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"\n[Eval] Running test prompt evaluation at batch {batch_idx+1}...")
            results = []
            haiku_count = 0
            for prompt in test_prompts:
                response = generate_haiku(prompt)
                is_hk, sylls = is_haiku(response)
                results.append(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables: {sylls}\nIs haiku: {is_hk}\n")
                print(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables: {sylls}\nIs haiku: {is_hk}\n")
                if is_hk:
                    haiku_count += 1
            print(f"[Eval] {haiku_count}/{len(test_prompts)} outputs are valid haikus.")
            eval_file = output_dir / f"test_prompt_eval_batch{batch_idx+1}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write("\n".join(results))
                f.write(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.\n")
            print(f"[Eval] Test prompt evaluation logs saved to {eval_file}")

    avg_epoch_loss = epoch_loss / len(dataset)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_epoch_loss:.4f}")

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
    is_hk, sylls = is_haiku(response)
    results.append(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables: {sylls}\nIs haiku: {is_hk}\n")
    print(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables: {sylls}\nIs haiku: {is_hk}\n")
    if is_hk:
        haiku_count += 1
print(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.")
output_file = output_dir / "test_prompt_eval.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
    f.write(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.\n")
print(f"Test prompt evaluation logs saved to {output_file}")

print("\nTesting the model with prompts:")
results = []
haiku_count = 0
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    response = generate_haiku(prompt)
    is_hk, sylls = is_haiku(response)
    print(f"Response:\n{response}")
    print(f"Syllables: {sylls}")
    print(f"Is haiku: {is_hk}")
    results.append(f"Prompt: {prompt}\nResponse:\n{response}\nSyllables: {sylls}\nIs haiku: {is_hk}\n")
    if is_hk:
        haiku_count += 1
print(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.")
# Save evaluation logs
output_file = output_dir / "test_prompt_eval.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
    f.write(f"\n{haiku_count}/{len(test_prompts)} outputs are valid haikus.\n")
print(f"Test prompt evaluation logs saved to {output_file}")