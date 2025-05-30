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

# Load models
tkz = transformers.AutoTokenizer.from_pretrained('gpt2')
# Set padding token to be the eos token
tkz.pad_token = tkz.eos_token
plc = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
ref = transformers.AutoModelForCausalLM.from_pretrained('gpt2')

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
data_csv_path = Path('data/processed/haiku_dpo/haikus.csv')

class HaikuDPODataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expecting columns: question, chosen
                if row.get('query') and row.get('positive'):
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
batch_size = 64  # Lowered for better GPU utilization and stability
num_workers = 2 if torch.cuda.is_available() else 0
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

# Add mixed precision support
if torch.cuda.is_available():
    scaler = torch.amp.GradScaler(device='cuda')
else:
    scaler = None

print(f"Setup completed in {time.time() - start_time:.2f}s")

def clean_query(query):
    """Remove 'a haiku' or 'haiku' from the query"""
    return re.sub(r'\b(a\s+)?haiku\b', '', query, flags=re.IGNORECASE).strip()

def tokenise(qry, res):
    qry_ids = tkz(qry, return_tensors='pt', add_special_tokens=False).input_ids.to(device).long()
    res_ids = tkz(res, return_tensors='pt', add_special_tokens=False).input_ids.to(device).long()
    acc_ids = torch.cat([qry_ids, res_ids], dim=1)
    atn_msk = torch.ones_like(acc_ids).long()
    lbl_ids = acc_ids.clone()
    lbl_ids[:, :qry_ids.size(-1)] = -100
    return acc_ids, atn_msk, lbl_ids

def sum_logp(model, ids, msk, lbl):
    out = model(input_ids=ids, attention_mask=msk)
    log = out.logits.log_softmax(-1)[:, :-1]
    tgt = lbl[:, 1:].masked_fill(lbl[:, 1:] == -100, 0)
    tok = log.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    msk = lbl[:, 1:] != -100
    return (tok * msk).sum(-1)

# Training function
def train_step(qry, pos, neg):
    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        ids_pos, atn_msk_pos, lbl_pos = tokenise(qry, pos)
        ids_neg, atn_msk_neg, lbl_neg = tokenise(qry, neg)

        with torch.no_grad():
            logp_ref_pos = sum_logp(ref, ids_pos, atn_msk_pos, lbl_pos)
            logp_ref_neg = sum_logp(ref, ids_neg, atn_msk_neg, lbl_neg)

        logp_plc_pos = sum_logp(plc, ids_pos, atn_msk_pos, lbl_pos)
        logp_plc_neg = sum_logp(plc, ids_neg, atn_msk_neg, lbl_neg)
        
        delta_pos = logp_plc_pos - logp_ref_pos
        delta_neg = logp_plc_neg - logp_ref_neg

        mrg = delta_pos - delta_neg
        loss = -torch.log(torch.sigmoid(beta * mrg))
    return loss

# Train the model
print("Starting training...")
num_epochs = 5

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_loss = 0
    for batch in train_loader:
        batch_loss = 0
        optm.zero_grad()
        for i in range(len(batch['query'])):
            query = clean_query(batch['query'][i])
            positive = batch['positive'][i]
            negative = batch['negative'][i]
            loss = train_step(query, positive, negative)
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            batch_loss += loss.item() if not scaler else loss.detach().cpu().item()
        if scaler:
            scaler.step(optm)
            scaler.update()
        else:
            optm.step()
        avg_batch_loss = batch_loss / len(batch['query'])
        epoch_loss += batch_loss
    avg_epoch_loss = epoch_loss / len(dataset)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_epoch_loss:.4f}")

# Save the trained model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = CHECKPOINTS_DATA_DIR / f"haikufy_lora_model_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the adapter weights separately - much smaller than full model
plc.save_pretrained(output_dir)
tkz.save_pretrained(output_dir)
print(f"LoRA model saved to {output_dir}")

# Test the model with some prompts
def generate_haiku(prompt, max_length=50):
    # Create proper input with attention mask
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

test_prompts = [
    "Write about nature",
    "Describe the mountains",
    "Tell me about the ocean",
    "Reflect on the changing seasons",
    "Express feelings about love"
]

print("\nTesting the model with prompts:")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    response = generate_haiku(prompt)
    print(f"Response:\n{response}")