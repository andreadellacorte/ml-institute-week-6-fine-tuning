import torch
import transformers
import time
import re
from datasets import load_dataset
import random
from datetime import datetime
from pathlib import Path

# Import from the parent package directly
from config import CHECKPOINTS_DATA_DIR

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

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plc.to(device)
ref.to(device)
print(f"Using device: {device}")

# Setup optimizer
optm = torch.optim.Adam(plc.parameters(), lr=1e-5)
beta = 0.1

# Load haiku dataset
dataset = load_dataset("davanstrien/haiku_dpo", split="train")
print(f"Loaded {len(dataset)} examples from haiku_dpo dataset")
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

    optm.zero_grad()
    loss.backward()
    optm.step()

    return loss.item()

# Generation function
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

# Train the model
print("Starting training...")
num_epochs = 3
batch_size = 8
total_examples = min(100, len(dataset))  # Limit to 100 examples for faster training

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_loss = 0
    
    # Shuffle indices for this epoch
    indices = random.sample(range(len(dataset)), total_examples)
    
    for i in range(0, total_examples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_loss = 0
        
        for idx in batch_indices:
            example = dataset[idx]
            query = clean_query(example['question'])
            
            # Use the chosen response as positive
            positive = example['chosen']
            
            # Generate a negative response using the reference model
            with torch.no_grad():
                # Make sure we have a non-empty query
                if not query.strip():
                    query = "Write a response"
                
                # Tokenize and ensure we have proper input format
                input_text = query.strip()
                inputs = tkz(input_text, return_tensors='pt', padding=True)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                # Check if we have valid input
                if input_ids.size(1) > 0:
                    neg_output = ref.generate(
                        input_ids, 
                        attention_mask=attention_mask,
                        max_length=50, 
                        min_length=10,  # Ensure we generate something
                        num_return_sequences=1,
                        pad_token_id=tkz.eos_token_id,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.95,
                    )
                    negative = tkz.decode(neg_output[0][input_ids.size(1):], skip_special_tokens=True)
                else:
                    # Fallback for empty inputs
                    negative = "This is a generic response that's not too long."
            
            # Train on this example
            loss = train_step(query, positive, negative)
            batch_loss += loss
            
        avg_batch_loss = batch_loss / len(batch_indices)
        epoch_loss += batch_loss
        
        if i % 50 == 0 or i + batch_size >= total_examples:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size}, Loss: {avg_batch_loss:.4f}")
    
    avg_epoch_loss = epoch_loss / total_examples
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_epoch_loss:.4f}")

# Save the trained model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = CHECKPOINTS_DATA_DIR / f"haikufy_model_{timestamp}"
plc.save_pretrained(output_dir)
tkz.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Test the model with some prompts
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