import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import pickle
import numpy as np

# Paths
DATA_DIR = Path("data/processed/urbansound8k")

# Example: Use a ViT-GPT2 multimodal model (replace with a real multimodal model that supports spectrograms)
BASE_MODEL = "openai/clip-vit-base-patch16"

# Load data: X = spectrogram (np.array), y = label (text)
def load_spectrogram_data():
    X, y = [], []
    for pkl_file in DATA_DIR.glob("*_spectro.pkl"):
        label = pkl_file.stem.replace("_spectro", "").split("-")[-1]  # crude: expects label in filename
        with open(pkl_file, "rb") as f:
            spectro = pickle.load(f)
        X.append(spectro)
        # You should map label to text, e.g. 'dog_bark' -> 'the spectrogram of a dog barking'
        y.append(f"the spectrogram of a {label}")
    return X, y

# Dummy tokenizer/model for demonstration (replace with a real multimodal model/tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# Prepare dataset
def preprocess(X, y):
    # Flatten spectrogram and convert to string for toy example
    # Replace with real multimodal input pipeline
    return [{
        "input": str(x.flatten().tolist()),
        "label": label
    } for x, label in zip(X, y)]

X, y = load_spectrogram_data()
dataset = preprocess(X, y)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-4,
    report_to=[],
)

def collate_fn(batch):
    # Dummy: just tokenize label
    return tokenizer([b["label"] for b in batch], padding=True, truncation=True, return_tensors="pt")

# DPOTrainer (dummy, for demonstration)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset[:10],
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

# Train
trainer.train()

# Inference example
def predict_spectrogram(spectro):
    # Dummy: flatten and convert to string
    inputs = tokenizer([str(spectro.flatten().tolist())], padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1)
    return pred