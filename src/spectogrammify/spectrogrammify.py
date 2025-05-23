import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments
from spectogrammify.dataset import UrbanSoundDataset
from PIL import Image
import pickle
import numpy as np

# Load MiniGPT-4 and processor
minigpt4_model_id = "Vision-CAIR/MiniGPT-4"
model = AutoModelForVision2Seq.from_pretrained(minigpt4_model_id)
processor = AutoProcessor.from_pretrained(minigpt4_model_id)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# Prepare UrbanSoundDataset for training
train_dataset = UrbanSoundDataset(fold=1, augment=False, num_augmentations=0)

# Custom collate function for MiniGPT-4 (image + text)
def collate_fn(batch):
    images = [Image.fromarray(item["spectrogram_image"]) for item in batch]
    texts = [item["label"] for item in batch]
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return inputs

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-4,
    report_to=[],
)

# DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    tokenizer=processor,
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