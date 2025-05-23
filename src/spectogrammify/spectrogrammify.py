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
from datetime import datetime

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

# Set up output directory with timestamp
output_dir = Path('data/checkpoints') / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_spectrogrammify"
output_dir.mkdir(parents=True, exist_ok=True)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=str(output_dir),
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

# Save the LoRA-adapted model and processor after training
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f'Model and processor saved to {output_dir}')

# Inference example

def predict_spectrogram(image, text):
    inputs = processor(images=[image], text=[text], return_tensors="pt")
    outputs = model.generate(**inputs)
    # Post-process outputs as needed
    return outputs

# Evaluation: Submit some spectrograms from the dataset to the trained model and check the response vs expected class
num_eval_samples = 5
correct = 0
results = []

for i in range(num_eval_samples):
    sample = train_dataset[i]
    image = Image.fromarray(sample["spectrogram_image"])
    expected_label = sample["label"]
    # Use a generic prompt for MiniGPT-4 (can be empty or a question)
    prompt = "What is this a spectrogram of?"
    inputs = processor(images=[image], text=[prompt], return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Check if the expected class (e.g. 'dog', 'car', etc) is in the response
    class_name = expected_label.replace('the spectrogram of a ', '').strip().lower()
    match = class_name in response.lower()
    results.append({
        'expected': class_name,
        'response': response,
        'match': match
    })
    if match:
        correct += 1

accuracy = correct / num_eval_samples
with open(output_dir / "eval_classification.txt", "w") as f:
    for r in results:
        f.write(f"Expected: {r['expected']} | Response: {r['response']} | Match: {r['match']}\n")
    f.write(f"\nAccuracy: {accuracy:.2f}\n")
print(f"Evaluation complete. Accuracy: {accuracy:.2f}. Results saved to {output_dir}/eval_classification.txt")