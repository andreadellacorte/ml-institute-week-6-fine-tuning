import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from dataset import UrbanSoundDataset
from PIL import Image
import pickle
import numpy as np
from datetime import datetime
from inspect import signature
import warnings

warnings.filterwarnings("ignore")

# Custom Trainer to filter out unexpected batch keys
class FilteredTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_forward = model.forward if hasattr(model, 'forward') else model.__call__
        sig = signature(model_forward)
        valid_keys = set(sig.parameters.keys())
        filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
        outputs = model(**filtered_inputs)
        # Try to get loss from outputs, else compute manually
        loss = None
        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            # Compute loss manually if not present
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            labels = filtered_inputs.get("labels", None)
            if labels is not None:
                # Use CrossEntropyLoss, ignoring -100
                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # logits: (batch, seq, vocab), labels: (batch, seq)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                raise ValueError("No loss in model output and no labels to compute loss.")
        return (loss, outputs) if return_outputs else loss

# Load BLIP-1 model and processor
blip1_model_id = "Salesforce/blip-image-captioning-large"
model = BlipForConditionalGeneration.from_pretrained(blip1_model_id)
processor = BlipProcessor.from_pretrained(blip1_model_id, use_fast=True)

# 1. Reduce batch size to 1
# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()
# 4. Clear GPU memory before training
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Prepare UrbanSoundDataset for training
train_dataset = UrbanSoundDataset(fold=1, augment=False, num_augmentations=0, max_length=1000)

# Custom collate function for BLIP-1 (image + text)
def collate_fn(batch):
    images = [Image.fromarray(item["spectrogram_image"]) for item in batch]
    texts = [item["label"] for item in batch]
    # Ensure no empty labels
    for t in texts:
        if not t or not isinstance(t, str) or t.strip() == "":
            raise ValueError(f"Empty or invalid label in batch: {texts}")
    # Use text for BLIP-1 to generate labels
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True, max_length=16)
    if "labels" not in inputs or inputs["labels"] is None:
        # Try to manually tokenize the targets as labels
        labels = processor.tokenizer(texts, padding="longest", truncation=True, max_length=16, return_tensors="pt").input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        if labels is None:
            raise ValueError(f"Processor did not return labels for batch: {texts}")
    else:
        labels = inputs["labels"]
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
    return inputs
# Set up output directory with timestamp
output_dir = Path('data/checkpoints') / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_spectrogrammify"
output_dir.mkdir(parents=True, exist_ok=True)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=64,  # increased from 1 to 4
    num_train_epochs=5,  # increased from 1 to 5
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-4,
    report_to=[],
    remove_unused_columns=False,
    fp16=True,
)

# Trainer
trainer = FilteredTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=collate_fn,
)

# Train
trainer.train()

# Save the model and processor after training
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f'Model and processor saved to {output_dir}')

# Inference example

# Move model to device for inference
inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(inference_device)

def predict_spectrogram(image, text, device=None):
    if device is None:
        device = next(model.parameters()).device
    inputs = processor(images=[image], text=[text], return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return outputs

# Evaluation: Submit some spectrograms from the dataset to the trained model and check the response vs expected class
num_eval_samples = 100
results, accuracy = evaluate_model(model, processor, train_dataset, inference_device, num_samples=num_eval_samples)
with open(output_dir / "eval_classification.txt", "w") as f:
    for r in results:
        f.write(f"Expected: {r['expected']} | Response: {r['response']} | Similarity: {r['similarity']:.2f} | Match: {r['match']}\n")
    f.write(f"\nAccuracy: {accuracy:.2f}\n")

print(f"Evaluation complete. Accuracy: {accuracy:.2f}. Results saved to {output_dir}/eval_classification.txt")