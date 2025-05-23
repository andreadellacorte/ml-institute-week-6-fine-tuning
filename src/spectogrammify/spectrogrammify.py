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

# Custom Trainer to filter out unexpected batch keys
class FilteredTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Only keep keys that are in the model's forward signature
        model_forward = model.forward if hasattr(model, 'forward') else model.__call__
        sig = signature(model_forward)
        valid_keys = set(sig.parameters.keys())
        filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
        outputs = model(**filtered_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

# Load BLIP-1 model and processor
blip1_model_id = "Salesforce/blip-image-captioning-large"
model = BlipForConditionalGeneration.from_pretrained(blip1_model_id)
processor = BlipProcessor.from_pretrained(blip1_model_id)

# 1. Reduce batch size to 1
# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()
# 4. Clear GPU memory before training
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Prepare UrbanSoundDataset for training
train_dataset = UrbanSoundDataset(fold=1, augment=False, num_augmentations=0, max_length=200)

# Custom collate function for BLIP-1 (image + text)
def collate_fn(batch):
    images = [Image.fromarray(item["spectrogram_image"]) for item in batch]
    texts = [item["label"] for item in batch]
    # The processor will handle both input and target tokenization
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, max_length=16)
    # Replace padding token id with -100 for loss masking
    if "labels" in inputs:
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
    per_device_train_batch_size=1,  # changed from 2 to 1
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-4,
    report_to=[],
    remove_unused_columns=False,
    # 3. Use fp16 (mixed precision)
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
    # Use a generic prompt for BLIP-2 (can be empty or a question)
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