import streamlit as st
import torch
import transformers
from peft import PeftModel, PeftConfig
from pathlib import Path

st.title("Haikufy: GPT-2 Haiku Generator")

# Model selector
model_options = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl"
]
selected_model = st.selectbox("Select base model:", model_options, index=0)

# Update model loading functions to accept model_name
@st.cache_resource
def load_base_model(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

@st.cache_resource
def load_lora_model(checkpoint_dir, model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Generation function
def generate_haiku(model, tokenizer, device, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# UI
prompt = st.text_area("Enter a prompt for your haiku:", "A memory about nature")

# Update checkpoint list based on selected model
if "lora_checkpoint" not in st.session_state or "lora_model" not in st.session_state or st.session_state.lora_model != selected_model:
    checkpoints_root = Path("data/checkpoints")
    lora_dirs = sorted([
        d for d in checkpoints_root.glob(f"*_" + selected_model + "_haikufy_lora_model") if d.is_dir()
    ])
    st.session_state.lora_dirs = [str(d) for d in lora_dirs]
    st.session_state.lora_checkpoint = st.session_state.lora_dirs[-1] if lora_dirs else ""
    st.session_state.lora_model = selected_model

lora_checkpoint = st.selectbox(
    "LoRA checkpoint directory:",
    st.session_state.lora_dirs if "lora_dirs" in st.session_state else [],
    index=(st.session_state.lora_dirs.index(st.session_state.lora_checkpoint) if "lora_dirs" in st.session_state and st.session_state.lora_checkpoint in st.session_state.lora_dirs else 0)
) if st.session_state.get("lora_dirs") else st.text_input("LoRA checkpoint directory:", st.session_state.lora_checkpoint)

if st.button("Generate Haiku"):
    with st.spinner("Loading models and generating haikus..."):
        base_model, base_tokenizer, base_device = load_base_model(selected_model)
        lora_model, lora_tokenizer, lora_device = load_lora_model(lora_checkpoint, selected_model)
        base_haiku = generate_haiku(base_model, base_tokenizer, base_device, prompt)
        lora_haiku = generate_haiku(lora_model, lora_tokenizer, lora_device, prompt)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base GPT-2:")
        st.markdown(f'<div style="white-space: pre-wrap; word-break: break-word; border: 2px solid #fff; border-radius: 8px; padding: 12px; background: #f9f9f9; color: #222; min-height: 100px;">{base_haiku}</div>', unsafe_allow_html=True)
    with col2:
        st.subheader("Fine-tuned GPT-2:")
        st.markdown(f'<div style="white-space: pre-wrap; word-break: break-word; border: 2px solid #fff; border-radius: 8px; padding: 12px; background: #f9f9f9; color: #222; min-height: 100px;">{lora_haiku}</div>', unsafe_allow_html=True)

st.markdown("---")
if st.button("Play Haiku Inspiration Music 🎵"):
    st.video("https://www.youtube.com/watch?v=7hNMdWsIB3w")