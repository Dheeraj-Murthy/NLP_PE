#!/usr/bin/env python
"""Bare minimum test - just run Qwen model with a simple prompt."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded. Enter a prompt:")

prompt = """<|im_start|>system
prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|
<|im_start|>assistant
"""

inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print("\nResponse:")
print(response)