from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "bharatgenai/LegalParam"

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False   # 👈 THIS FIXES IT
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
)
model.to(device)
model.eval()

# Example legal query
user_input = "What steps should a farmer take to legally transfer agricultural land ownership?"

prompt = f"<user>\n{user_input}<assistant>\n"

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,   # IMPORTANT for MPS stability
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
