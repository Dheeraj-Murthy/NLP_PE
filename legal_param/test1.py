
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "bharatgenai/LegalParam"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Example legal query
user_input = "What steps should a farmer take to legally transfer agricultural land ownership?"

# 3 types of prompt
# 1. Generic QA
# 2. Context based QA (context as part of prompt)
# 3. Multi-turn conversation

# Based on your requirements use the type of prompt (refere the above examples)
prompt = f"<user>\n{user_input}<assistant>\n"
# prompt = f"<user>\n{user_or_rag_context}\n<assistant>\n"
# prompt = f"<user>\n{user_input1}\n<assistant>\n{user_input2}\n<user> {user_input3} <assistant>..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
