from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "bharatgenai/LegalParam"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("LegalParam terminal chatbot")
print("Type 'exit' or Ctrl+C to quit\n")

history = ""

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        prompt = history + f"\nUser: {user_input}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("Assistant:")[-1].strip()

        print(f"Bot: {answer}\n")

        history += f"\nUser: {user_input}\nAssistant: {answer}"

    except KeyboardInterrupt:
        break
