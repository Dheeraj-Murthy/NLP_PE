from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os

def main():
    """Simple test of LegalParam with a single query"""
    
    # Disable warnings
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    model_name = "bharatgenai/LegalParam"
    
    print("Loading LegalParam model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Load and fix config
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Fix rope_scaling
    if getattr(config, "rope_scaling", None) is None:
        config.rope_scaling = {}
    if "type" not in config.rope_scaling:
        config.rope_scaling["type"] = "linear"
        config.rope_scaling["factor"] = 1.0
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Set up tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test query
    query = "What is the Right to Information Act?"
    
    # Try different prompt formats
    prompts = [
        f"Question: {query}\nAnswer:",
        f"<user>\n{query}\n<assistant>\n",
        f"User: {query}\nAssistant:",
        query
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: Using prompt format: {prompt[:50]}...")
        print(f"{'='*60}")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,  # Use greedy for testing
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )
            
            # Decode response
            full_response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if len(inputs.input_ids[0]) < len(output[0]):
                generated_tokens = output[0][len(inputs.input_ids[0]):]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = full_response
            
            print(f"Generated: {generated_text}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Test completed!")
    
    # Interactive test (optional)
    try_interactive = input(f"\n{'='*60}\nTry interactive mode? (y/n): ").lower().strip()
    
    if try_interactive == 'y':
        print(f"\n{'='*60}")
        print("Interactive Mode - Type 'exit' to quit")
        print(f"{'='*60}")
        
        while True:
            try:
                user_input = input("\nYour query: ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    break
                
                prompt = f"Question: {user_input}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                generated_tokens = output[0][len(inputs.input_ids[0]):]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                print(f"LegalParam: {generated_text}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()