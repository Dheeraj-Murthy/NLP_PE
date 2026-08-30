from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os

def setup_environment():
    """Set up environment for better performance"""
    # Disable HF token warning
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Set memory efficient attention for Mac/Apple Silicon
    if torch.backends.mps.is_available():
        torch.backends.mps.enabled = True
        print("Using MPS (Apple Silicon) backend")
    elif torch.cuda.is_available():
        print("Using CUDA backend")
    else:
        print("Using CPU backend")

def load_model_and_tokenizer():
    """Load LegalParam model and tokenizer with proper configuration"""
    model_name = "bharatgenai/LegalParam"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Fix rope_scaling configuration issue
    if getattr(config, "rope_scaling", None) is None:
        config.rope_scaling = {}
    
    if "type" not in config.rope_scaling:
        config.rope_scaling["type"] = "linear"
        config.rope_scaling["factor"] = 1.0
    
    print("Loading model...")
    # Determine appropriate device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Using device: {device} with dtype: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print("Model loaded successfully!")
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, max_new_tokens=256):
    """Generate response from the model with proper parameters"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            repetition_penalty=1.1
        )
    
    # Decode and clean up the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<assistant>" in response:
        assistant_response = response.split("<assistant>")[-1].strip()
    elif "Assistant:" in response:
        assistant_response = response.split("Assistant:")[-1].strip()
    else:
        assistant_response = response.strip()
    
    return assistant_response

def main():
    """Main function to demonstrate LegalParam usage"""
    setup_environment()
    
    try:
        model, tokenizer, device = load_model_and_tokenizer()
        
        # Example legal queries
        queries = [
            "What steps should a farmer take to legally transfer agricultural land ownership in India?",
            "What are the key provisions of the Right to Information Act?",
            "Explain the concept of judicial review in Indian constitutional law."
        ]
        
        print("\n" + "="*80)
        print("LegalParam - Legal AI Assistant")
        print("="*80)
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 50)
            
            # Format prompt according to the model's expected format
            prompt = f"<user>\n{query}<assistant>\n"
            
            try:
                response = generate_response(model, tokenizer, prompt, device)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error generating response: {e}")
                continue
            
            print("-" * 50)
        
        # Interactive mode
        print("\n" + "="*80)
        print("Interactive Mode - Type 'exit' to quit")
        print("="*80)
        
        while True:
            try:
                user_input = input("\nYour query: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                
                if not user_input:
                    continue
                
                prompt = f"<user>\n{user_input}<assistant>\n"
                response = generate_response(model, tokenizer, prompt, device)
                print(f"\nLegalParam: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())