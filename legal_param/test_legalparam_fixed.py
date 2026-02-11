from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os

def main():
    """LegalParam test with correct prompt formatting using apply_chat_template"""
    
    # Disable warnings
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    model_name = "bharatgenai/LegalParam"
    
    print("Loading LegalParam model with correct formatting...")
    
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
    
    # Test queries
    test_queries = [
        "What is the Right to Information Act?",
        "What are the key provisions of the Indian Constitution?",
        "Explain the concept of judicial review in India."
    ]
    
    print(f"\n{'='*80}")
    print("Testing LegalParam with apply_chat_template format")
    print(f"{'='*80}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 50)
        
        try:
            # Format using apply_chat_template (correct format for Param models)
            conversation = [
                {"role": "system", "content": "You are a helpful legal assistant specializing in Indian law."},
                {"role": "user", "content": query}
            ]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                conversation=conversation,
                return_tensors="pt",
                add_generation_prompt=True
            )
            
            inputs = inputs.to(model.device)
            
            print("Template applied successfully!")
            
            # Generate with improved parameters
            with torch.no_grad():
                output = model.generate(
                    inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Extract only generated tokens (exclude prompt)
            generated_tokens = output[0][inputs.shape[-1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"LegalParam: {generated_text}")
            
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to simple format if apply_chat_template fails
            try:
                print("Trying fallback simple format...")
                simple_prompt = f"Question: {query}\nAnswer:"
                inputs = tokenizer(simple_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.8,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2
                    )
                
                full_response = tokenizer.decode(output[0], skip_special_tokens=True)
                if "Answer:" in full_response:
                    answer = full_response.split("Answer:")[-1].strip()
                else:
                    answer = full_response
                
                print(f"Fallback response: {answer}")
                
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
        
        print("-" * 50)
    
    # Interactive mode
    try:
        interactive = input(f"\n{'='*50}\nTry interactive mode? (y/n): ").lower().strip()
        
        if interactive == 'y':
            print(f"\n{'='*60}")
            print("LegalParam Interactive Mode - Type 'exit' to quit")
            print(f"{'='*60}")
            
            while True:
                try:
                    user_input = input("\nYour legal question: ").strip()
                    if user_input.lower() in {"exit", "quit"}:
                        break
                    
                    if not user_input:
                        continue
                    
                    # Use chat template format
                    conversation = [
                        {"role": "system", "content": "You are a helpful legal assistant specializing in Indian law."},
                        {"role": "user", "content": user_input}
                    ]
                    
                    inputs = tokenizer.apply_chat_template(
                        conversation=conversation,
                        return_tensors="pt",
                        add_generation_prompt=True
                    ).to(model.device)
                    
                    with torch.no_grad():
                        output = model.generate(
                            inputs,
                            max_new_tokens=250,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=50,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )
                    
                    generated_tokens = output[0][inputs.shape[-1]:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    print(f"\nLegalParam: {generated_text}")
                    print("-" * 40)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print("If responses are still repetitive, try:")
    print("1. Lowering temperature (0.5-0.6)")
    print("2. Increasing repetition_penalty (1.2-1.5)")
    print("3. Using different prompts")

if __name__ == "__main__":
    main()