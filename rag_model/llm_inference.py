import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import gc

class QwenInference:
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = False
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None
    ) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        do_sample = do_sample or self.do_sample
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            generated_tokens = outputs.sequences[0][len(inputs[0]):]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Dict[str, Any]:
        memory_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_memory_percent": None
        }
        
        if torch.cuda.is_available():
            memory_info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            memory_info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
            memory_info["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated()
            
            for i in range(torch.cuda.device_count()):
                memory_info[f"device_{i}_memory_allocated"] = torch.cuda.memory_allocated(i)
                memory_info[f"device_{i}_memory_reserved"] = torch.cuda.memory_reserved(i)
        
        try:
            import psutil
            memory_info["cpu_memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        
        return memory_info
    
    def estimate_generation_cost(self, prompt: str) -> Dict[str, Any]:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        prompt_tokens = len(self.tokenizer.encode(prompt))
        estimated_output_tokens = self.max_new_tokens
        total_tokens = prompt_tokens + estimated_output_tokens
        
        return {
            "prompt_tokens": prompt_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "total_tokens": total_tokens,
            "estimated_vram_gb": total_tokens * 2 / (1024**3)
        }
    
    def is_model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
    
    def reload_model(self):
        self.unload_model()
        self._load_model()
    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer  
            self.tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ Model unloaded")

if __name__ == "__main__":
    inference = QwenInference()
    
    print("\n" + "=" * 50)
    print("Model Memory Info:")
    memory_info = inference.get_memory_info()
    for key, value in memory_info.items():
        if isinstance(value, (int, float)):
            if "memory" in key and value > 1024**3:
                print(f"{key}: {value / (1024**3):.2f} GB")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("Testing with sample prompt...")
    
    sample_prompt = """You are a legal assistant. Answer ONLY using the provided context. If the answer is not present, say: "Not found in the provided cases."

Context:
[1] The Supreme Court held that educational institutions must follow due process when implementing fee structures. The Court emphasized that any increase in fees must be reasonable and proportionate to the services provided.

Source: ABC University v. State (Supreme Court of India, 2019, ¶23)
Section: judgment

Question:
What did the Supreme Court say about educational fees?

Answer:"""
    
    cost_estimate = inference.estimate_generation_cost(sample_prompt)
    print(f"Prompt tokens: {cost_estimate['prompt_tokens']}")
    print(f"Estimated total tokens: {cost_estimate['total_tokens']}")
    print(f"Estimated VRAM: {cost_estimate['estimated_vram_gb']:.2f} GB")
    
    print("\nGenerating response...")
    response = inference.generate_response(sample_prompt)
    print(f"Response: {response}")
    
    print("\n" + "=" * 50)
    print("Testing no-answer scenario...")
    
    no_context_prompt = """You are a legal assistant. Answer ONLY using the provided context. If the answer is not present, say: "Not found in the provided cases."

Context:
[1] The Court discussed matters related to contract law and commercial disputes.

Source: XYZ Corp v. ABC Ltd (Delhi High Court, 2021, ¶15)  
Section: judgment

Question:
What are the regulations for educational institutions?

Answer:"""
    
    response = inference.generate_response(no_context_prompt)
    print(f"Response: {response}")
