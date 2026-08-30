# LegalParam Setup Guide

## Overview
LegalParam is a domain-specialized large language model fine-tuned from Param-1-2.9B-Instruct on Indian legal data. This guide will help you set it up and run it on your system.

## Prerequisites
- Python 3.8 or higher
- Git
- At least 8GB RAM (16GB+ recommended)
- Sufficient disk space (~6GB for model files)

## Quick Setup

### Option 1: Automated Setup
```bash
# Clone or navigate to the legalparam_experimental directory
cd NLPv2/legalparam_experimental

# Run the setup script
./setup.sh

# Activate the environment
source venv/bin/activate

# Test the model
python manual_tests/smoke_test_simple.py
```

### Option 2: Manual Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test the model
python manual_tests/smoke_test_simple.py
```

## Files Created

### `requirements.txt`
Contains all necessary dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- accelerate>=0.20.0
- safetensors>=0.3.0
- tokenizers>=0.13.0

### `manual_tests/smoke_test_simple.py`
A comprehensive test script that:
- Loads the LegalParam model with proper configuration
- Fixes common configuration issues (rope_scaling)
- Tests multiple prompt formats
- Includes an optional interactive mode

### `legalparam_demo.py`
Full-featured demo with:
- Device detection (CUDA/MPS/CPU)
- Multiple example queries
- Interactive chat mode
- Error handling

## Common Issues and Solutions

### 1. Memory Issues
If you run out of memory:
- Use CPU instead of GPU: Set `device="cpu"` in the script
- Reduce `max_new_tokens` parameter
- Close other memory-intensive applications

### 2. Model Loading Errors
The model requires specific configuration fixes:
- rope_scaling configuration is automatically patched
- Trust remote code is enabled for custom model architecture

### 3. Generation Issues
If you get garbled output:
- Try different prompt formats (the test script does this automatically)
- Use greedy decoding (do_sample=False) for more predictable results
- Adjust temperature and top_p parameters

## Usage Examples

### Basic Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bharatgenai/LegalParam"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

query = "What are the key provisions of the RTI Act?"
prompt = f"Question: {query}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### Interactive Mode
Run `python legalparam_demo.py` and it will:
1. Load the model
2. Show example responses
3. Enter interactive mode where you can ask legal questions

## Model Capabilities

LegalParam is trained on Indian legal data and can help with:
- Legal Q&A (acts, laws, policies)
- Document summarization
- Legal concept explanations
- Multi-turn conversations

### Example Queries
- "What steps should a farmer take to legally transfer agricultural land ownership?"
- "Explain the concept of judicial review in Indian constitutional law"
- "What are the key provisions of the Right to Information Act?"

## Performance Tips

1. **Device Selection**: The script automatically detects and uses the best available device
2. **Memory Optimization**: Uses mixed precision on compatible hardware
3. **Generation Parameters**: Start with conservative settings and adjust based on results

## Troubleshooting

### Model Downloads Slowly
- The model is ~3GB and may take time to download
- Consider using `HF_TOKEN` environment variable for faster downloads

### Generated Text is Repetitive
- Adjust temperature (0.7-0.9) and top_p (0.8-0.95)
- Try different prompt formats
- Use sampling instead of greedy decoding

### Runtime Errors
- Check Python version compatibility
- Ensure all dependencies are installed
- Verify sufficient disk space and memory

## Next Steps

Once the setup is working:
1. Experiment with different legal queries
2. Try the interactive mode for conversations
3. Adjust generation parameters for your use case
4. Consider integrating into your own applications

## Support

For issues specific to:
- **LegalParam model**: Check the [HuggingFace model page](https://huggingface.co/bharatgenai/LegalParam)
- **Transformers library**: Refer to [HuggingFace documentation](https://huggingface.co/docs/transformers/)
- **Setup issues**: Review this guide and test the basic setup script first