# ONNX Model Quantization
Python ONNX code for quantizing an ONNX machine learning model down to INT8

## Quick Start
1. Set environment variables:
   ```
   export REPO_ID=[Hugging Face repo path to your model]
   ```
2. Run script:
   ```
   python3 quantize_onnx.py
   ```

### EXAMPLE:
```
export REPO_ID=armwaheed/stable-diffusion-3.5-medium-onnx
python3 quantize_onnx.py
```
