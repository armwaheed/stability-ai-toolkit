# ONNX Model Quantization
Python ONNX code for quantizing an ONNX machine learning model down to INT8
* This script uses `onnxruntime.quantization.quantize_dynamic()` to quantize `*.onnx` files to `onnxruntime.quantization.QuantType.QInt8` (8-bit integer)
* Addtionally, if a `*.onnx` file has an associated `*.onnx_data` file, then these two are combined into a single `QInt8` output file
* The output file naming convention is `*.int8.onnx`
* Output files will be generated in location `[REPO ROOT]/quantized_models`

## Quick Start
1. Set environment variables:
   ```
   export REPO_ID=[Hugging Face repo path to your model]
   ```
2. Run script:
   ```
   python3 quantize_onnx.py
   ```
   **NOTE:** [quantize_onnx.py](./quantize_onnx.py) is hard coded to iterate through the follwing [stable-diffusion-3.5-medium-onnx](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main) folders, so please update this list if you're trying to quantize your own model with different folders:
   ```
   # Subdirectories to process
   subdirs = [
      "text_encoder",
      "text_encoder_2",
      "text_encoder_3",
      "transformer",
      "vae_decoder",
      "vae_encoder",
   ]
   ```
   **NOTE:** Only folders with `*.onnx` and `*.onnx_data` files are iterated through, for quantization. Other files and folders can be preserved as is, and used with the quantized `model.int8.onnx`


### EXAMPLE:
```
export REPO_ID=armwaheed/stable-diffusion-3.5-medium-onnx
python3 quantize_onnx.py
```

## Quantization Model Size Reduction
Example quantization results for [Stable Diffusion 3.5 Medium ONNX](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main) (SD3.5 M ONNX)
|[SD3.5 M ONNX](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main)|Original ONNX Size (GB)|[SD3.5 M ONNX INT8](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx-int8/tree/main)|Quantized ONNX Size (GB)|
|-----------------------------------------|-----------------------|-------------------|------------------------|
|Total Original ONNX Size|32.494 GB|Total Quantized ONNX Size|8.827 GB|
|[text_encoder](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/text_encoder)/model.onnx|0.495 GB|model.int8.onnx|0.124 GB|
|[text_encoder_2](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/text_encoder_2)/model.onnx<br/>[text_encoder_2](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/text_encoder_2)/model.onnx_data|0.001 GB<br/>2.78 GB|model.int8.onnx|0.698 GB|
|[text_encoder_3](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/text_encoder_3)/model.onnx<br/>[text_encoder_3](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/text_encoder_3)/model.onnx_data|0.0005 GB<br/>19 GB|model.int8.onnx|4.764 GB|
|[transformer](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/transformer)/model.onnx<br/>[transformer](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/transformer)/model.onnx_data|0.002 GB<br/>9.88 GB|model.int8.onnx|3.156 GB|
|[vae_decoder](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/vae_decoder)/model.onnx|0.198 GB|model.int8.onnx|0.05 GB|
|[vae_encoder](https://huggingface.co/armwaheed/stable-diffusion-3.5-medium-onnx/tree/main/vae_encoder)/model.onnx|0.137 GB|model.int8.onnx|0.035 GB|