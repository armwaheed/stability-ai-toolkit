# Quantizing Stable Diffusion 3.5 Medium using ONNX Live (Olive)

## Quick Start
1. Open a web browser, log in to Hugging Face and register your name and email,
   to use [stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)(SD3.5M)
2. Create a new Hugging Face [user access token](https://huggingface.co/docs/hub/en/security-tokens),
   which will capture that you completed the registration form
3. Clone this repo to your machine and change into the directory for this demo:
   ```
   cd ./stable-diffusion-onnx-olive
   ```
4. Set up the app in a Python virtual environment:

   ```
   python -m venv <your_environment_name>
   source <your_environment_name>/bin/activate
   ```
5. Set your `HF_TOKEN` inside your virtual environment
   ```
   export HF_TOKEN=<Hugging Face user access token>
   ```
6. Install dependencies
   ```
   pip install -r requirements.txt
   ```

7. Export Stable Diffusion 3.5 Medium from PyTorch to ONNX format:
   ```
   optimum-cli export onnx \
     --model stabilityai/stable-diffusion-3.5-medium \
     stable-diffusion-3.5-medium-onnx
   ```

8. Convert each SD3.5M component to asymmetrically quanitized 8-bit integer activations and symmetrically quantized 4-bit integer weights:

   **NOTE:** The output `model.onnx` files will be nested in the `build` directory

   * `text_encoder`:
      ```
      olive run --config olive-config/text_encoder/text_encoder_a8.json
      olive run --config olive-config/text_encoder/text_encoder_w4.json
      ```
   * `text_encoder_2`:
     ```
     python olive-config/text_encoder_2/register_external_data.py
     python olive-config/text_encoder_2/quanitize_a8.py
     olive run --config olive-config/text_encoder_2/text_encoder_2_w4.json
     ```