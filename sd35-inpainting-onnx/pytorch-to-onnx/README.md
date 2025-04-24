# Converting Stable Diffusion 3.5 Medium From PyTorch to ONNX
The scripts in this folder are tools for converting Stable Diffusion 3.5 Medium from a PyTorch model to an ONNX Runtime model.

# Usage
1. Open a web browser, log in to Hugging Face and register your name and email,
   to use [stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
2. Create a new Hugging Face [user access token](https://huggingface.co/docs/hub/en/security-tokens),
   which will capture that you completed the registration form
3. Clone this repo to your machine and change into the directory with the ONNX conversion scripts:
   ```
   cd ./stability-ai-toolkit/sd35-inpainting-onnx/pytorch-to-onnx/
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

7. Convert Stable Diffusion 3.5 Medium from PyTorch to ONNX, using the instructions in [pytorch_to_onnx.sh](./pytorch_to_onnx.sh) (reproduced below, for convenience):
   ```
   optimum-cli export onnx \
    --model stabilityai/stable-diffusion-3.5-medium \
    stable-diffusion-3.5-medium-onnx
   ```
8. Upload the exported model to your Huggingface repo, using [huggingface_upload.py](./huggingface_upload.py) (you will need a Hugging Face Pro account as the exported folder / files will be very large):
   ```
   python huggingface_upload.py [path-to/your-hugging-face-repo] [onnx-model-name]
   
   # Example:
   #
   # python huggingface_upload.py armwaheed/stable-diffusion-3.5-medium-onnx stable-diffusion-3.5-medium-onnx
   ```
