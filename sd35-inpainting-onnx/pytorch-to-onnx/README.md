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
   git lfs install

   python huggingface_upload.py [path-to/hugging-face-repo] [path/to/model/folder] [--commit_message "Your commit message"] [--token YOUR_HF_TOKEN]
   ```
   **Example:**
   ```
   git lfs install

   python huggingface_upload.py armwaheed/stable-diffusion-3.5-medium-onnx ~/workspaces/git/stable-diffusion-3.5-medium-onnx --commit_message "Initial commit" --token hf_XXXXXXXX 
   ```
   **WARNING:**
   This script fails on images, with the below error message, so be sure to upload images manually in the Hugging Face web UI for your repo:
   ```
   An error occurred during upload:
   ...
   Your push was rejected because an LFS pointer pointed to a file that does not exist. ... Offending file: - sd3.5_medium_demo.jpg - mmdit-x.png
   ```