# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This app makes a simple inpainting UI for Stable Diffusion using ONNX Runtime.
"""

import gradio as gr
import os
import onnxruntime as ort

# Instead of using PyTorch-based pipelines, we import the ONNX Runtime version.
# The `optimum` package from Hugging Face provides pre-converted pipelines.
from optimum.onnxruntime import ORTStableDiffusion3InpaintPipeline
from huggingface_hub import login
from types import SimpleNamespace


class StableUI:
    _pipe = None

    def __init__(self):
        pass

    def login_to_hugging_face(self):
        # Ensure that your Hugging Face token is set as an environment variable
        # HUGGING_FACE_HUB_TOKEN or HF_TOKEN. If not, the login() function will prompt you.
        if os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN'):
            print("Hugging Face access token set")
        else:
            login()
            print("\nWARNING: To avoid the Hugging Face login prompt in the future, please set the HF_TOKEN environment variable:\n\n    export HF_TOKEN=<YOUR HUGGING FACE USER ACCESS TOKEN>\n")

    def _check_provider(self):
        """
        Check available ONNX Runtime execution providers.
        Priority:
         - CUDAExecutionProvider (GPU via CUDA)
         - DmlExecutionProvider (Microsoft DirectML, e.g. on Windows with AMD GPUs)
         - CPUExecutionProvider (fallback)
        """
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            provider = "CUDAExecutionProvider"
        elif "DmlExecutionProvider" in available_providers:
            provider = "DmlExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
        print(f"Using ONNX Runtime provider: {provider}")
        return provider

    def _predict(self, mask, strength, guidance_scale, prompt, negative_prompt, progress=gr.Progress(track_tqdm=True)):
        # Extract the background image and mask image from the gradio ImageMask input.
        image = mask['background'].convert("RGB")
        mask_image = mask['layers'][0].convert("L")
        mask_image = mask_image.resize(image.size)

        width, height = image.size

        # Calculate new dimensions preserving aspect ratio:
        min_dim = min(width, height)

        # First scale up so that the smallest side is at least 512px.
        if min_dim < 512:
            scale_factor = 512 / min_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = width
            new_height = height

        # Ensure dimensions are multiples of 64
        new_width = (new_width // 64) * 64
        new_height = (new_height // 64) * 64

        # Apply final resizing
        image = image.resize((new_width, new_height))
        mask_image = mask_image.resize((new_width, new_height))

        # Run the inpainting inference using the ONNX Runtime pipeline.
        # The ONNX pipeline API is similar to the diffusers pipeline API.
        output = self._pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            width=new_width,
            height=new_height,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt
        )
        return output.images[0]

    def _start_gradio(self):
        white_brush = gr.Brush(default_color='#FFFFFF', colors=['#FFFFFF'], color_mode='fixed')
        gr.Interface(
            self._predict,
            title='Stable Diffusion 3.5 Medium In-Painting (ONNX Runtime)',
            inputs=[
                gr.ImageMask(type='pil', label='Inpaint', brush=white_brush),
                gr.Slider(minimum=0, maximum=1, value=1.0, label="strength (increase inpainting strength)"),
                gr.Slider(minimum=1, maximum=10, value=7.5, label="guidance scale (increase to apply text prompt)"),
                gr.Textbox(label='prompt'),
                gr.Textbox(label='negative prompt')
            ],
            outputs=gr.Image(type="pil")
        ).launch(debug=True, share=True)

    def start_inpaint(self):
        provider = self._check_provider()
        # Load the ONNX Runtime pipeline. This downloads and loads the pre-converted models.
        # Note: You must have the corresponding optimized model on the Hugging Face hub.
        self._pipe = ORTStableDiffusion3InpaintPipeline.from_pretrained(
            "armwaheed/stable-diffusion-3.5-medium-onnx",
            provider=provider
        )
        self._start_gradio()
        return 0


def main():
    ui = StableUI()
    ui.login_to_hugging_face()
    ui.start_inpaint()


if __name__ == "__main__":
    main()

