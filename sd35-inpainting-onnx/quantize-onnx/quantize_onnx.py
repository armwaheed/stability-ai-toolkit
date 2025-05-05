# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# quantize_onnx.py
# Quantize ONNX model to INT8
#
# Created by Waheed Brown on May 5, 2025

"""
This app quantizes an ONNX model down to INT8

USAGE:

Set environment variables:

export REPO_ID=[Hugging Face repo path to your model]
export FILENAME=[output filenane]

Run script:

python3 quantize_onnx.py

EXAMPLE:

export REPO_ID=armwaheed/stable-diffusion-3.5-medium-onnx
python3 quantize_onnx.py
"""

import os
from huggingface_hub import snapshot_download, HfApi, login
from onnxruntime.quantization import quantize_dynamic, QuantType

def login_to_hugging_face():
    # Ensure that your Hugging Face token is set as an environment variable
    # HUGGING_FACE_HUB_TOKEN or HF_TOKEN. If not, the login() function will prompt you.
    if os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN'):
        print("Hugging Face access token set")
    else:
        login()
        print("\nWARNING: To avoid the Hugging Face login prompt in the future, please set the HF_TOKEN environment variable:\n\n    export HF_TOKEN=<YOUR HUGGING FACE USER ACCESS TOKEN>\n")

def main():
    login_to_hugging_face()

    repo_id = os.getenv('REPO_ID')
    download_dir = snapshot_download(
        repo_id=repo_id,
        revision="main",
        allow_patterns=["*.onnx", "*.onnx_data"]
    )
    onnx_path = os.path.join(download_dir, "transformer", "model.onnx")

    quantized_path = onnx_path.replace(".onnx", ".int8.onnx")
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized model written to: {quantized_path}")

if __name__ == "__main__":
    main()

