# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# quantize_onnx.py
# Quantize ONNX models in specified Hugging Face repo directories to INT8
#
# Created by Waheed Brown on May 5, 2025

"""
This app quantizes ONNX models down to INT8 for all subdirectories in a HF repo.

USAGE:

Set environment variables:

export REPO_ID=[Hugging Face repo path to your model]
export OUTPUT_DIR=[Optional local output directory for quantized models]

Run script:

python3 quantize_onnx.py

EXAMPLE:

export REPO_ID=armwaheed/stable-diffusion-3.5-medium-onnx
python3 quantize_onnx.py
"""

import os
from huggingface_hub import snapshot_download, login
from onnxruntime.quantization import quantize_dynamic, QuantType

def login_to_hugging_face():
    # Ensure that your Hugging Face token is set as an environment variable
    if os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN'):
        print("Hugging Face access token set")
    else:
        login()
        print("\nWARNING: To avoid the Hugging Face login prompt in the future, please set the HF_TOKEN environment variable:\n\n    export HF_TOKEN=<YOUR HUGGING FACE USER ACCESS TOKEN>\n")

def main():
    login_to_hugging_face()

    repo_id = os.getenv('REPO_ID')
    if not repo_id:
        raise ValueError("Please set the REPO_ID environment variable to your HF repo path.")

    # Download ONNX and external data files into HF cache
    download_dir = snapshot_download(
        repo_id=repo_id,
        revision="main",
        allow_patterns=["*.onnx", "*.onnx_data"]
    )

    # Where to write your quantized models (outside HF cache)
    output_root = os.getenv('OUTPUT_DIR', './.quantized_models')
    os.makedirs(output_root, exist_ok=True)

    # Subdirectories to process
    subdirs = [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "transformer",
        "vae_decoder",
        "vae_encoder",
    ]

    for subdir in subdirs:
        src_dir = os.path.join(download_dir, subdir)
        if not os.path.isdir(src_dir):
            print(f"Skipping missing directory: {subdir}")
            continue

        for root, _, files in os.walk(src_dir):
            for fname in files:
                if fname.endswith(".onnx"):
                    onnx_path = os.path.join(root, fname)
                    # Compute relative path for output
                    rel_path = os.path.relpath(onnx_path, download_dir)
                    quantized_path = os.path.join(
                        output_root,
                        rel_path.replace(".onnx", ".int8.onnx")
                    )
                    os.makedirs(os.path.dirname(quantized_path), exist_ok=True)

                    print(f"Quantizing: {onnx_path} -> {quantized_path}")
                    try:
                        quantize_dynamic(
                            onnx_path,
                            quantized_path,
                            weight_type=QuantType.QInt8
                        )
                        print(f"Successfully wrote: {quantized_path}\n")
                    except Exception as e:
                        print(f"Failed to quantize {onnx_path}: {e}\n")

if __name__ == "__main__":
    main()