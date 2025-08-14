# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# register_external_data.py
# Quantizing Stable Diffusion 3.5 Medium using ONNX Live (Olive)
#
# Created by Waheed Brown on August 13, 2025

import onnx
from onnx import helper

MODEL_DIR = "./stable-diffusion-3.5-medium-onnx/text_encoder_2"
MODEL_IN  = f"{MODEL_DIR}/model.onnx"   # expects model.onnx_data next to it

# Load with external data
m = onnx.load(MODEL_IN, load_external_data=True)

# Ensure a sane default opset (use ai.onnx opset 17)
# Remove any weird/empty imports and replace with a clean one.
del m.opset_import[:]
m.opset_import.extend([helper.make_operatorsetid("", 17)])

# Ensure a valid IR version (9 works well with ORT quantization)
m.ir_version = 9

# (Optional) tag metadata
m.producer_name = (m.producer_name or "sd35") + "-irfix"

# Save BACK to the same path, preserving external data format
onnx.save(m, MODEL_IN, save_as_external_data=True)
print("Patched header and saved (external data preserved):", MODEL_IN)