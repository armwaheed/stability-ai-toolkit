# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# pytorch_to_onnx.sh
# Pytorch to ONNX Model Conversion
#
# Created by Waheed Brown on April 24, 2025
#
# Example usage:
#
# optimum-cli export onnx \
#  --model stabilityai/stable-diffusion-3.5-medium \
#  stable-diffusion-3.5-medium-onnx

optimum-cli export onnx \
  --model ${HUGGINGFACE_SOURCE_MODEL} \
  ${EXPORTED_MODEL_NAME}
