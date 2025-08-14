# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# quanitize_a8.py
# Quantizing Stable Diffusion 3.5 Medium using ONNX Live (Olive)
#
# Created by Waheed Brown on August 13, 2025

import os, numpy as np, onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

MODEL_IN  = "./stable-diffusion-3.5-medium-onnx/text_encoder_2/model.onnx"
MODEL_A8  = "./build/.text_encoder_2_a8/model.onnx"
os.makedirs(os.path.dirname(MODEL_A8), exist_ok=True)

NUM_SAMPLES = 64
DT = np.int64

def detect_vocab_and_seqlen(model_path, default_vocab=49408, default_seq=77):
    m = onnx.load(model_path, load_external_data=True)
    vocab, seq = None, None
    # Find token embedding (vocab size = first dim)
    for init in m.graph.initializer:
        if len(init.dims) == 2 and "embedding" in init.name.lower() and "token" in init.name.lower():
            vocab = int(init.dims[0])
            break
    # Find positional embedding (seq len = first dim, usually small like 77/128/256)
    for init in m.graph.initializer:
        if len(init.dims) == 2 and "position" in init.name.lower():
            first = int(init.dims[0])
            # sanity: typical CLIP/OpenCLIP ~77; T5 often larger but still <= 512
            if 1 <= first <= 512:
                seq = first
                break
    if vocab is None:
        vocab = default_vocab
    if seq is None:
        seq = default_seq
    print(f"[info] detected vocab_size={vocab}, seq_len={seq}")
    return vocab, seq

VOCAB_SIZE, SEQ_LEN = detect_vocab_and_seqlen(MODEL_IN)

class TE2Reader(CalibrationDataReader):
    def __init__(self, num_samples=NUM_SAMPLES, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, dtype=DT):
        self._it = iter([
            {"input_ids": np.random.randint(0, vocab_size, size=(1, seq_len), dtype=dtype)}
            for _ in range(num_samples)
        ])
    def get_next(self):
        return next(self._it, None)

quantize_static(
    model_input=MODEL_IN,
    model_output=MODEL_A8,
    calibration_data_reader=TE2Reader(),
    activation_type=QuantType.QUInt8,   # A8 activations
    weight_type=QuantType.QInt8,        # keep weights 8-bit; do W4 next in Olive
    use_external_data_format=True       # keep split .onnx/.onnx_data
)

print("Wrote A8 model to:", MODEL_A8)