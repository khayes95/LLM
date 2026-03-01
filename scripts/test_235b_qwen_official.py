#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Qwen3-VL-235B with official Qwen example code.
Uses FP8 model and expert parallel as recommended.
"""
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    messages = [
        {
            "role": "user",
            "content": [
              {
                  "type": "image",
                  "image": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
              },
              {"type": "text", "text": "Read all the text in the image."},
            ],
        }
    ]

    # Use FP8 model as recommended for A100
    checkpoint_path = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
    print(f"Loading processor from {checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]

    print(f"\nLoading LLM with {torch.cuda.device_count()} GPUs...")
    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt'][:200]=!r}")
    print('\n' + '>' * 40)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 40)
        print(f"Generated text: {generated_text!r}")
