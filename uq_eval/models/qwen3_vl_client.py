"""Qwen3-VL model client for uq_eval framework.

Supports Qwen3-VL models including MoE variants for VLM benchmarks.
- Qwen3-VL-235B-A22B-Thinking: MoE model, 235B total / 22B active

Usage:
    client = Qwen3VLClient(model_name="Qwen/Qwen3-VL-235B-A22B-Thinking")
    response = client.generate(request)
"""
from __future__ import annotations

import base64
import io

import torch
from PIL import Image
from transformers import AutoProcessor

from ..types import ModelRequest, ModelResponse
from .base import BaseModelClient


def decode_data_url(data_url: str) -> Image.Image | None:
    """Decode base64 data URL to PIL Image."""
    if not data_url:
        return None
    try:
        if data_url.startswith("data:"):
            # data:image/jpeg;base64,/9j/4AAQ...
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            # Assume raw base64
            img_bytes = base64.b64decode(data_url)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


class Qwen3VLClient(BaseModelClient):
    """Qwen3-VL model client for uq_eval."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-235B-A22B-Thinking",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        use_flash_attn: bool = True,
        **kwargs,  # Ignore api_key, base_url, timeout_s, etc.
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device_map = device_map
        self.use_flash_attn = use_flash_attn

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._dtype = dtype_map.get(self.dtype, torch.bfloat16)

        print(f"Loading Qwen3-VL processor from {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print(f"Loading Qwen3-VL model from {self.model_name} (dtype={self.dtype})...")

        # Try to import the MoE class, fall back to regular VL class
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
            model_class = Qwen3VLMoeForConditionalGeneration
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration
            print("Warning: Using Qwen2VL class, MoE class not available")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self._dtype,
            "device_map": self.device_map,
        }

        if self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = model_class.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.model.eval()
        print("Qwen3-VL client ready.")

    @property
    def device(self) -> torch.device:
        """Get the device of the first model parameter."""
        return next(self.model.parameters()).device

    def _extract_images_from_messages(self, messages: list[dict]) -> list[Image.Image]:
        """Extract PIL Images from message content."""
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # Handle input_image type (OpenAI Responses API format)
                        if item.get("type") == "input_image":
                            img_url = item.get("image_url", "")
                            img = decode_data_url(img_url)
                            if img:
                                images.append(img)
                        # Handle image_url type (Chat Completions API format)
                        elif item.get("type") == "image_url":
                            url_data = item.get("image_url", {})
                            if isinstance(url_data, dict):
                                img_url = url_data.get("url", "")
                            else:
                                img_url = url_data
                            img = decode_data_url(img_url)
                            if img:
                                images.append(img)
        return images

    def _convert_to_qwen_messages(self, messages: list[dict], images: list[Image.Image]) -> list[dict]:
        """Convert OpenAI-format messages to Qwen3-VL format."""
        qwen_messages = []
        img_idx = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                qwen_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                qwen_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") in ("text", "input_text"):
                            qwen_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") in ("input_image", "image_url"):
                            if img_idx < len(images):
                                qwen_content.append({
                                    "type": "image",
                                    "image": images[img_idx]
                                })
                                img_idx += 1
                    elif isinstance(item, str):
                        qwen_content.append({"type": "text", "text": item})

                if qwen_content:
                    qwen_messages.append({
                        "role": role,
                        "content": qwen_content
                    })

        return qwen_messages

    def generate(self, req: ModelRequest) -> ModelResponse:
        """Generate response for a request."""
        # Extract images from messages
        images = self._extract_images_from_messages(req.messages)

        # Convert to Qwen3-VL format
        qwen_messages = self._convert_to_qwen_messages(req.messages, images)

        # Process with the processor
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        gen_config = {
            'max_new_tokens': req.max_output_tokens,
            'do_sample': req.temperature > 0,
        }
        if req.temperature > 0:
            gen_config['temperature'] = req.temperature

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **gen_config,
                )

            # Decode only the generated tokens (not the input)
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )[0]
        except Exception as e:
            response_text = f"Error: {str(e)}"

        # Clear cache to prevent memory buildup
        torch.cuda.empty_cache()

        return ModelResponse(
            text=response_text,
            raw=None,
            usage={"model": self.model_name},
        )
