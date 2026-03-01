"""InternVL model client for uq_eval framework.

Supports InternVL3.5 and InternVL3 models for VLM benchmarks.
- InternVL3.5-38B: ~80GB VRAM (2x A100-80GB)
- InternVL3-78B: ~160GB VRAM (4x A100-80GB)
- InternVL3.5-241B-A28B: MoE model, ~80GB VRAM (only 28B active)

Usage:
    client = InternVLClient(model_name="OpenGVLab/InternVL3_5-38B")
    response = client.generate(request)
"""
from __future__ import annotations

import base64
import io
import re

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types import ModelRequest, ModelResponse
from .base import BaseModelClient


# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    """Build image transform for InternVL."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


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


class InternVLClient(BaseModelClient):
    """InternVL3 model client for uq_eval."""

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-38B",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        input_size: int = 448,
        **kwargs,  # Ignore api_key, base_url, timeout_s, etc.
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device_map = device_map
        self.input_size = input_size

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._dtype = dtype_map.get(self.dtype, torch.bfloat16)
        self.transform = build_transform(self.input_size)

        print(f"Loading InternVL tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print(f"Loading InternVL model from {self.model_name} (dtype={self.dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            device_map=self.device_map,
        )
        self.model.eval()
        print("InternVL client ready.")

    @property
    def device(self) -> torch.device:
        """Get the device of the first model parameter."""
        return next(self.model.parameters()).device

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for InternVL."""
        pixel_values = self.transform(image).unsqueeze(0)
        return pixel_values.to(self._dtype).to(self.device)

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

    def _extract_text_from_messages(self, messages: list[dict]) -> str:
        """Extract text content from messages."""
        texts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                if role == "system":
                    texts.append(f"System: {content}")
                elif role == "user":
                    texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") in ("text", "input_text"):
                            text = item.get("text", "")
                            if text:
                                if role == "system":
                                    texts.append(f"System: {text}")
                                else:
                                    texts.append(text)
                    elif isinstance(item, str):
                        texts.append(item)

        return "\n\n".join(texts)

    def generate(self, req: ModelRequest) -> ModelResponse:
        """Generate response for a request."""
        # Extract images and text from messages
        images = self._extract_images_from_messages(req.messages)
        prompt = self._extract_text_from_messages(req.messages)

        # Process first image if present (InternVL handles one image at a time)
        if images:
            pixel_values = self.preprocess_image(images[0])
        else:
            pixel_values = None

        # Generate
        gen_config = {
            'max_new_tokens': req.max_output_tokens,
            'do_sample': req.temperature > 0,
        }
        if req.temperature > 0:
            gen_config['temperature'] = req.temperature

        try:
            if pixel_values is not None:
                response_text = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    gen_config,
                )
            else:
                # Text-only - use standard generation
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=req.max_output_tokens,
                        do_sample=req.temperature > 0,
                        temperature=req.temperature if req.temperature > 0 else None,
                    )
                response_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
        except Exception as e:
            response_text = f"Error: {str(e)}"

        # Clear cache to prevent memory buildup
        torch.cuda.empty_cache()

        return ModelResponse(
            text=response_text,
            raw=None,
            usage={"model": self.model_name},
        )
