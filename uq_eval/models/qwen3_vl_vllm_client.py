"""Qwen3-VL model client using vLLM for high-throughput inference.

vLLM provides 2-4x speedup over standard HuggingFace transformers via:
- PagedAttention for efficient KV cache management
- Continuous batching for higher throughput
- Optimized CUDA kernels

Supports Qwen3-VL models including MoE variants for VLM benchmarks.
- Qwen3-VL-30B-A3B-Instruct: MoE model, 30B total / 3B active
- Qwen3-VL-235B-A22B-Thinking: MoE model, 235B total / 22B active

Usage:
    client = Qwen3VLvLLMClient(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct")
    response = client.generate(request)

Requirements:
    - vLLM >= 0.13.0 (install via: pip install vllm)
    - finegrain_vlm conda environment recommended
"""
from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

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


class Qwen3VLvLLMClient(BaseModelClient):
    """Qwen3-VL model client using vLLM for high-throughput inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768,
        trust_remote_code: bool = True,
        **kwargs,  # Ignore api_key, base_url, timeout_s, etc.
    ):
        """Initialize vLLM-based Qwen3-VL client.

        Args:
            model_name: HuggingFace model ID or local path
            dtype: Data type for inference (bfloat16, float16, auto)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length
            trust_remote_code: Whether to trust remote code in model
        """
        self.model_name = model_name
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        # Import vLLM here to avoid import errors if not installed
        try:
            from vllm import LLM, SamplingParams
            self._SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for Qwen3VLvLLMClient. "
                "Install via: pip install vllm>=0.13.0"
            )

        print(f"Loading Qwen3-VL with vLLM from {self.model_name}...")
        print(f"  dtype={dtype}, tensor_parallel_size={tensor_parallel_size}")
        print(f"  gpu_memory_utilization={gpu_memory_utilization}")

        self.llm = LLM(
            model=self.model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            # Limit number of sequences to avoid OOM on large batches
            max_num_seqs=16,
        )
        print("vLLM Qwen3-VL client ready.")

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

    def _build_prompt_with_images(
        self, messages: list[dict], images: list[Image.Image]
    ) -> dict[str, Any]:
        """Build vLLM-compatible prompt with images.

        vLLM expects a dict with 'prompt' and optional 'multi_modal_data'.
        """
        # Build text prompt from messages
        text_parts = []
        img_idx = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                text_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                msg_text = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") in ("text", "input_text"):
                            msg_text.append(item.get("text", ""))
                        elif item.get("type") in ("input_image", "image_url"):
                            # Add image placeholder
                            msg_text.append("<image>")
                            img_idx += 1
                    elif isinstance(item, str):
                        msg_text.append(item)
                text_parts.append(f"{role}: {''.join(msg_text)}")

        # Join all parts
        full_prompt = "\n".join(text_parts)
        if not full_prompt.endswith("assistant:"):
            full_prompt += "\nassistant:"

        # Build the input dict
        result = {"prompt": full_prompt}
        if images:
            result["multi_modal_data"] = {"image": images}

        return result

    def generate(self, req: ModelRequest) -> ModelResponse:
        """Generate response for a request using vLLM."""
        # Extract images from messages
        images = self._extract_images_from_messages(req.messages)

        # Build vLLM-compatible prompt
        vllm_input = self._build_prompt_with_images(req.messages, images)

        # Set up sampling parameters
        sampling_params = self._SamplingParams(
            max_tokens=req.max_output_tokens,
            temperature=req.temperature if req.temperature > 0 else 0.0,
        )

        try:
            # Generate with vLLM
            outputs = self.llm.generate([vllm_input], sampling_params)
            response_text = outputs[0].outputs[0].text if outputs else ""
        except Exception as e:
            response_text = f"Error: {str(e)}"

        return ModelResponse(
            text=response_text,
            raw=None,
            usage={"model": self.model_name, "backend": "vllm"},
        )

    def generate_batch(self, requests: list[ModelRequest]) -> list[ModelResponse]:
        """Generate responses for multiple requests in a batch.

        This is the main advantage of vLLM - efficient batch processing
        with continuous batching.
        """
        # Prepare all inputs
        vllm_inputs = []
        for req in requests:
            images = self._extract_images_from_messages(req.messages)
            vllm_input = self._build_prompt_with_images(req.messages, images)
            vllm_inputs.append(vllm_input)

        # Use the same sampling params for all (could be made per-request)
        # Use the first request's params as template
        first_req = requests[0] if requests else None
        sampling_params = self._SamplingParams(
            max_tokens=first_req.max_output_tokens if first_req else 1024,
            temperature=first_req.temperature if first_req and first_req.temperature > 0 else 0.0,
        )

        try:
            # Generate all at once - vLLM handles batching efficiently
            outputs = self.llm.generate(vllm_inputs, sampling_params)

            responses = []
            for output in outputs:
                response_text = output.outputs[0].text if output.outputs else ""
                responses.append(ModelResponse(
                    text=response_text,
                    raw=None,
                    usage={"model": self.model_name, "backend": "vllm"},
                ))
            return responses
        except Exception as e:
            # On error, return error responses for all
            error_response = ModelResponse(
                text=f"Error: {str(e)}",
                raw=None,
                usage={"model": self.model_name, "backend": "vllm"},
            )
            return [error_response] * len(requests)
