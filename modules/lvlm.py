from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


@dataclass(slots=True)
class LVLMResponse:
    text: str
    model_name: str
    prompt: str


class QwenVLBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        mm_processor_cache_gb: int = 0,
        enforce_eager: bool = False,
    ) -> None:
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        self.llm = LLM(
            model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt={
                "image": 1,
                "video": 0,
            },
            mm_processor_cache_gb=mm_processor_cache_gb,
            enforce_eager=enforce_eager,
        )

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def _build_chat_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> LVLMResponse:
        return self.generate_batch(
            [image],
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )[0]

    def generate_batch(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> list[LVLMResponse]:
        if not images:
            return []

        chat_prompt = self._build_chat_prompt(
            prompt,
            system_prompt=system_prompt,
        )

        vllm_inputs = [
            {
                "prompt": chat_prompt,
                "multi_modal_data": {
                    "image": self._prepare_image(image),
                },
            }
            for image in images
        ]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
        )

        return [
            LVLMResponse(
                text=output.outputs[0].text.strip(),
                model_name=self.model_name,
                prompt=prompt,
            )
            for output in outputs
        ]