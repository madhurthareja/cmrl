"""MedGemma VQA client for vLLM OpenAI-compatible servers."""

from __future__ import annotations

import asyncio
import base64
import logging
from functools import partial
from dataclasses import dataclass
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class MedGemmaConfig:
    """Configuration for MedGemma vLLM endpoint."""

    base_url: str = "http://localhost:8000"
    model_name: str = "medgemma-4b-it_Q4_K_M"
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    timeout: int = 60


class MedGemmaVQAClient:
    """Client that sends multimodal prompts to a MedGemma model served by vLLM."""

    def __init__(self, config: Optional[MedGemmaConfig] = None) -> None:
        self.config = config or MedGemmaConfig()
        self._session = requests.Session()

    @staticmethod
    def _encode_image(image_bytes: bytes, mime_type: str) -> str:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_image}"

    def answer_question(
        self,
        image_bytes: bytes,
        question: str,
        *,
        mime_type: str = "image/png",
        context: Optional[str] = None,
    ) -> Dict:
        """Send image-question pair to MedGemma and return the model output."""

        if not image_bytes:
            raise ValueError("Image payload is empty")

        image_payload = self._encode_image(image_bytes, mime_type)

        user_segments = []
        if context:
            user_segments.append({"type": "text", "text": f"Context:\n{context}\n"})

        user_segments.extend(
            [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_payload}},
            ]
        )

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are MedGemma, a medical vision-language specialist. "
                        "Provide clinically grounded answers, cite key findings when possible, "
                        "and acknowledge uncertainty if information is insufficient."
                    ),
                },
                {"role": "user", "content": user_segments},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

        try:
            response = self._session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.error("MedGemma request failed: %s", exc)
            raise RuntimeError(f"MedGemma request failed: {exc}") from exc

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("MedGemma returned no choices")

        choice = choices[0]
        answer_segments = []
        message_content = choice.get("message", {}).get("content", [])

        if isinstance(message_content, list):
            for segment in message_content:
                if isinstance(segment, dict) and segment.get("type") == "text":
                    answer_segments.append(segment.get("text", ""))
                elif isinstance(segment, str):
                    answer_segments.append(segment)
        elif isinstance(message_content, str):
            answer_segments.append(message_content)

        answer_text = "\n".join(segment.strip() for segment in answer_segments if isinstance(segment, str) and segment.strip())
        if not answer_text:
            fallback = choice.get("message", {}).get("content", "")
            answer_text = fallback.strip() if isinstance(fallback, str) else str(fallback)

        return {
            "answer": answer_text,
            "raw_response": data,
            "model": choice.get("model") or data.get("model", self.config.model_name),
            "usage": data.get("usage", {}),
        }

    async def answer_question_async(
        self,
        image_bytes: bytes,
        question: str,
        *,
        mime_type: str = "image/png",
        context: Optional[str] = None,
    ) -> Dict:
        """Async helper for compatibility with async Flask views."""

        loop = asyncio.get_running_loop()
        bound_call = partial(
            self.answer_question,
            image_bytes,
            question,
            mime_type=mime_type,
            context=context,
        )
        # run_in_executor only supports positional forwarding, so wrap keywords via partial
        return await loop.run_in_executor(None, bound_call)