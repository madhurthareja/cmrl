from __future__ import annotations

import base64
import json

import pytest

from backend.models.medgemma_vqa import MedGemmaVQAClient, MedGemmaConfig


class _StubResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ValueError(f"HTTP {self.status_code}")


def test_answer_question_builds_payload_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MedGemmaConfig(
        base_url="http://localhost:1234",
        model_name="test-medgemma",
        max_tokens=64,
        temperature=0.1,
        top_p=0.9,
        timeout=15,
    )
    client = MedGemmaVQAClient(config)

    captured: dict = {}

    def fake_post(url: str, json: dict | None = None, timeout: int | None = None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "finding A"},
                            {"type": "text", "text": "finding B"},
                        ]
                    },
                    "model": "returned-model",
                }
            ],
            "usage": {"total_tokens": 42},
        }
        return _StubResponse(payload)

    monkeypatch.setattr(client._session, "post", fake_post)

    image_bytes = b"\x89PNG\r\n\x1a\n"
    result = client.answer_question(
        image_bytes=image_bytes,
        question="What is shown?",
        mime_type="image/png",
        context="History: prior stroke",
    )

    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["timeout"] == 15
    assert captured["json"]["model"] == "test-medgemma"

    user_content = captured["json"]["messages"][1]["content"]
    assert user_content[0]["text"].startswith("Context:\nHistory: prior stroke")
    assert user_content[1]["text"] == "What is shown?"

    image_block = user_content[2]["image_url"]["url"]
    assert image_block.startswith("data:image/png;base64,")
    # Ensure the encoded payload matches the input bytes
    encoded_payload = image_block.split(",", 1)[1]
    assert base64.b64decode(encoded_payload) == image_bytes

    assert result["answer"] == "finding A\nfinding B"
    assert result["usage"] == {"total_tokens": 42}
    assert result["model"] == "returned-model"


def test_answer_question_empty_image_raises() -> None:
    client = MedGemmaVQAClient()
    with pytest.raises(ValueError):
        client.answer_question(b"", "Question?")


def test_answer_question_no_choices_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MedGemmaVQAClient()

    def fake_post(*_, **__):
        return _StubResponse({"choices": []})

    monkeypatch.setattr(client._session, "post", fake_post)

    with pytest.raises(RuntimeError):
        client.answer_question(b"image", "Question?")


@pytest.mark.asyncio
async def test_answer_question_async_wraps_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MedGemmaVQAClient()

    called = {}

    def fake_answer(image_bytes: bytes, question: str, *, mime_type: str, context: str | None):
        called["args"] = (image_bytes, question, mime_type, context)
        return {"answer": "ok"}

    monkeypatch.setattr(client, "answer_question", fake_answer)

    result = await client.answer_question_async(
        b"\xff\xd8\xff",
        "Is there pneumonia?",
        mime_type="image/jpeg",
        context="cx",
    )

    assert called["args"] == (b"\xff\xd8\xff", "Is there pneumonia?", "image/jpeg", "cx")
    assert result == {"answer": "ok"}
