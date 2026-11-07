"""Language model clients. The agent only needs one method: complete()."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def complete(self, system: str, user: str, *, schema: dict[str, Any] | None = None) -> str:
        """Return the model's reply. If `schema` is given the reply should be JSON for it."""
        ...


class LLMError(RuntimeError):
    """The model could not be reached or gave no usable answer."""


class OllamaClient:
    """Talks to a local Ollama server (https://ollama.com).

    Models of 8 billion parameters or fewer fit in 8 GB of GPU memory. Temperature 0 keeps
    answers repeatable. When a schema is given Ollama constrains the output to valid JSON
    for it.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        host: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        try:
            import ollama
        except ImportError as error:  # pragma: no cover - depends on the install
            raise LLMError('Install the agent extras: pip install "autoeda[agent]"') from error
        self.model = model
        self.temperature = temperature
        self._client = ollama.Client(host=host, timeout=timeout)

    def complete(self, system: str, user: str, *, schema: dict[str, Any] | None = None) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema,
                options={"temperature": self.temperature, "seed": 0},
            )
        except Exception as error:  # connection refused, model missing, timeout
            raise LLMError(f"Ollama call failed: {error}") from error
        return str(response["message"]["content"])


class FakeLLM:
    """A scripted model for tests and for measuring the pipeline without a real model.

    `responses` is either a list (replies are returned in order, the last one repeats) or a
    function taking (system, user) and returning the reply.
    """

    def __init__(self, responses: list[str] | Callable[[str, str], str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, schema: dict[str, Any] | None = None) -> str:
        self.calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def extract_json(text: str) -> Any:
    """Parse the first JSON object in a reply, tolerating code fences and chatter."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object in the reply")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(cleaned[start:])
    return value


def ask_for_model(
    llm: LLM, system: str, user: str, model: type[T], *, retries: int = 1
) -> tuple[T | None, str | None]:
    """Ask for JSON matching `model`. Returns (parsed, None) or (None, last error message).

    On a bad reply the error is shown to the model once so it can correct itself.
    """
    schema = model.model_json_schema()
    prompt = user
    error = "no reply"
    for _ in range(retries + 1):
        try:
            reply = llm.complete(system, prompt, schema=schema)
        except LLMError as exc:
            return None, str(exc)
        try:
            return model.model_validate(extract_json(reply)), None
        except (ValueError, ValidationError) as exc:
            error = str(exc).splitlines()[0][:300]
            prompt = (
                f"{user}\n\nYour previous reply was not valid: {error}\n"
                "Reply again with only the JSON object."
            )
    return None, error
