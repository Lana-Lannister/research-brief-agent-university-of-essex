"""LLM access behind one narrow interface.

Design decisions
----------------
* **Only semantic tasks reach the model** (decomposition, reformulation,
  relevance scoring, synthesis). Anything an exact rule can decide stays in
  deterministic code, because LLMs produce fluent but unsupported output
  (Klesel and Wittmann, 2025; Walters and Wilder, 2023).
* **Structured output + validate + re-prompt.** The model is asked for JSON;
  the reply is parsed and validated against a Pydantic schema. On failure the
  validation error is fed back and the model retries, up to a fixed limit.
  This is the "log failures and drive re-prompting" loop promised in Unit 6
  Section 6, and a bounded form of self-correction from feedback (Shinn et al.,
  2023). The bound matters: unbounded self-repair can loop indefinitely.
* **Provider is swappable** (Strategy pattern; Gamma et al., 1994). Agents
  depend on `LLMClient`, never on Gemini, so tests inject a fake and the
  offline mode injects a transparent heuristic baseline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMOutputError(RuntimeError):
    """Raised when the model cannot produce valid output within the retry budget."""


def extract_json(text: str) -> object:
    """Parse JSON from a model reply that may be wrapped in Markdown fences.

    Models frequently add ```json fences or a sentence of preamble even when
    asked not to; recovering the first JSON object/array is cheaper and more
    reliable than another round-trip to the API.
    """
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


class LLMClient(ABC):
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.calls = 0  # exposed for evaluation (cost per run)

    @abstractmethod
    def _generate(self, task: str, system: str, user: str) -> str:
        """Return raw text for one prompt."""

    def complete_json(self, task: str, system: str, user: str, schema: Type[T]) -> T:
        prompt = user
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            raw = self._generate(task, system, prompt)
            try:
                return schema.model_validate(extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log.warning("llm_invalid_output task=%s attempt=%d error=%s",
                            task, attempt + 1, str(exc).splitlines()[0])
                # Feed the concrete error back: models repair specific
                # complaints far more reliably than a generic "try again".
                prompt = (f"{user}\n\nYour previous reply was invalid: {exc}.\n"
                          "Reply again with ONLY valid JSON matching the schema.")
        raise LLMOutputError(f"{task}: no valid output after "
                             f"{self.max_retries + 1} attempts: {last_error}")


class GeminiClient(LLMClient):
    """Google Gemini via the public REST endpoint (free tier, Unit 6 Section 3).

    The REST API is used instead of the SDK to keep dependencies to `requests`
    and to make the exact request auditable in the code.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, temperature: float = 0.2,
                 max_retries: int = 2, timeout: float = 60.0,
                 session: requests.Session | None = None):
        super().__init__(max_retries)
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set (see README, Configuration).")
        self.api_key, self.model, self.temperature = api_key, model, temperature
        self.timeout = timeout
        self.session = session or requests.Session()

    def _generate(self, task: str, system: str, user: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature,
                                 "responseMimeType": "application/json"},
        }
        # Free-tier quotas return HTTP 429; back off rather than fail the run.
        for wait in (0, 5, 15, 30):
            if wait:
                log.info("gemini_rate_limited sleeping=%ss", wait)
                time.sleep(wait)
            resp = self.session.post(self.URL.format(model=self.model), json=body,
                                     headers={"x-goog-api-key": self.api_key},
                                     timeout=self.timeout)
            if resp.status_code != 429 and resp.status_code < 500:
                break
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            # A safety block or empty candidate is treated like invalid output.
            raise LLMOutputError(f"Gemini returned no text: {data}") from exc


class ScriptedLLM(LLMClient):
    """Returns queued replies in order. Used by tests to make agents deterministic."""

    def __init__(self, replies: list[str | dict | list], max_retries: int = 2):
        super().__init__(max_retries)
        self.replies = deque(r if isinstance(r, str) else json.dumps(r) for r in replies)
        self.prompts: list[tuple[str, str]] = []

    def _generate(self, task: str, system: str, user: str) -> str:
        self.prompts.append((task, user))
        if not self.replies:
            raise AssertionError(f"ScriptedLLM ran out of replies at task={task}")
        return self.replies.popleft()


class CallbackLLM(LLMClient):
    """Delegates to a function. Lets the offline heuristic plug in as a provider."""

    def __init__(self, fn: Callable[[str, str, str], str], max_retries: int = 0):
        super().__init__(max_retries)
        self.fn = fn

    def _generate(self, task: str, system: str, user: str) -> str:
        return self.fn(task, system, user)
