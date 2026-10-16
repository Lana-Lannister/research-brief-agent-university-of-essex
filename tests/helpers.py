"""Shared test doubles.

Tests replay recorded/synthetic responses instead of calling live APIs, so
they are deterministic, fast and runnable without keys (Unit 6, Section 6).
"""

from __future__ import annotations

import json
from pathlib import Path

from research_agent.models import Paper, SubQuestion

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code, self._payload = status_code, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Returns queued responses for GET/POST and records the requests made."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def _next(self, method, url, **kw):
        self.requests.append({"method": method, "url": url, **kw})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def get(self, url, **kw):
        return self._next("GET", url, **kw)

    def post(self, url, **kw):
        return self._next("POST", url, **kw)


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text("utf-8"))


def paper(i: int, abstract_words: int = 30, doi: bool = True, **kw) -> Paper:
    return Paper(paper_id=f"P{i}", title=f"Paper number {i} on agents",
                 abstract=" ".join(["word"] * abstract_words) if abstract_words else None,
                 doi=f"10.5555/t.{i}" if doi else None, year=2024, **kw)


def subs(n: int = 3) -> list[SubQuestion]:
    return [SubQuestion(id=i, text=f"Sub-question number {i}", search_query=f"query {i}")
            for i in range(1, n + 1)]
