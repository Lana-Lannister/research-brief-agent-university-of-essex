"""Unit tests: JSON extraction, validate-and-re-prompt loop, Gemini client transport."""

import json
import unittest
from unittest import mock

from pydantic import BaseModel

from research_agent.llm import GeminiClient, LLMOutputError, ScriptedLLM, extract_json
from tests.helpers import FakeResponse, FakeSession, load_fixture


class _Out(BaseModel):
    search_query: str


class TestExtractJson(unittest.TestCase):
    def test_plain_fenced_and_preamble(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here it is:\n{"a": [1, 2]}\nThanks'), {"a": [1, 2]})

    def test_garbage_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            extract_json("no json here")


class TestCompleteJson(unittest.TestCase):
    def test_reprompts_with_error_then_succeeds(self):
        llm = ScriptedLLM(["not json", {"wrong": 1}, {"search_query": "ok query"}], max_retries=2)
        out = llm.complete_json("t", "sys", "user", _Out)
        self.assertEqual(out.search_query, "ok query")
        self.assertEqual(llm.calls, 3)
        # The validation error is fed back on the retry prompt.
        self.assertIn("previous reply was invalid", llm.prompts[1][1])

    def test_gives_up_after_budget(self):
        llm = ScriptedLLM(["x", "y"], max_retries=1)
        with self.assertRaises(LLMOutputError):
            llm.complete_json("t", "sys", "user", _Out)


class TestGeminiClient(unittest.TestCase):
    def test_requires_key(self):
        with self.assertRaises(ValueError):
            GeminiClient("", "m")

    @mock.patch("research_agent.llm.time.sleep")
    def test_backs_off_on_429_and_parses(self, sleep):
        session = FakeSession([FakeResponse(429), FakeResponse(200, load_fixture("gemini_response.json"))])
        client = GeminiClient("k", "gemini-test", session=session)
        out = client.complete_json("reformulate", "sys", "user", _Out)
        self.assertEqual(out.search_query, "agents review automation")
        self.assertEqual(len(session.requests), 2)
        sleep.assert_called_once_with(5)
        req = session.requests[-1]
        self.assertEqual(req["headers"]["x-goog-api-key"], "k")
        self.assertNotIn("k", req["url"])  # key never in URL (would leak to logs)
        self.assertEqual(req["json"]["generationConfig"]["responseMimeType"], "application/json")

    def test_empty_candidate_is_output_error(self):
        session = FakeSession([FakeResponse(200, {"candidates": []})])
        with self.assertRaises(LLMOutputError):
            GeminiClient("k", "m", session=session).complete_json("t", "s", "u", _Out)


if __name__ == "__main__":
    unittest.main()
