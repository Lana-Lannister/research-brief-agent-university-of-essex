"""Robustness tests added after the first green run, probing failure paths
that the happy-path suite did not exercise (see TESTING.md, remediation log)."""

import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

from research_agent.agents import EvaluationSynthesisAgent, PlanningAgent, RetrievalAgent
from research_agent.cli import main
from research_agent.config import Settings
from research_agent.heuristic import heuristic_llm
from research_agent.llm import CallbackLLM
from research_agent.models import Assessment, Decision
from research_agent.orchestrator import Orchestrator
from research_agent.review import AutoApproveReviewer, ConsoleReviewer
from research_agent.sources import FixtureCrossrefValidator, SourceUnavailable
from tests.helpers import DEMO, paper


class _DownSearch:
    name = "down"

    def search(self, q, c):
        raise SourceUnavailable("both sources unreachable")


class TestSourcesDown(unittest.TestCase):
    def test_all_sources_down_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            s = Settings(output_dir=Path(d))
            llm = CallbackLLM(heuristic_llm)
            orch = Orchestrator(s, PlanningAgent(llm), RetrievalAgent(_DownSearch()),
                                EvaluationSynthesisAgent(llm),
                                FixtureCrossrefValidator(DEMO / "crossref_fixtures.json"),
                                AutoApproveReviewer())
            result = orch.run("How do agents plan literature searches?")
            self.assertEqual(result.status, "failed")
            self.assertIn("unreachable", result.message)


class _ClosedInputReviewer:
    automatic = False

    def review_subquestions(self, subs):
        raise EOFError

    def review_evidence(self, a, p):  # pragma: no cover
        raise EOFError


class TestCancellation(unittest.TestCase):
    def test_closed_input_at_checkpoint_cancels_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            llm = CallbackLLM(heuristic_llm)
            orch = Orchestrator(Settings(output_dir=Path(d)), PlanningAgent(llm),
                                RetrievalAgent(_DownSearch()), EvaluationSynthesisAgent(llm),
                                FixtureCrossrefValidator(DEMO / "crossref_fixtures.json"),
                                _ClosedInputReviewer())
            result = orch.run("How do agents plan literature searches?")
            self.assertEqual(result.status, "cancelled")
            self.assertEqual(list(Path(d).iterdir()), [])


class TestCliConfiguration(unittest.TestCase):
    def test_missing_api_key_gives_message_not_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            old = dict(os.environ)
            os.environ.update(LLM_PROVIDER="gemini", GEMINI_API_KEY="", LOG_DIR=d, OUTPUT_DIR=d)
            err = StringIO()
            try:
                with redirect_stdout(StringIO()), redirect_stderr(err):
                    code = main(["Some question about agents?", "--auto-approve", "--quiet"])
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertEqual(code, 2)
            self.assertIn("GEMINI_API_KEY", err.getvalue())


class TestConsoleReviewer(unittest.TestCase):
    def _papers(self):
        return {"S2:demo001": paper(1).model_copy(update={"paper_id": "S2:demo001"}),
                "S2:demo002": paper(2).model_copy(update={"paper_id": "S2:demo002"})}

    def _assess(self):
        return [Assessment(paper_id=i, subquestion_id=1, relevance_score=0.9, selected=True)
                for i in ("S2:demo001", "S2:demo002")]

    def test_refine_accepts_ids_without_source_prefix(self):
        answers = iter(["f", "demo001"])
        r = ConsoleReviewer(lambda _: next(answers), lambda *_: None)
        out = r.review_evidence(self._assess(), self._papers())
        self.assertEqual(out.decision, Decision.REFINE)
        self.assertEqual(out.deselected, {"S2:demo001"})

    def test_refine_reprompts_on_unknown_id(self):
        answers = iter(["f", "nonsense", "S2:demo002"])
        printed = []
        r = ConsoleReviewer(lambda _: next(answers), lambda *a: printed.append(a))
        out = r.review_evidence(self._assess(), self._papers())
        self.assertEqual(out.deselected, {"S2:demo002"})
        self.assertTrue(any("Unknown" in str(p) for p in printed))


if __name__ == "__main__":
    unittest.main()
