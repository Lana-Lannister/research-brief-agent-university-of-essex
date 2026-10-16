"""Functional (end-to-end) tests of the orchestrated workflow, offline.

Each test drives the full pipeline through one branch of Unit 6 Diagrams 2
and 3, using the demo fixtures, the lexical baseline LLM and a scripted
researcher. Live-API end-to-end runs are documented separately in TESTING.md.
"""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from research_agent.agents import EvaluationSynthesisAgent, PlanningAgent, RetrievalAgent
from research_agent.cli import main
from research_agent.config import Settings
from research_agent.heuristic import heuristic_llm
from research_agent.llm import CallbackLLM, ScriptedLLM
from research_agent.models import Decision
from research_agent.orchestrator import Orchestrator, Orchestrator as _O
from research_agent.review import AutoApproveReviewer, ReviewResult, ScriptedReviewer
from research_agent.sources import FixtureCrossrefValidator, FixtureSearch
from research_agent.storage import render_markdown
from tests.helpers import DEMO, paper

QUESTION = "How effective are LLM-based agents at automating systematic literature reviews?"
APPROVE = ReviewResult(Decision.APPROVE)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(output_dir=Path(self.tmp.name), relevance_cutoff=0.5)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, reviewer, llm=None, settings=None):
        s = settings or self.settings
        llm = llm or CallbackLLM(heuristic_llm)
        self.search = FixtureSearch(DEMO / "search_fixtures.json")
        return Orchestrator(s, PlanningAgent(llm), RetrievalAgent(self.search, s.record_threshold),
                            EvaluationSynthesisAgent(llm, s.relevance_cutoff),
                            FixtureCrossrefValidator(DEMO / "crossref_fixtures.json"), reviewer)


class TestHappyPath(Base):
    def test_full_run_produces_grounded_brief(self):
        result = self.build(AutoApproveReviewer()).run(QUESTION, run_id="t1")
        self.assertEqual(result.status, "completed")
        brief = result.brief
        md, js = result.paths
        self.assertTrue(md.exists() and js.exists())
        # JSON round-trips to the same brief (storage is lossless).
        self.assertEqual(json.loads(js.read_text())["run_id"], "t1")
        # Every summary belongs to a selected paper (grounding).
        selected = {p.paper_id for p in brief.selected_papers}
        self.assertEqual({s.paper_id for s in brief.summaries}, selected)
        # Duplicate record (demo002 appears under two queries) merged once.
        ids = [p.paper_id for p in brief.selected_papers]
        self.assertEqual(len(ids), len(set(ids)))
        # Auto-approval is disclosed in the brief's limitations.
        self.assertTrue(any("auto-approve" in l for l in brief.limitations))

    def test_reformulation_happens_once_for_thin_subquestion(self):
        orch = self.build(AutoApproveReviewer())
        result = orch.run(QUESTION)
        self.assertEqual(result.stats["reformulations"], 1)
        sq2 = result.brief.sub_questions[1]
        self.assertTrue(sq2.retried_once)
        self.assertFalse(sq2.insufficient_coverage)
        self.assertEqual(len(self.search.queries), 4)  # 3 planned + 1 reformulated

    def test_unverified_dois_are_flagged_not_hidden(self):
        result = self.build(AutoApproveReviewer()).run(QUESTION)
        flags = {p.paper_id: p.metadata_flag for p in result.brief.selected_papers}
        self.assertTrue(flags["S2:demo008"].startswith("title_mismatch"))
        self.assertIn("not verified", render_markdown(result.brief))
        self.assertTrue(any("could not be verified" in l for l in result.brief.limitations))


class TestReviewBranches(Base):
    def test_revision_at_checkpoint1_replans(self):
        reviewer = ScriptedReviewer([ReviewResult(Decision.REVISE, "narrow it"), APPROVE], [APPROVE])
        result = self.build(reviewer).run(QUESTION)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(reviewer.seen_subq), 2)

    def test_refine_removes_deselected_papers(self):
        reviewer = ScriptedReviewer([APPROVE], [ReviewResult(Decision.REFINE,
                                                             deselected={"S2:demo008"})])
        result = self.build(reviewer).run(QUESTION)
        self.assertNotIn("S2:demo008", [p.paper_id for p in result.brief.selected_papers])

    def test_scope_rejection_allowed_once(self):
        reject = ReviewResult(Decision.REJECT_SCOPE, "wrong scope")
        reviewer = ScriptedReviewer([APPROVE, APPROVE], [reject, APPROVE])
        result = self.build(reviewer).run(QUESTION)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stats["scope_revisions"], 1)
        self.assertEqual(len(reviewer.seen_evidence), 2)

    def test_second_scope_rejection_stops_run(self):
        reject = ReviewResult(Decision.REJECT_SCOPE, "still wrong")
        reviewer = ScriptedReviewer([APPROVE, APPROVE], [reject, reject])
        result = self.build(reviewer).run(QUESTION)
        self.assertEqual(result.status, "stopped_scope")
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])  # nothing saved

    def test_deselecting_everything_yields_no_evidence(self):
        reviewer = ScriptedReviewer([APPROVE], [ReviewResult(
            Decision.REFINE, deselected={f"S2:demo{i:03d}" for i in range(1, 11)})])
        self.assertEqual(self.build(reviewer).run(QUESTION).status, "no_evidence")


class TestFailureModes(Base):
    def test_invalid_llm_output_fails_closed(self):
        llm = ScriptedLLM(["nonsense"] * 3, max_retries=2)
        result = self.build(AutoApproveReviewer(), llm=llm).run(QUESTION)
        self.assertEqual(result.status, "failed")
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_threshold_unreachable_flags_coverage(self):
        s = replace(self.settings, record_threshold=10)
        result = self.build(AutoApproveReviewer(), settings=s).run(QUESTION)
        self.assertTrue(all(sq.insufficient_coverage for sq in result.brief.sub_questions))
        self.assertTrue(any("fewer than 10" in l for l in result.brief.limitations))
        self.assertEqual(result.stats["reformulations"], 3)  # one per sub-question, no more

    def test_dedup_merges_subquestion_links_and_keeps_abstract(self):
        a = paper(1, abstract_words=0, subquestion_ids=[1])
        b = paper(1, subquestion_ids=[2])
        [merged] = _O.deduplicate_by_doi([a, b])
        self.assertEqual(merged.subquestion_ids, [1, 2])
        self.assertIsNotNone(merged.abstract)


class TestCli(unittest.TestCase):
    def test_offline_cli_run_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            import os
            old = dict(os.environ)
            os.environ.update(OUTPUT_DIR=f"{d}/out", LOG_DIR=f"{d}/logs")
            try:
                buf = StringIO()
                with redirect_stdout(buf):
                    code = main([QUESTION, "--offline", "--auto-approve", "--quiet",
                                 "--relevance-cutoff", "0.5"])
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertEqual(code, 0)
            self.assertIn("completed", buf.getvalue())
            self.assertEqual(len(list(Path(d, "out").glob("*.md"))), 1)


if __name__ == "__main__":
    unittest.main()
