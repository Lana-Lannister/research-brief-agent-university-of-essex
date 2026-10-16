"""Unit tests: each agent in isolation, with a scripted LLM."""

import unittest

from research_agent.agents import EvaluationSynthesisAgent, PlanningAgent, RetrievalAgent
from research_agent.llm import LLMOutputError, ScriptedLLM
from research_agent.models import Constraints, SubQuestion
from tests.helpers import paper, subs


def _plan(*queries):
    return {"sub_questions": [{"id": i + 1, "text": f"Sub-question about {q}", "search_query": q}
                              for i, q in enumerate(queries)]}


class TestPlanningAgent(unittest.TestCase):
    def test_plan_renumbers_and_marks_unapproved(self):
        llm = ScriptedLLM([_plan("a b", "c d", "e f")])
        out = PlanningAgent(llm).plan_research("Question?", Constraints())
        self.assertEqual([s.id for s in out], [1, 2, 3])
        self.assertFalse(any(s.approved for s in out))

    def test_duplicate_queries_removed_then_too_few_reprompt_fails(self):
        llm = ScriptedLLM([_plan("a b", "A B", "c d")])
        with self.assertRaises(LLMOutputError):
            PlanningAgent(llm, n_min=3).plan_research("Q?", Constraints())

    def test_too_many_truncated_to_max(self):
        llm = ScriptedLLM([_plan(*[f"q{i} x" for i in range(7)])])
        self.assertEqual(len(PlanningAgent(llm, n_max=5).plan_research("Q?", Constraints())), 5)

    def test_feedback_is_included_in_revision_prompt(self):
        llm = ScriptedLLM([_plan("a b", "c d", "e f")])
        PlanningAgent(llm).revise_plan("Q?", Constraints(), "focus on healthcare")
        self.assertIn("focus on healthcare", llm.prompts[0][1])

    def test_reformulation_must_differ(self):
        sub = SubQuestion(id=1, text="About agents", search_query="agents review")
        self.assertEqual(PlanningAgent(ScriptedLLM([{"search_query": "agents"}]))
                         .reformulate_query(sub, 1), "agents")
        with self.assertRaises(LLMOutputError):
            PlanningAgent(ScriptedLLM([{"search_query": "Agents Review"}])).reformulate_query(sub, 1)


class TestRetrievalAgent(unittest.TestCase):
    class _Search:
        name = "stub"

        def __init__(self, papers):
            self.papers = papers

        def search(self, q, c):
            return self.papers

    def test_threshold_counts_only_usable_abstracts(self):
        papers = [paper(1), paper(2), paper(3, abstract_words=5), paper(4, abstract_words=0)]
        agent = RetrievalAgent(self._Search(papers), record_threshold=3)
        out = agent.retrieve_evidence(subs(1)[0], Constraints())
        self.assertEqual(out.usable, 2)
        self.assertFalse(out.threshold_met)
        self.assertTrue(out.reformulation_requested)
        self.assertTrue(all(p.subquestion_ids == [1] for p in out.papers))

    def test_threshold_met(self):
        agent = RetrievalAgent(self._Search([paper(i) for i in range(3)]), record_threshold=3)
        self.assertTrue(agent.retrieve_evidence(subs(1)[0], Constraints()).threshold_met)


class TestEvaluationAgent(unittest.TestCase):
    def test_discards_ungrounded_ids_and_applies_cutoff(self):
        reply = {"assessments": [
            {"paper_id": "P1", "subquestion_id": 1, "relevance_score": 0.9, "rationale": "r"},
            {"paper_id": "P2", "subquestion_id": 1, "relevance_score": 0.4, "rationale": "r"},
            {"paper_id": "INVENTED", "subquestion_id": 1, "relevance_score": 1.0, "rationale": "r"},
            {"paper_id": "P1", "subquestion_id": 99, "relevance_score": 1.0, "rationale": "r"}],
            "themes": ["T", "t "], "gaps": ["G"]}
        agent = EvaluationSynthesisAgent(ScriptedLLM([reply]), relevance_cutoff=0.6)
        out = agent.evaluate_evidence([paper(1), paper(2)], subs(2))
        self.assertEqual([(a.paper_id, a.selected) for a in out.assessments],
                         [("P1", True), ("P2", False)])
        self.assertEqual(out.themes, ["T"])  # case-insensitive dedupe

    def test_batches_long_inputs(self):
        empty = {"assessments": [], "themes": [], "gaps": []}
        llm = ScriptedLLM([empty, empty, empty])
        EvaluationSynthesisAgent(llm, batch_size=4).evaluate_evidence(
            [paper(i) for i in range(10)], subs(1))
        self.assertEqual(llm.calls, 3)

    def test_abstracts_are_truncated_in_prompt(self):
        llm = ScriptedLLM([{"assessments": [], "themes": [], "gaps": []}])
        EvaluationSynthesisAgent(llm, abstract_word_limit=10).evaluate_evidence(
            [paper(1, abstract_words=500)], subs(1))
        self.assertLess(llm.prompts[0][1].count("word"), 20)

    def test_synthesis_only_for_approved_and_fills_missing(self):
        reply = {"summaries": [{"paper_id": "P1", "summary": "S1"},
                               {"paper_id": "GHOST", "summary": "invented"}]}
        out = EvaluationSynthesisAgent(ScriptedLLM([reply])).synthesise("Q", [paper(1), paper(2)])
        self.assertEqual([s.paper_id for s in out], ["P1", "P2"])
        self.assertEqual(out[0].summary, "S1")
        self.assertIn("No summary", out[1].summary)

    def test_no_llm_call_when_nothing_approved(self):
        llm = ScriptedLLM([])
        self.assertEqual(EvaluationSynthesisAgent(llm).synthesise("Q", []), [])
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
