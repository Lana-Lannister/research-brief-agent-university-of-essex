"""Planning Agent: goal -> sub-questions -> search queries.

Planning by decomposition is the agent's pro-active behaviour: it turns one
under-specified goal into a set of actionable sub-goals (Wang et al., 2024;
Wooldridge and Jennings, 1995). The LLM proposes; this class enforces the
structural rules the model may break (count, uniqueness, id order), because
those are exact checks that should not depend on model compliance.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from .. import prompts
from ..llm import LLMClient, LLMOutputError
from ..models import Constraints, SubQuestion

log = logging.getLogger(__name__)


class _PlanOut(BaseModel):
    sub_questions: list[SubQuestion] = Field(min_length=1)


class _QueryOut(BaseModel):
    search_query: str = Field(min_length=2)


class PlanningAgent:
    name = "planning"

    def __init__(self, llm: LLMClient, n_min: int = 3, n_max: int = 5):
        self.llm, self.n_min, self.n_max = llm, n_min, n_max

    def plan_research(self, question: str, constraints: Constraints,
                      feedback: str | None = None) -> list[SubQuestion]:
        user = prompts.plan_prompt(question, constraints.model_dump(), self.n_min,
                                   self.n_max, feedback)
        out = self.llm.complete_json("plan", prompts.PLANNER_SYSTEM, user, _PlanOut)
        subs = self._enforce_structure(out.sub_questions)
        log.info("plan_created n=%d feedback=%s queries=%s", len(subs), bool(feedback),
                 [s.search_query for s in subs])
        return subs

    def revise_plan(self, question: str, constraints: Constraints,
                    feedback: str) -> list[SubQuestion]:
        # Revision regenerates the whole plan with the researcher's feedback,
        # rather than patching single items, so the sub-questions stay
        # mutually consistent (non-overlapping) after the change.
        return self.plan_research(question, constraints, feedback)

    def reformulate_query(self, sub: SubQuestion, hits: int) -> str:
        user = prompts.reformulate_prompt(sub.text, sub.search_query, hits)
        new = self.llm.complete_json("reformulate", prompts.PLANNER_SYSTEM, user, _QueryOut)
        query = new.search_query.strip().strip('"')
        if query.lower() == sub.search_query.lower():
            # Re-running an identical query cannot change the result; fail
            # loudly so the orchestrator flags coverage instead of wasting a call.
            raise LLMOutputError("reformulation returned the original query")
        log.info("query_reformulated subq=%d old=%r new=%r", sub.id, sub.search_query, query)
        return query

    def _enforce_structure(self, subs: list[SubQuestion]) -> list[SubQuestion]:
        seen, unique = set(), []
        for s in subs:
            key = s.search_query.lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        if len(unique) < self.n_min:
            raise LLMOutputError(f"planner returned {len(unique)} distinct sub-questions; "
                                 f"need at least {self.n_min}")
        # Truncate rather than reject when there are too many: extra
        # sub-questions are usable, just over budget (API calls, review effort).
        unique = unique[: self.n_max]
        return [s.model_copy(update={"id": i, "approved": False}) for i, s in enumerate(unique, 1)]
