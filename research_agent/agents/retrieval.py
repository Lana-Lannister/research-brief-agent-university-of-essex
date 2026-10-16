"""Retrieval Agent: executes searches and judges whether they succeeded.

The agent is *reactive*: it perceives the outcome of its own action (how many
usable records came back) and, when the outcome falls short, requests a
reformulation instead of passing thin evidence downstream. The request goes
through the orchestrator, not directly to the planner, so the retry limit is
enforced in one place (Unit 6, Diagram 2 steps 16-25).

The threshold test is deliberately deterministic: "enough records with an
abstract" is countable, so spending an LLM call on it would add cost and
nondeterminism for no gain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..models import Constraints, Paper, SubQuestion
from ..sources import LiteratureSearch

log = logging.getLogger(__name__)


@dataclass
class RetrievalOutcome:
    papers: list[Paper]
    usable: int
    threshold_met: bool

    @property
    def reformulation_requested(self) -> bool:
        return not self.threshold_met


class RetrievalAgent:
    name = "retrieval"

    def __init__(self, search: LiteratureSearch, record_threshold: int = 3):
        self.search, self.record_threshold = search, record_threshold

    def retrieve_evidence(self, sub: SubQuestion, constraints: Constraints,
                          query: str | None = None) -> RetrievalOutcome:
        q = query or sub.search_query
        papers = [p.model_copy(update={"subquestion_ids": [sub.id]})
                  for p in self.search.search(q, constraints)]
        usable = self.count_usable(papers)
        met = self.assess_threshold(usable)
        log.info("retrieved subq=%d query=%r source=%s records=%d usable=%d threshold_met=%s",
                 sub.id, q, self.search.name, len(papers), usable, met)
        return RetrievalOutcome(papers, usable, met)

    @staticmethod
    def count_usable(papers: list[Paper]) -> int:
        # The system works at abstract level (Unit 6 scope), so a record
        # without an abstract cannot be evaluated and does not count.
        return sum(1 for p in papers if p.abstract and len(p.abstract.split()) >= 20)

    def assess_threshold(self, usable: int) -> bool:
        return usable >= self.record_threshold
