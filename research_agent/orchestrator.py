"""Deterministic orchestrator: sequences the agents and owns every exact rule.

Why a deterministic coordinator rather than a fourth "manager" LLM agent:
multi-agent gains are task-dependent, and architectures without a central
verification point propagate errors more strongly (Kim et al., 2026). Here
all hand-offs pass through one class that (a) validates state, (b) enforces
the autonomy budget (one automatic reformulation per sub-question, one
researcher scope revision per run), (c) deduplicates and validates DOIs, and
(d) logs every transition. The control flow is a direct transcription of
Unit 6 Diagrams 2 and 3, so the design can be checked against the code.

Role separation follows the Gaia methodology (Wooldridge, Jennings and Kinny,
2000): each agent has one role and responsibilities; the interaction protocol
lives here, not inside the agents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .agents import EvaluationSynthesisAgent, PlanningAgent, RetrievalAgent
from .config import Settings
from .llm import LLMOutputError
from .models import Brief, Constraints, Decision, EvaluationResult, Paper, SubQuestion
from .review import ResearcherInterface
from .sources import CrossrefValidator, SourceUnavailable
from .storage import save_brief

log = logging.getLogger(__name__)

# Unit 6 allows unlimited revisions at checkpoint 1 because they are cheap.
# In practice an unbounded loop is a hang risk for scripted/batch reviewers,
# so a high safety cap is applied; a human will never plausibly reach it.
MAX_PLAN_REVISIONS_SAFETY = 10


@dataclass
class RunResult:
    run_id: str
    status: str  # "completed" | "stopped_scope" | "no_evidence" | "failed"
    brief: Brief | None = None
    paths: tuple[Path, Path] | None = None
    message: str = ""
    stats: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, settings: Settings, planner: PlanningAgent, retriever: RetrievalAgent,
                 evaluator: EvaluationSynthesisAgent, validator: CrossrefValidator,
                 reviewer: ResearcherInterface):
        self.s = settings
        self.planner, self.retriever, self.evaluator = planner, retriever, evaluator
        self.validator, self.reviewer = validator, reviewer
        self.retry_counts: dict[int, int] = {}
        self.scope_revision_count = 0

    # ------------------------------------------------------------------ run
    def run(self, question: str, constraints: Constraints | None = None,
            run_id: str | None = None) -> RunResult:
        constraints = constraints or Constraints()
        run_id = run_id or uuid.uuid4().hex[:8]
        log.info("run_start run=%s question=%r constraints=%s", run_id, question,
                 constraints.model_dump())
        try:
            return self._run(question, constraints, run_id)
        except (LLMOutputError, SourceUnavailable) as exc:
            # The run fails closed: no partial brief is written when the model
            # cannot produce valid output, or when every literature source is
            # unreachable. Remediation R1: SourceUnavailable originally escaped
            # as a raw traceback with nothing in the run log (TESTING.md).
            log.error("run_failed run=%s reason=%s: %s", run_id, type(exc).__name__, exc)
            return RunResult(run_id, "failed", message=str(exc), stats=self._stats())
        except (EOFError, KeyboardInterrupt):
            # Remediation R5: closing the input (Ctrl-D) or pressing Ctrl-C at a
            # review prompt raised a traceback. Abandoning a checkpoint is a
            # legitimate researcher decision, so it is logged as a cancellation
            # and, as with failure, nothing is saved.
            log.info("run_cancelled run=%s at=human_review", run_id)
            return RunResult(run_id, "cancelled", message="Run cancelled at a review "
                             "checkpoint; nothing was saved.", stats=self._stats())

    def _run(self, question: str, c: Constraints, run_id: str) -> RunResult:
        subs = self._plan_with_review(question, c, self.planner.plan_research(question, c))

        while True:
            papers = self._retrieve_all(subs, c)
            papers = self.deduplicate_by_doi(papers)
            papers = self.validate_dois(papers)
            evaluation = self.evaluator.evaluate_evidence(papers, subs)
            by_id = {p.paper_id: p for p in papers}

            review = self.reviewer.review_evidence(evaluation.assessments, by_id)
            log.info("checkpoint2 run=%s decision=%s", run_id, review.decision.value)
            if review.decision is Decision.REJECT_SCOPE:
                if self.scope_revision_count >= self.s.max_scope_revisions:
                    msg = "Scope rejected twice; restart with a rephrased research question."
                    log.info("run_stopped run=%s reason=scope_limit", run_id)
                    return RunResult(run_id, "stopped_scope", message=msg, stats=self._stats())
                self.scope_revision_count += 1
                self.retry_counts.clear()  # new sub-questions get a fresh retry budget
                revised = self.planner.revise_plan(question, c, review.feedback)
                subs = self._plan_with_review(question, c, revised)
                continue
            approved = self._approved_papers(evaluation, by_id, review.deselected)
            break

        if not approved:
            log.info("run_no_evidence run=%s", run_id)
            return RunResult(run_id, "no_evidence", message="No evidence was approved; "
                             "nothing to synthesise.", stats=self._stats())

        summaries = self.evaluator.synthesise(question, approved)
        brief = Brief(research_question=question, sub_questions=subs, selected_papers=approved,
                      summaries=summaries, themes=evaluation.themes, gaps=evaluation.gaps,
                      limitations=self._limitations(subs, approved, c), run_id=run_id)
        paths = save_brief(brief, self.s.output_dir)
        log.info("run_completed run=%s papers=%d md=%s", run_id, len(approved), paths[0])
        return RunResult(run_id, "completed", brief, paths, stats=self._stats())

    # ---------------------------------------------------------- checkpoints
    def _plan_with_review(self, question: str, c: Constraints,
                          subs: list[SubQuestion]) -> list[SubQuestion]:
        for _ in range(MAX_PLAN_REVISIONS_SAFETY):
            review = self.reviewer.review_subquestions(subs)
            log.info("checkpoint1 decision=%s feedback=%r", review.decision.value, review.feedback)
            if review.decision is Decision.APPROVE:
                return [s.model_copy(update={"approved": True}) for s in subs]
            subs = self.planner.revise_plan(question, c, review.feedback)
        raise LLMOutputError("sub-question review did not converge")

    # ------------------------------------------------------------ retrieval
    def _retrieve_all(self, subs: list[SubQuestion], c: Constraints) -> list[Paper]:
        papers: list[Paper] = []
        for i, sub in enumerate(subs):
            outcome = self.retriever.retrieve_evidence(sub, c)
            if outcome.reformulation_requested and \
                    self.retry_counts.get(sub.id, 0) < self.s.max_auto_reformulations:
                self.retry_counts[sub.id] = self.retry_counts.get(sub.id, 0) + 1
                try:
                    new_query = self.planner.reformulate_query(sub, outcome.usable)
                    retry = self.retriever.retrieve_evidence(sub, c, new_query)
                    # Keep the better of the two result sets rather than
                    # blindly trusting the reformulation.
                    if retry.usable >= outcome.usable:
                        sub = sub.model_copy(update={"search_query": new_query})
                        outcome = retry
                    sub = sub.model_copy(update={"retried_once": True})
                except LLMOutputError as exc:
                    log.warning("reformulation_failed subq=%d reason=%s", sub.id, exc)
            if not outcome.threshold_met:
                sub = sub.model_copy(update={"insufficient_coverage": True})
                log.warning("insufficient_coverage subq=%d usable=%d", sub.id, outcome.usable)
            subs[i] = sub
            papers += outcome.papers
        return papers

    @staticmethod
    def deduplicate_by_doi(papers: list[Paper]) -> list[Paper]:
        """Merge duplicates, keeping the first record and the union of sub-question links."""
        merged: dict[str, Paper] = {}
        for p in papers:
            key = p.dedup_key
            if key in merged:
                keep = merged[key]
                ids = sorted(set(keep.subquestion_ids) | set(p.subquestion_ids))
                # Prefer whichever record carries an abstract.
                base = keep if keep.abstract or not p.abstract else p
                merged[key] = base.model_copy(update={"subquestion_ids": ids})
            else:
                merged[key] = p
        log.info("deduplicated before=%d after=%d", len(papers), len(merged))
        return list(merged.values())

    def validate_dois(self, papers: list[Paper]) -> list[Paper]:
        out = [self.validator.validate(p) for p in papers]
        log.info("doi_validation verified=%d unverified=%d",
                 sum(p.doi_verified for p in out), sum(not p.doi_verified for p in out))
        for p in out:
            if not p.doi_verified:
                log.info("doi_flag paper=%s doi=%s flag=%s", p.paper_id, p.doi, p.metadata_flag)
        return out

    @staticmethod
    def _approved_papers(ev: EvaluationResult, by_id: dict[str, Paper],
                         deselected: set[str]) -> list[Paper]:
        ids: list[str] = []
        for a in ev.assessments:  # already sorted by score
            if a.selected and a.paper_id not in deselected and a.paper_id not in ids:
                ids.append(a.paper_id)
        return [by_id[i] for i in ids]

    # -------------------------------------------------------- transparency
    def _limitations(self, subs: list[SubQuestion], approved: list[Paper],
                     c: Constraints) -> list[str]:
        """Generated deterministically from run facts, so they cannot be hallucinated."""
        lim = ["Relevance and summaries are based on titles and abstracts only; methods, "
               "results and limitations in full texts were not read.",
               f"Search source: {self.retriever.search.name}; papers outside its index are missed.",
               "Relevance scores are LLM judgements pre-selecting evidence for human approval, "
               "not a quality appraisal."]
        thin = [s.id for s in subs if s.insufficient_coverage]
        if thin:
            lim.append(f"Sub-question(s) {thin} returned fewer than {self.s.record_threshold} "
                       "usable records even after reformulation.")
        unverified = [p for p in approved if not p.doi_verified]
        if unverified:
            lim.append(f"{len(unverified)} selected paper(s) could not be verified against "
                       "Crossref; check their citation details manually.")
        if c.year_from or c.year_to:
            lim.append(f"Restricted to publication years {c.year_from or '…'}–{c.year_to or '…'}.")
        if getattr(self.reviewer, "automatic", False):
            lim.append("Run in auto-approve mode: no human reviewed the sub-questions or evidence.")
        return lim

    def _stats(self) -> dict:
        return {"llm_calls": self.planner.llm.calls,
                "reformulations": sum(self.retry_counts.values()),
                "scope_revisions": self.scope_revision_count}
