"""Human-review checkpoints (the «boundary» ResearcherInterface in Diagram 1).

Two checkpoints keep the researcher in evaluative control while agents do the
operative work (Zhu et al., 2026): after decomposition (cheap to correct,
unlimited revisions) and after evidence selection (expensive to correct,
scope rejection capped). This sits at Parasuraman, Sheridan and Wickens'
(2000) middle levels of automation: the system proposes, the human approves.

Reviewers are interchangeable so the same orchestrator runs interactively,
in batch (--auto-approve, logged as such), or in tests (scripted decisions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import Assessment, Decision, Paper, SubQuestion


@dataclass
class ReviewResult:
    decision: Decision
    feedback: str = ""
    deselected: set[str] = field(default_factory=set)


class ResearcherInterface(Protocol):
    def review_subquestions(self, subs: list[SubQuestion]) -> ReviewResult: ...

    def review_evidence(self, assessments: list[Assessment],
                        papers: dict[str, Paper]) -> ReviewResult: ...


class AutoApproveReviewer:
    """Approves everything. For unattended demos only; the log records that no human reviewed."""

    automatic = True

    def review_subquestions(self, subs):
        return ReviewResult(Decision.APPROVE)

    def review_evidence(self, assessments, papers):
        return ReviewResult(Decision.APPROVE)


class ScriptedReviewer:
    """Replays a fixed list of decisions (functional tests of each review branch)."""

    automatic = False

    def __init__(self, subq: list[ReviewResult], evidence: list[ReviewResult]):
        self.subq, self.evidence = list(subq), list(evidence)
        self.seen_subq: list[list[SubQuestion]] = []
        self.seen_evidence: list[list[Assessment]] = []

    def review_subquestions(self, subs):
        self.seen_subq.append(subs)
        return self.subq.pop(0)

    def review_evidence(self, assessments, papers):
        self.seen_evidence.append(assessments)
        return self.evidence.pop(0)


class ConsoleReviewer:
    """Interactive terminal checkpoints."""

    automatic = False

    def __init__(self, input_fn=input, print_fn=print):
        self.input, self.print = input_fn, print_fn

    def review_subquestions(self, subs):
        self.print("\n=== Human Review 1: sub-questions ===")
        for s in subs:
            self.print(f"  [{s.id}] {s.text}\n      query: {s.search_query}")
        choice = self.input("Approve (a) or request revision (r)? ").strip().lower()
        if choice.startswith("r"):
            return ReviewResult(Decision.REVISE, self.input("Describe the revision: ").strip())
        return ReviewResult(Decision.APPROVE)

    def review_evidence(self, assessments, papers):
        self.print("\n=== Human Review 2: selected evidence ===")
        selected = [a for a in assessments if a.selected]
        shown = set()
        for a in selected:
            if a.paper_id in shown:
                continue
            shown.add(a.paper_id)
            p = papers[a.paper_id]
            tick = "verified" if p.doi_verified else f"UNVERIFIED ({p.metadata_flag})"
            self.print(f"  {a.paper_id}  score={a.relevance_score:.2f}  SQ{a.subquestion_id}\n"
                       f"      {p.title} ({p.year}) DOI {p.doi or '-'} [{tick}]\n"
                       f"      why: {a.rationale}")
        choice = self.input("Approve (a), refine by deselecting (f), or reject scope (x)? ").strip().lower()
        if choice.startswith("f"):
            return ReviewResult(Decision.REFINE, deselected=self._ask_ids(shown))
        if choice.startswith("x"):
            return ReviewResult(Decision.REJECT_SCOPE, self.input("What is wrong with the scope? ").strip())
        return ReviewResult(Decision.APPROVE)

    def _ask_ids(self, valid: set[str]) -> set[str]:
        """Resolve typed ids against the papers shown, re-asking on unknown ids.

        Remediation R3: ids were originally accepted verbatim, so a typo or a
        missing "S2:" prefix silently deselected nothing and the unwanted
        paper reached the brief. At a human checkpoint a silent no-op is
        worse than an error, because the researcher believes they acted.
        """
        by_suffix = {v.split(":", 1)[-1]: v for v in valid}
        while True:
            typed = self.input("Paper ids to remove (space separated): ").split()
            resolved = {t if t in valid else by_suffix.get(t) for t in typed}
            if None not in resolved:
                return resolved
            unknown = [t for t in typed if t not in valid and t not in by_suffix]
            self.print(f"Unknown id(s): {' '.join(unknown)}. Choose from: {' '.join(sorted(valid))}")
