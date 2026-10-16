"""Evaluation and Synthesis Agent: relevance scoring, theme/gap analysis, summaries.

Evaluation and synthesis are one agent because synthesis consumes evaluation
output directly; a separate agent would add a hand-off (another place for
state to drift) with no new decision in between (Unit 6, Section 4).

Grounding rules enforced in code, not trusted to the prompt:
* Assessments or summaries that reference a paper id or sub-question id the
  model was not given are discarded and logged. This is a cheap, exact guard
  against the model inventing sources (Walters and Wilder, 2023).
* Summaries are generated only from *approved* papers' abstracts
  (content grounding; Maynez et al., 2020).
* The LLM's score proposes a selection; the researcher's review at
  checkpoint 2 decides it. The model acts as an assessor, not the judge of
  record, because LLM judges show position and verbosity biases
  (Zheng et al., 2023).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from .. import prompts
from ..llm import LLMClient
from ..models import Assessment, EvaluationResult, Paper, PaperSummary, SubQuestion

log = logging.getLogger(__name__)


class _SummariesOut(BaseModel):
    summaries: list[PaperSummary]


class EvaluationSynthesisAgent:
    name = "evaluation_synthesis"

    def __init__(self, llm: LLMClient, relevance_cutoff: float = 0.6, batch_size: int = 10,
                 abstract_word_limit: int = 220):
        self.llm, self.cutoff = llm, relevance_cutoff
        # Batching bounds prompt length; long inputs dilute attention to
        # items in the middle of the context (Liu et al., 2024).
        self.batch_size, self.word_limit = batch_size, abstract_word_limit

    def _paper_payload(self, p: Paper) -> dict:
        words = (p.abstract or "").split()
        return {"paper_id": p.paper_id, "title": p.title, "year": p.year,
                "abstract": " ".join(words[: self.word_limit])}

    def evaluate_evidence(self, papers: list[Paper], subs: list[SubQuestion]) -> EvaluationResult:
        valid_papers = {p.paper_id for p in papers}
        valid_subs = {s.id for s in subs}
        sub_payload = [{"id": s.id, "text": s.text} for s in subs]
        assessments: dict[tuple[str, int], Assessment] = {}
        themes, gaps = [], []
        for start in range(0, len(papers), self.batch_size):
            batch = papers[start:start + self.batch_size]
            user = prompts.evaluate_prompt(sub_payload, [self._paper_payload(p) for p in batch])
            out = self.llm.complete_json("evaluate", prompts.EVALUATOR_SYSTEM, user,
                                         EvaluationResult)
            for a in out.assessments:
                if a.paper_id not in valid_papers or a.subquestion_id not in valid_subs:
                    log.warning("discarded_ungrounded_assessment paper=%s subq=%s",
                                a.paper_id, a.subquestion_id)
                    continue
                key = (a.paper_id, a.subquestion_id)
                # Keep the highest score if the model repeats a pair.
                if key not in assessments or a.relevance_score > assessments[key].relevance_score:
                    assessments[key] = a.model_copy(update={"selected": a.relevance_score >= self.cutoff})
            themes += out.themes
            gaps += out.gaps
        result = EvaluationResult(assessments=sorted(assessments.values(),
                                                     key=lambda a: -a.relevance_score),
                                  themes=_dedupe(themes), gaps=_dedupe(gaps))
        log.info("evaluated papers=%d assessments=%d preselected=%d",
                 len(papers), len(result.assessments),
                 len({a.paper_id for a in result.assessments if a.selected}))
        return result

    def synthesise(self, question: str, approved: list[Paper]) -> list[PaperSummary]:
        if not approved:
            return []
        valid = {p.paper_id for p in approved}
        user = prompts.synthesise_prompt(question, [self._paper_payload(p) for p in approved])
        out = self.llm.complete_json("synthesise", prompts.EVALUATOR_SYSTEM, user, _SummariesOut)
        summaries = {s.paper_id: s for s in out.summaries if s.paper_id in valid}
        dropped = len(out.summaries) - len(summaries)
        if dropped:
            log.warning("discarded_ungrounded_summaries n=%d", dropped)
        # A missing summary is reported, never silently omitted, so the brief
        # cannot look more complete than the evidence supports.
        for p in approved:
            summaries.setdefault(p.paper_id, PaperSummary(
                paper_id=p.paper_id, summary="[No summary produced; read the abstract.]"))
        return [summaries[p.paper_id] for p in approved]


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out
