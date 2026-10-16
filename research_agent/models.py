"""Typed messages exchanged between agents and the orchestrator.

Why Pydantic: every hand-off between agents is a place where an LLM's free text
could leak malformed state into the next stage. Validating each message at the
boundary turns "the model returned something odd" into an explicit, logged
ValidationError at the point of failure, instead of a confusing error three
stages later. This is the "typed messages and validated states" mitigation for
error propagation across hand-offs in the Unit 6 risk table (Kim et al., 2026).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# A DOI is "10." + registrant code + "/" + suffix (DOI Foundation syntax).
# The pattern is deliberately permissive on the suffix because real suffixes
# contain parentheses, semicolons and similar characters (e.g. older Wiley
# DOIs), so validating the suffix strictly would reject genuine records.
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")


def normalise_doi(raw: Optional[str]) -> Optional[str]:
    """Return a canonical lower-case DOI, or None if the input is not a DOI.

    DOIs are case-insensitive by specification, but sources disagree on case
    and on prefixes (Semantic Scholar returns bare DOIs, OpenAlex returns
    https://doi.org/ URLs). Normalising once here is what makes DOI-based
    deduplication exact rather than heuristic.
    """
    if not raw:
        return None
    doi = raw.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "http://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    doi = doi.rstrip(".")
    return doi if _DOI_PATTERN.match(doi) else None


class Decision(str, Enum):
    """Outcomes a researcher can return at a human-review checkpoint."""

    APPROVE = "APPROVE"
    REVISE = "REVISE"          # checkpoint 1: ask planner to revise sub-questions
    REFINE = "REFINE"          # checkpoint 2: deselect some papers, keep the rest
    REJECT_SCOPE = "REJECT_SCOPE"  # checkpoint 2: scope is wrong, replan (capped)


class Constraints(BaseModel):
    """Researcher-supplied limits that bound the search."""

    year_from: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1900, le=2100)
    max_papers_per_subquestion: int = Field(default=8, ge=1, le=50)


class SubQuestion(BaseModel):
    id: int = Field(ge=1)
    text: str = Field(min_length=5)
    search_query: str = Field(min_length=2)
    approved: bool = False
    retried_once: bool = False
    insufficient_coverage: bool = False

    @field_validator("search_query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        # LLMs sometimes wrap queries in quotes; Semantic Scholar then treats
        # the whole string as a phrase and recall collapses.
        return value.strip().strip('"').strip("'").strip()


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    source: str = "semantic_scholar"
    subquestion_ids: list[int] = Field(default_factory=list)
    doi_verified: bool = False
    metadata_flag: Optional[str] = None  # why verification failed, if it did

    @field_validator("doi", mode="before")
    @classmethod
    def _normalise(cls, value: Optional[str]) -> Optional[str]:
        return normalise_doi(value)

    @property
    def dedup_key(self) -> str:
        # Fall back to a normalised title only when no DOI exists, because
        # titles collide across editions and preprints far more than DOIs.
        if self.doi:
            return f"doi:{self.doi}"
        return "title:" + re.sub(r"[^a-z0-9]", "", self.title.lower())


class Assessment(BaseModel):
    paper_id: str
    subquestion_id: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    selected: bool = False


class EvaluationResult(BaseModel):
    assessments: list[Assessment]
    themes: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    paper_id: str
    summary: str


class Brief(BaseModel):
    """Final research brief. Every summary must point at an approved paper."""

    research_question: str
    sub_questions: list[SubQuestion]
    selected_papers: list[Paper]
    summaries: list[PaperSummary]
    themes: list[str]
    gaps: list[str]
    limitations: list[str]
    run_id: str
