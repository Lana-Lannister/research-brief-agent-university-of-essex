"""External scholarly APIs: search (Semantic Scholar, OpenAlex) and DOI validation (Crossref).

Design decisions
----------------
* Retrieval goes to curated scholarly indexes, never to the LLM. Asking a
  model for references invites fabricated or corrupted citations (Walters
  and Wilder, 2023); an index returns records that exist.
* Semantic Scholar is primary for its abstract coverage (Kinney et al., 2023);
  OpenAlex is the fallback (Priem, Piwowar and Orr, 2022). Fallback is
  triggered by *transport* failure only, not by low recall, so the two
  sources are never silently mixed for one query and provenance stays clear.
* Crossref is an independent authority for DOI metadata (Hendricks et al.,
  2020). Verification compares title and year, because a DOI that resolves
  but belongs to a different paper is exactly the corruption we must catch.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from pathlib import Path
from typing import Protocol

import requests

from .models import Constraints, Paper, normalise_doi

log = logging.getLogger(__name__)


class SourceUnavailable(RuntimeError):
    pass


def _get_json(session: requests.Session, url: str, params: dict, headers: dict,
              timeout: float, retries: int = 3) -> dict:
    """GET with exponential back-off on 429/5xx.

    Public scholarly APIs throttle unauthenticated clients aggressively;
    back-off with a small retry budget is the standard remedy and keeps the
    run deterministic in length (bounded by retries x max wait).
    """
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise SourceUnavailable(str(exc)) from exc
        else:
            if resp.status_code == 404:
                return {}
            if resp.status_code < 400:
                return resp.json()
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise SourceUnavailable(f"{url} -> HTTP {resp.status_code}")
        log.info("http_retry url=%s attempt=%d sleep=%.1fs", url, attempt + 1, delay)
        time.sleep(delay)
        delay *= 2
    raise SourceUnavailable(url)  # pragma: no cover


class LiteratureSearch(Protocol):
    name: str

    def search(self, query: str, constraints: Constraints) -> list[Paper]: ...


class SemanticScholarSearch:
    name = "semantic_scholar"
    URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,authors,externalIds"

    def __init__(self, api_key: str = "", timeout: float = 20.0,
                 session: requests.Session | None = None):
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.timeout, self.session = timeout, session or requests.Session()

    def search(self, query: str, constraints: Constraints) -> list[Paper]:
        params = {"query": query, "fields": self.FIELDS,
                  "limit": constraints.max_papers_per_subquestion}
        if constraints.year_from or constraints.year_to:
            params["year"] = f"{constraints.year_from or ''}-{constraints.year_to or ''}"
        data = _get_json(self.session, self.URL, params, self.headers, self.timeout)
        return [p for p in (self.parse(r) for r in data.get("data", [])) if p]

    @staticmethod
    def parse(record: dict) -> Paper | None:
        if not record.get("title") or not record.get("paperId"):
            return None
        ids = record.get("externalIds") or {}
        return Paper(paper_id=f"S2:{record['paperId']}", title=record["title"].strip(),
                     authors=[a.get("name", "") for a in record.get("authors") or []],
                     year=record.get("year"), abstract=record.get("abstract"),
                     doi=ids.get("DOI"), source="semantic_scholar")


class OpenAlexSearch:
    name = "openalex"
    URL = "https://api.openalex.org/works"

    def __init__(self, mailto: str = "", timeout: float = 20.0,
                 session: requests.Session | None = None):
        self.mailto, self.timeout = mailto, timeout
        self.session = session or requests.Session()

    def search(self, query: str, constraints: Constraints) -> list[Paper]:
        params = {"search": query, "per-page": constraints.max_papers_per_subquestion}
        filters = []
        if constraints.year_from:
            filters.append(f"from_publication_date:{constraints.year_from}-01-01")
        if constraints.year_to:
            filters.append(f"to_publication_date:{constraints.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if self.mailto:
            params["mailto"] = self.mailto
        data = _get_json(self.session, self.URL, params, {}, self.timeout)
        return [p for p in (self.parse(r) for r in data.get("results", [])) if p]

    @staticmethod
    def rebuild_abstract(inverted: dict | None) -> str | None:
        # OpenAlex stores abstracts as {word: [positions]} for licensing
        # reasons; rebuilding word order is required before any LLM use.
        if not inverted:
            return None
        slots = {pos: word for word, positions in inverted.items() for pos in positions}
        return " ".join(slots[i] for i in sorted(slots))

    @classmethod
    def parse(cls, record: dict) -> Paper | None:
        if not record.get("title") or not record.get("id"):
            return None
        return Paper(paper_id="OA:" + record["id"].rsplit("/", 1)[-1],
                     title=record["title"].strip(),
                     authors=[a["author"]["display_name"] for a in record.get("authorships", [])
                              if a.get("author", {}).get("display_name")],
                     year=record.get("publication_year"),
                     abstract=cls.rebuild_abstract(record.get("abstract_inverted_index")),
                     doi=record.get("doi"), source="openalex")


class FallbackSearch:
    """Primary source, falling back to a secondary one on transport failure."""

    def __init__(self, primary: LiteratureSearch, secondary: LiteratureSearch):
        self.primary, self.secondary = primary, secondary
        self.name = f"{primary.name}+{secondary.name}"

    def search(self, query: str, constraints: Constraints) -> list[Paper]:
        try:
            return self.primary.search(query, constraints)
        except SourceUnavailable as exc:
            log.warning("source_fallback from=%s to=%s reason=%s",
                        self.primary.name, self.secondary.name, exc)
            return self.secondary.search(query, constraints)


class FixtureSearch:
    """Replays recorded search responses from disk (tests and offline demo).

    Unit 6 Section 6 commits to tests that replay recorded API responses so
    they are deterministic and independent of live API access.
    """

    name = "fixture"

    def __init__(self, path: Path):
        self.responses: dict[str, list[dict]] = json.loads(Path(path).read_text("utf-8"))
        self.queries: list[str] = []

    def search(self, query: str, constraints: Constraints) -> list[Paper]:
        self.queries.append(query)
        records = self.responses.get(query.lower(), [])
        papers = [p for p in (SemanticScholarSearch.parse(r) for r in records) if p]
        return papers[: constraints.max_papers_per_subquestion]


def _normalise_title(title: str) -> str:
    title = re.sub(r"<[^>]+>", " ", title)  # Crossref titles may contain JATS/MathML tags
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split())


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalise_title(a), _normalise_title(b)).ratio()


class CrossrefValidator:
    """Checks that a paper's DOI exists and that Crossref's metadata matches it."""

    URL = "https://api.crossref.org/works/{doi}"

    def __init__(self, mailto: str = "", title_threshold: float = 0.85,
                 timeout: float = 20.0, session: requests.Session | None = None):
        self.title_threshold, self.timeout = title_threshold, timeout
        self.session = session or requests.Session()
        # Crossref asks clients to identify themselves; doing so routes
        # requests to the more reliable "polite" pool.
        self.headers = {"User-Agent": f"research-agent/1.0 (mailto:{mailto})"} if mailto else {}

    def fetch(self, doi: str) -> dict:
        data = _get_json(self.session, self.URL.format(doi=doi), {}, self.headers, self.timeout)
        return data.get("message", {})

    def validate(self, paper: Paper) -> Paper:
        if not paper.doi:
            return paper.model_copy(update={"doi_verified": False, "metadata_flag": "no_doi"})
        try:
            meta = self.fetch(paper.doi)
        except SourceUnavailable as exc:
            # Unavailable is not the same as invalid: flag, do not discard.
            return paper.model_copy(update={"doi_verified": False,
                                            "metadata_flag": f"crossref_unavailable: {exc}"})
        return self.compare(paper, meta, self.title_threshold)

    @staticmethod
    def compare(paper: Paper, meta: dict, threshold: float = 0.85) -> Paper:
        if not meta:
            return paper.model_copy(update={"doi_verified": False, "metadata_flag": "doi_not_found"})
        cr_title = (meta.get("title") or [""])[0]
        sim = title_similarity(paper.title, cr_title)
        cr_year = None
        for key in ("published-print", "published-online", "issued", "published"):
            parts = (meta.get(key) or {}).get("date-parts") or [[None]]
            if parts[0] and parts[0][0]:
                cr_year = parts[0][0]
                break
        problems = []
        if sim < threshold:
            problems.append(f"title_mismatch({sim:.2f})")
        # Online-first publication routinely differs from the issue year by one.
        if paper.year and cr_year and abs(paper.year - cr_year) > 1:
            problems.append(f"year_mismatch({paper.year}!={cr_year})")
        return paper.model_copy(update={"doi_verified": not problems,
                                        "metadata_flag": ",".join(problems) or None})


class FixtureCrossrefValidator(CrossrefValidator):
    """Crossref validator that reads recorded `works/{doi}` messages from disk."""

    def __init__(self, path: Path, title_threshold: float = 0.85):
        super().__init__(title_threshold=title_threshold)
        self.records: dict[str, dict] = json.loads(Path(path).read_text("utf-8"))

    def fetch(self, doi: str) -> dict:
        return self.records.get(doi, {})
