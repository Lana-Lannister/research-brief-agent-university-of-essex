"""Run-time configuration, loaded once from environment variables / .env.

Secrets never appear in code or logs: python-dotenv reads them from a local
.env file that is git-ignored (Unit 6, Section 3). Every numeric limit that
shapes agent autonomy lives here, so the bounds on autonomy are visible and
auditable in one place rather than scattered as magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional at import time so tests run without python-dotenv installed
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class Settings:
    # --- LLM -------------------------------------------------------------
    llm_provider: str = "gemini"            # "gemini" or "scripted" (offline)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2            # low: we want repeatable plans, not creativity
    llm_max_retries: int = 2                # re-prompts when output fails validation

    # --- Literature sources --------------------------------------------
    semantic_scholar_api_key: str = ""      # optional; raises rate limit when set
    crossref_mailto: str = ""               # Crossref "polite pool" etiquette
    http_timeout_s: float = 20.0

    # --- Bounded autonomy (Unit 6, Diagram 3) ---------------------------
    record_threshold: int = 3               # usable records needed per sub-question
    max_auto_reformulations: int = 1        # automatic replanning per sub-question
    max_scope_revisions: int = 1            # researcher-triggered scope rejections
    min_subquestions: int = 3
    max_subquestions: int = 5
    relevance_cutoff: float = 0.6           # LLM score needed to pre-select a paper
    title_match_threshold: float = 0.85     # Crossref title similarity to verify

    output_dir: Path = field(default_factory=lambda: Path("output"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))


def load_settings(env_file: str | None = ".env") -> Settings:
    if load_dotenv and env_file and Path(env_file).exists():
        load_dotenv(env_file)
    env = os.environ
    return Settings(
        llm_provider=env.get("LLM_PROVIDER", "gemini"),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        gemini_model=env.get("GEMINI_MODEL", "gemini-2.5-flash"),
        semantic_scholar_api_key=env.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        crossref_mailto=env.get("CROSSREF_MAILTO", ""),
        record_threshold=int(env.get("RECORD_THRESHOLD", 3)),
        relevance_cutoff=float(env.get("RELEVANCE_CUTOFF", 0.6)),
        output_dir=Path(env.get("OUTPUT_DIR", "output")),
        log_dir=Path(env.get("LOG_DIR", "logs")),
    )
