"""Command-line entry point: wires concrete components into the orchestrator.

All construction happens here (composition root), so agents and the
orchestrator never import a concrete API client. Swapping live services for
recorded fixtures is therefore a flag, not a code change.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from .agents import EvaluationSynthesisAgent, PlanningAgent, RetrievalAgent
from .config import load_settings
from .heuristic import heuristic_llm
from .llm import CallbackLLM, GeminiClient
from .logging_setup import setup_logging
from .models import Constraints
from .orchestrator import Orchestrator
from .review import AutoApproveReviewer, ConsoleReviewer
from .sources import (CrossrefValidator, FallbackSearch, FixtureCrossrefValidator,
                      FixtureSearch, OpenAlexSearch, SemanticScholarSearch)

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research_agent",
                                description="Agent-based academic literature search assistant.")
    p.add_argument("question", help="High-level research question (English).")
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)
    p.add_argument("--max-papers", type=int, default=8, help="Records per sub-question.")
    p.add_argument("--auto-approve", action="store_true",
                   help="Skip both human checkpoints (unattended demo; recorded in the brief).")
    p.add_argument("--offline", action="store_true",
                   help="Use recorded fixtures and the lexical baseline: no network or API key.")
    p.add_argument("--llm", choices=["gemini", "heuristic"], help="Override LLM_PROVIDER.")
    p.add_argument("--relevance-cutoff", type=float)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = load_settings()
    if args.relevance_cutoff is not None:
        s = replace(s, relevance_cutoff=args.relevance_cutoff)
    run_id = uuid.uuid4().hex[:8]
    log_path = setup_logging(s.log_dir, run_id, verbose=not args.quiet)

    provider = args.llm or ("heuristic" if args.offline else s.llm_provider)
    if provider == "heuristic":
        llm = CallbackLLM(heuristic_llm)
    else:
        try:
            llm = GeminiClient(s.gemini_api_key, s.gemini_model, s.llm_temperature,
                               s.llm_max_retries)
        except ValueError as exc:
            # Remediation R2: a missing key used to crash with a traceback.
            # Configuration errors are the user's to fix, so say how, and
            # exit with the conventional usage-error code.
            print(f"Configuration error: {exc}\nOr run with --offline to try the "
                  "system without a key.", file=sys.stderr)
            return 2

    if args.offline:
        search = FixtureSearch(DEMO_DIR / "search_fixtures.json")
        validator = FixtureCrossrefValidator(DEMO_DIR / "crossref_fixtures.json",
                                             s.title_match_threshold)
    else:
        search = FallbackSearch(SemanticScholarSearch(s.semantic_scholar_api_key, s.http_timeout_s),
                                OpenAlexSearch(s.crossref_mailto, s.http_timeout_s))
        validator = CrossrefValidator(s.crossref_mailto, s.title_match_threshold, s.http_timeout_s)

    orch = Orchestrator(
        s, PlanningAgent(llm, s.min_subquestions, s.max_subquestions),
        RetrievalAgent(search, s.record_threshold),
        EvaluationSynthesisAgent(llm, s.relevance_cutoff), validator,
        AutoApproveReviewer() if args.auto_approve else ConsoleReviewer())

    c = Constraints(year_from=args.year_from, year_to=args.year_to,
                    max_papers_per_subquestion=args.max_papers)
    result = orch.run(args.question, c, run_id=run_id)

    print(f"\nRun {result.run_id}: {result.status}. {result.message}")
    if result.paths:
        print(f"Brief saved: {result.paths[0]}\n             {result.paths[1]}")
    print(f"Log: {log_path}\nStats: {result.stats}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
