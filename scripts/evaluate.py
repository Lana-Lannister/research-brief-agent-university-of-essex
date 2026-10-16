"""Evaluate a run against the success measures promised in Unit 6, Section 6.

Measures: retrieval relevance (precision/recall of the approved selection
against hand-labelled relevance judgements), citation accuracy (share of
selected papers whose DOI Crossref verified), goal preservation (share of
sub-questions that still cover the research question's content words) and
storage success (both files written and the JSON re-loads).

Gold labels exist only for the offline demo fixtures, so this script measures
the *pipeline*; evaluating a live run needs labels written for that question.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_agent.heuristic import keywords  # noqa: E402
from research_agent.models import Brief  # noqa: E402

# Hand-labelled relevance for the synthetic demo corpus (paper -> sub-question facet).
GOLD = {"S2:demo001": "methods", "S2:demo003": "methods", "S2:demo004": "methods",
        "S2:demo002": "evaluation", "S2:demo005": "evaluation", "S2:demo010": "evaluation",
        "S2:demo006": "limitations", "S2:demo007": "limitations", "S2:demo008": "limitations"}


def evaluate(json_path: Path) -> dict:
    brief = Brief.model_validate_json(json_path.read_text("utf-8"))
    selected = {p.paper_id for p in brief.selected_papers}
    relevant = set(GOLD)
    tp = len(selected & relevant)
    q_terms = set(keywords(brief.research_question))
    preserved = sum(1 for s in brief.sub_questions
                    if len(q_terms & set(keywords(s.text))) / max(len(q_terms), 1) >= 0.5)
    return {
        "selected": len(selected),
        "precision": round(tp / len(selected), 2) if selected else 0.0,
        "recall": round(tp / len(relevant), 2),
        "citation_accuracy": round(sum(p.doi_verified for p in brief.selected_papers)
                                   / max(len(selected), 1), 2),
        "goal_preservation": round(preserved / len(brief.sub_questions), 2),
        "storage_ok": json_path.with_suffix(".md").exists(),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1])), indent=2))
