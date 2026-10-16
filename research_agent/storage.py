"""Persist the brief as Markdown (human reading) and JSON (machine re-use).

Two formats because they serve different readers: Markdown for the researcher,
JSON so the selection can be re-loaded, audited or diffed between runs.
Writes go to a temp file then an atomic rename, so an interrupted run never
leaves a half-written brief that looks complete.
"""

from __future__ import annotations

import os
from pathlib import Path

from .models import Brief


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def render_markdown(brief: Brief) -> str:
    summaries = {s.paper_id: s.summary for s in brief.summaries}
    lines = [f"# Research brief\n", f"**Question:** {brief.research_question}  ",
             f"**Run id:** `{brief.run_id}`\n", "## Sub-questions\n"]
    lines += [f"{s.id}. {s.text}  \n   _query:_ `{s.search_query}`"
              + ("  \n   ⚠ insufficient coverage" if s.insufficient_coverage else "")
              for s in brief.sub_questions]
    lines.append("\n## Selected papers\n")
    for i, p in enumerate(brief.selected_papers, 1):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        status = "Crossref-verified" if p.doi_verified else f"not verified: {p.metadata_flag}"
        doi = f"https://doi.org/{p.doi}" if p.doi else "no DOI"
        lines.append(f"### {i}. {p.title} ({p.year or 'n.d.'})\n{authors}  \n{doi} — _{status}_  \n"
                     f"Sub-questions: {', '.join(map(str, p.subquestion_ids))}\n\n"
                     f"{summaries.get(p.paper_id, '')}\n")
    lines.append("## Themes\n")
    lines += [f"- {t}" for t in brief.themes] or ["- (none identified)"]
    lines.append("\n## Gaps\n")
    lines += [f"- {g}" for g in brief.gaps] or ["- (none identified)"]
    lines.append("\n## Limitations of this search\n")
    lines += [f"- {l}" for l in brief.limitations]
    return "\n".join(lines) + "\n"


def save_brief(brief: Brief, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"brief_{brief.run_id}.md", out_dir / f"brief_{brief.run_id}.json"
    _atomic_write(md, render_markdown(brief))
    _atomic_write(js, brief.model_dump_json(indent=2))
    return md, js
