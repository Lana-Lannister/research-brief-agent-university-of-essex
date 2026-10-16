"""Prompt templates.

Kept separate from agent logic so prompt changes are reviewable diffs on their
own. Every prompt (1) fixes an output schema, (2) forbids inventing references,
and (3) passes inputs as a JSON block, so the model's evidence is exactly
what the orchestrator chose to send it. Inputs are placed after the
instructions and kept short (abstract-level), because long contexts degrade
how reliably models use information placed in the middle (Liu et al., 2024).
"""

from __future__ import annotations

import json

INPUT_MARKER = "INPUT_JSON:"


def _with_input(instructions: str, payload: dict) -> str:
    return f"{instructions}\n\n{INPUT_MARKER}\n{json.dumps(payload, ensure_ascii=False)}"


PLANNER_SYSTEM = (
    "You are the Planning Agent of an academic literature-search system. You "
    "decompose a research question into focused sub-questions and keyword "
    "search queries for the Semantic Scholar search API. You never cite or "
    "invent papers. Reply with JSON only."
)

EVALUATOR_SYSTEM = (
    "You are the Evaluation and Synthesis Agent of an academic literature-search "
    "system. You judge relevance strictly from the titles and abstracts provided "
    "and summarise only what those abstracts state. Never add facts, papers or "
    "citations that are not in the input. Reply with JSON only."
)


def plan_prompt(question: str, constraints: dict, n_min: int, n_max: int,
                feedback: str | None = None) -> str:
    instr = (
        f"Decompose the research question into {n_min}-{n_max} non-overlapping "
        "sub-questions that together cover it. For each, write a search_query "
        "of 3-8 keywords (no quotes, no boolean operators) suited to a "
        "relevance-ranked academic search engine.\n"
        'Schema: {"sub_questions": [{"id": int, "text": str, "search_query": str}]}'
    )
    if feedback:
        instr += ("\nThe researcher reviewed a previous plan and asked for this "
                  f"revision; follow it precisely: {feedback}")
    return _with_input(instr, {"research_question": question, "constraints": constraints})


def reformulate_prompt(sub_question: str, failed_query: str, hits: int) -> str:
    instr = (
        f"The query below returned only {hits} usable records. Write ONE "
        "alternative query that broadens recall while staying on the "
        "sub-question: use synonyms or broader terms and fewer keywords.\n"
        'Schema: {"search_query": str}'
    )
    return _with_input(instr, {"sub_question": sub_question, "failed_query": failed_query})


def evaluate_prompt(sub_questions: list[dict], papers: list[dict]) -> str:
    instr = (
        "For every (paper, sub-question) pair where the paper is plausibly "
        "relevant, give relevance_score in [0,1] and a one-sentence rationale "
        "grounded in the abstract. Use 0.8-1.0 only when the abstract directly "
        "addresses the sub-question. Then list 2-5 cross-cutting themes and "
        "1-4 gaps (aspects of the sub-questions the abstracts do not cover).\n"
        'Schema: {"assessments": [{"paper_id": str, "subquestion_id": int, '
        '"relevance_score": float, "rationale": str}], "themes": [str], "gaps": [str]}'
    )
    return _with_input(instr, {"sub_questions": sub_questions, "papers": papers})


def synthesise_prompt(question: str, papers: list[dict]) -> str:
    instr = (
        "Write a 2-3 sentence summary of each paper using ONLY its abstract, "
        "addressed to a researcher investigating the question. Do not mention "
        "papers that are not in the input.\n"
        'Schema: {"summaries": [{"paper_id": str, "summary": str}]}'
    )
    return _with_input(instr, {"research_question": question, "papers": papers})
