"""Offline, non-LLM baseline that implements the same JSON contract as the LLM.

Purpose: (1) lets the whole pipeline run with no API key or network, so the
README demo and the functional tests always work; (2) gives an explicit
lexical baseline to compare the LLM against. Keyword overlap is a crude
relevance signal (it misses synonyms and paraphrase), which is precisely the
gap the LLM is meant to close; keeping the baseline visible keeps that claim
testable rather than assumed.
"""

from __future__ import annotations

import json
import re

from .prompts import INPUT_MARKER

_STOP = set("""a an and are as at be by can do does for from has have how in into is it its
of on or that the their this to was what when which while who why with within more most
effective effectiveness using use used based between across about""".split())

_FACETS = [("What approaches and architectures are used for {t}?", "methods"),
           ("What evidence exists on the performance of {t}?", "evaluation"),
           ("What risks and limitations affect {t}?", "limitations")]


def keywords(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    out: list[str] = []
    for w in words:
        if w not in _STOP and w not in out:
            out.append(w)
    return out


def _payload(user: str) -> dict:
    return json.loads(user.split(INPUT_MARKER, 1)[1])


def heuristic_llm(task: str, system: str, user: str) -> str:
    data = _payload(user)
    if task == "plan":
        topic_kw = keywords(data["research_question"])[:6]
        topic = " ".join(topic_kw)
        subs = [{"id": i, "text": text.format(t=topic), "search_query": f"{topic} {facet}"}
                for i, (text, facet) in enumerate(_FACETS, 1)]
        return json.dumps({"sub_questions": subs})
    if task == "reformulate":
        # Broaden by dropping the most specific (last) keyword.
        words = data["failed_query"].split()
        return json.dumps({"search_query": " ".join(words[:-1]) if len(words) > 2 else words[0]})
    if task == "evaluate":
        assessments = []
        for p in data["papers"]:
            doc = set(keywords(f"{p['title']} {p.get('abstract') or ''}"))
            for s in data["sub_questions"]:
                q = set(keywords(s["text"]))
                score = round(len(q & doc) / max(len(q), 1), 2)
                if score > 0:
                    assessments.append({"paper_id": p["paper_id"], "subquestion_id": s["id"],
                                        "relevance_score": score,
                                        "rationale": f"keyword overlap {sorted(q & doc)}"})
        counts: dict[str, int] = {}
        for p in data["papers"]:
            for k in keywords(p["title"]):
                counts[k] = counts.get(k, 0) + 1
        themes = [k for k, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n > 1][:4]
        return json.dumps({"assessments": assessments, "themes": themes,
                           "gaps": ["Lexical baseline cannot infer gaps; use an LLM provider."]})
    if task == "synthesise":
        out = []
        for p in data["papers"]:
            sentences = re.split(r"(?<=[.!?])\s+", p.get("abstract") or "")
            out.append({"paper_id": p["paper_id"], "summary": " ".join(sentences[:2])})
        return json.dumps({"summaries": out})
    raise ValueError(f"unknown task {task}")
