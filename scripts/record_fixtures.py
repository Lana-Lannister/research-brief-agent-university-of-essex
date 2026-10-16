"""Record live Semantic Scholar and Crossref responses for replay in tests.

Run once with network access:  python scripts/record_fixtures.py "agent planning literature"
Writes tests/fixtures/live_s2_<slug>.json and live_crossref_<n>.json, so new
tests can replay genuine API output (Unit 6, Section 6) without live calls.
"""
import json
import re
import sys
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
query = " ".join(sys.argv[1:]) or "llm agents systematic literature review"
slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
s2 = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                  params={"query": query, "limit": 5, "fields": "title,abstract,year,authors,externalIds"},
                  timeout=30)
s2.raise_for_status()
(OUT / f"live_s2_{slug}.json").write_text(json.dumps(s2.json(), indent=1))
for i, rec in enumerate(s2.json().get("data", [])):
    doi = (rec.get("externalIds") or {}).get("DOI")
    if doi:
        cr = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30)
        if cr.ok:
            (OUT / f"live_crossref_{slug}_{i}.json").write_text(json.dumps(cr.json(), indent=1))
print(f"Recorded fixtures for {query!r} in {OUT}")
