# Research Brief Agent: an LLM-powered planning agent for academic literature search

Individual implementation of the Group D design (Unit 6), for the *Intelligent Agents* Development Project (Unit 11).
Author: Lana Menezes Fidelis

## 1. Purpose

A manual literature search is slow and error-prone (van de Schoot et al., 2021). This application receives a high-level research question as its goal and produces a **research brief**, saved as Markdown and JSON. To do this it plans sub-questions, retrieves scholarly records, evaluates them, and synthesises the evidence. The brief lists the selected papers with Crossref-checked DOIs, abstract-grounded summaries, themes, gaps and the search's own limitations. The intended users are students and researchers who are starting a search. The output is a starting point and is not presented as a finished review.

## 2. Solution overview

```
 Researcher ──question──▶ ORCHESTRATOR (deterministic, not an agent)
                              │  owns: retry/scope limits, dedup, DOI validation, storage, logging
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
  Planning Agent        Retrieval Agent          Evaluation & Synthesis Agent
  (LLM) decompose,      Semantic Scholar         (LLM) score relevance, themes,
  revise, reformulate   → OpenAlex fallback;     gaps; summarise approved
                        judges own threshold     abstracts only
        ▲                     │ "threshold unmet" → one reformulation via orchestrator
        └─────────────────────┘
  Human Review 1: approve/revise sub-questions    Human Review 2: approve / refine / reject scope (max 1)
                         Crossref API validates DOIs ─▶ brief_<run>.md + brief_<run>.json + logs/run_<run>.log
```

**Autonomy.** The system works toward its goal on its own. Given a goal, it plans by decomposition, acts on external tools and checks its own results. When retrieval is thin it requests a re-plan (it is reactive), and it pursues the goal across all sub-questions (it is pro-active). These are the properties Wooldridge and Jennings (1995) use to define agency. The autonomy has bounds: two human checkpoints, one automatic reformulation per sub-question and one scope revision per run (Section 4).

### Design-to-code traceability

| Unit 6 design element | Implementation |
|---|---|
| Planning / Retrieval / Evaluation & Synthesis agents (Diagram 1) | `research_agent/agents/planning.py`, `retrieval.py`, `evaluation.py` |
| Deterministic orchestrator, `retryCounts`, `scopeRevisionCount` | `research_agent/orchestrator.py` |
| Interaction protocol, steps 1-46 (Diagram 2) | `Orchestrator._run`, `_plan_with_review`, `_retrieve_all` |
| Decision logic: retry < 1, scope revision < 1, flag insufficient coverage (Diagram 3) | `Orchestrator._retrieve_all`, `_run` (REJECT_SCOPE branch) |
| `ResearcherInterface` boundary, `Decision` enum | `research_agent/review.py`, `models.py` |
| Typed messages, validated state | Pydantic models in `models.py` |
| Semantic Scholar primary, OpenAlex fallback, Crossref validation | `research_agent/sources.py` |
| Gemini Flash free tier | `GeminiClient` in `llm.py` |
| Markdown + JSON storage, logging | `storage.py`, `logging_setup.py` |
| pytest with replayed API responses | `tests/`, `tests/fixtures/`, `demo/` |

### Key design decisions (full rationale in code comments)

1. **The LLM only performs semantic tasks.** Thresholds, DOI checks, deduplication and storage are exact rules, so they are kept in code. LLMs generate fluent but unsupported content (Klesel and Wittmann, 2025), and they fabricate citations when asked for them (Walters and Wilder, 2023). For that reason the model is never asked for references; it only judges records that the indexes returned.
2. **A deterministic orchestrator coordinates the agents, not a manager LLM.** Gains from multi-agent systems depend on the task, and errors spread further without a central verification point (Kim et al., 2026). Roles and protocol follow Gaia (Wooldridge, Jennings and Kinny, 2000).
3. **Validate, re-prompt, then fail closed.** Every LLM reply is parsed against a schema. Errors are fed back for a bounded number of retries (a limited form of the self-correction in Shinn et al., 2023). If the reply is still invalid, the run stops without writing a brief.
4. **Grounding is enforced in code.** Assessments and summaries that cite ids the model was not given are discarded. Summaries use only approved abstracts (Maynez et al., 2020). Limitations are generated from run facts, so the model cannot invent them.
5. **The human makes the evaluative decisions.** LLM scores pre-select evidence and the researcher approves it, because LLM judges are biased (Zheng et al., 2023). The checkpoints implement medium-level automation (Parasuraman, Sheridan and Wickens, 2000) and keep evaluative agency with the human (Zhu et al., 2026; Bolaños et al., 2024).
6. **Prompts stay short.** Abstracts are truncated and papers are evaluated in batches, because models use information in the middle of long contexts poorly (Liu et al., 2024).

## 3. Installation

Requires Python 3.11 or later.

```bash
git clone <your-repo-url> research-brief-agent && cd research-brief-agent
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then edit .env
```

Configuration in `.env`. Only `GEMINI_API_KEY` is needed for live runs:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Free key from Google AI Studio |
| `GEMINI_MODEL` | Default `gemini-2.5-flash`; any Gemini model your key can call |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional; raises the shared rate limit |
| `CROSSREF_MAILTO` | Your email, which routes requests to Crossref's polite pool |
| `RECORD_THRESHOLD`, `RELEVANCE_CUTOFF` | Autonomy and selection bounds (defaults 3 and 0.6) |

## 4. Running

**Live run with both human checkpoints:**
```bash
python -m research_agent "How effective are LLM-based agents at automating systematic literature reviews?" --year-from 2020
```
At checkpoint 1, type `a` to approve or `r` to request a revision. At checkpoint 2, type `a` to approve, `f` to deselect papers by id, or `x` to reject the scope (allowed once).

**Offline demo (no key, no network).** This uses the synthetic recorded corpus in `demo/` and a transparent keyword baseline in place of the LLM:
```bash
python -m research_agent "How effective are LLM-based agents at automating systematic literature reviews?" --offline --relevance-cutoff 0.5
```

Other options: `--auto-approve` skips both checkpoints for unattended runs (this is stated in the brief's limitations), `--max-papers N`, `--year-to`, `--llm heuristic`, and `--quiet`.

**Outputs:** `output/brief_<run>.md`, `output/brief_<run>.json` and `logs/run_<run>.log`, which has one `key=value` line per decision: queries, record counts, reformulations, DOI flags, review decisions. Exit codes: 0 means completed, 1 means stopped, failed or no evidence, and 2 means a configuration error.

## 5. Testing

```bash
python -m pytest --cov=research_agent      # or: python -m unittest discover -s tests -t . -v
python scripts/evaluate.py output/brief_<run>.json
```
The suite has 55 unit, functional and robustness tests, with 95% line coverage, and runs offline. The test strategy, evaluation results, the remediation log (five defects found and fixed) and the plan for live end-to-end testing are in **TESTING.md**. Captured outputs are in `evidence/`.

## 6. Project structure

```
research_agent/   models.py  config.py  llm.py  prompts.py  sources.py  review.py
                  orchestrator.py  storage.py  heuristic.py  logging_setup.py  cli.py
                  agents/ planning.py  retrieval.py  evaluation.py
tests/            unit, functional and robustness tests; fixtures/ (API-format responses)
demo/             synthetic search + Crossref corpus for offline runs and tests
scripts/          evaluate.py  record_fixtures.py  stdlib_coverage.py
evidence/         test output, coverage, run logs, sample brief, interactive session
```

## 7. Limitations and legal, social, ethical and professional considerations

- **Scope.** The system reads abstracts only, uses one primary source and assumes questions in English. Summaries can omit methods or caveats from the full text, and every brief says so.
- **Automation bias.** Researchers may approve pre-selected evidence too readily. Every paper is shown with its score, rationale and verification status, and auto-approve runs are labelled in the brief.
- **Legal and terms of use.** The system uses only official APIs within their published terms, with back-off on rate limits. It stores no personal data beyond author names from public metadata. Keys stay in an untracked `.env` file.
- **Reproducibility.** LLM outputs vary between runs and model versions, so each run logs the prompts' inputs and every decision for audit.
- **Offline baseline.** The keyword baseline cannot use revision feedback or detect synonyms. It exists to make the pipeline testable and to give the LLM a baseline to beat.

## 8. Acknowledgements and sources

Libraries: requests (Apache-2.0), Pydantic (MIT), python-dotenv (BSD-3), pytest and pytest-cov (MIT). Services: Google Gemini API; Semantic Scholar Academic Graph API (Allen Institute for AI; Kinney et al., 2023); OpenAlex (Priem, Piwowar and Orr, 2022); Crossref REST API (Hendricks et al., 2020). Design patterns: Strategy and composition root (Gamma et al., 1994).

**Use of AI.** *[Complete this statement in line with the UoEO guidance in the Student Information area, declaring any generative-AI assistance used in producing this submission.]*

## References

Bolaños, F., Salatino, A., Osborne, F. and Motta, E. (2024) 'Artificial intelligence for literature reviews: opportunities and challenges', *Artificial Intelligence Review*, 57, article 259. Available at: https://doi.org/10.1007/s10462-024-10902-3

Gamma, E., Helm, R., Johnson, R. and Vlissides, J. (1994) *Design patterns: elements of reusable object-oriented software*. Reading, MA: Addison-Wesley.

Hendricks, G., Tkaczyk, D., Lin, J. and Feeney, P. (2020) 'Crossref: the sustainable source of community-owned scholarly metadata', *Quantitative Science Studies*, 1(1), pp. 414-427. Available at: https://doi.org/10.1162/qss_a_00022

Kim, Y. et al. (2026) 'Capable language models can outgrow the benefits of collaboration', *Nature Machine Intelligence*, 8, pp. 1157-1172. Available at: https://doi.org/10.1038/s42256-026-01268-y

Kinney, R. et al. (2023) 'The Semantic Scholar Open Data Platform', *arXiv*, 2301.10140. Available at: https://doi.org/10.48550/arXiv.2301.10140

Klesel, M. and Wittmann, H.F. (2025) 'Retrieval-augmented generation (RAG)', *Business & Information Systems Engineering*, 67, pp. 551-561. Available at: https://doi.org/10.1007/s12599-025-00945-3

Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F. and Liang, P. (2024) 'Lost in the middle: how language models use long contexts', *Transactions of the Association for Computational Linguistics*, 12, pp. 157-173. Available at: https://doi.org/10.1162/tacl_a_00638

Maynez, J., Narayan, S., Bohnet, B. and McDonald, R. (2020) 'On faithfulness and factuality in abstractive summarization', *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. Online: ACL, pp. 1906-1919. Available at: https://doi.org/10.18653/v1/2020.acl-main.173

Parasuraman, R., Sheridan, T.B. and Wickens, C.D. (2000) 'A model for types and levels of human interaction with automation', *IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans*, 30(3), pp. 286-297. Available at: https://doi.org/10.1109/3468.844354

Priem, J., Piwowar, H. and Orr, R. (2022) 'OpenAlex: a fully-open index of scholarly works, authors, venues, institutions, and concepts', *arXiv*, 2205.01833. Available at: https://doi.org/10.48550/arXiv.2205.01833

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. and Yao, S. (2023) 'Reflexion: language agents with verbal reinforcement learning', *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. New Orleans, USA. Available at: https://doi.org/10.48550/arXiv.2303.11366

van de Schoot, R. et al. (2021) 'An open source machine learning framework for efficient and transparent systematic reviews', *Nature Machine Intelligence*, 3, pp. 125-133. Available at: https://doi.org/10.1038/s42256-020-00287-7

Walters, W.H. and Wilder, E.I. (2023) 'Fabrication and errors in the bibliographic citations generated by ChatGPT', *Scientific Reports*, 13, article 14045. Available at: https://doi.org/10.1038/s41598-023-41032-5

Wang, L. et al. (2024) 'A survey on large language model based autonomous agents', *Frontiers of Computer Science*, 18, article 186345. Available at: https://doi.org/10.1007/s11704-024-40231-1

Wooldridge, M. and Jennings, N.R. (1995) 'Intelligent agents: theory and practice', *The Knowledge Engineering Review*, 10(2), pp. 115-152. Available at: https://doi.org/10.1017/S0269888900008122

Wooldridge, M., Jennings, N.R. and Kinny, D. (2000) 'The Gaia methodology for agent-oriented analysis and design', *Autonomous Agents and Multi-Agent Systems*, 3, pp. 285-312. Available at: https://doi.org/10.1023/A:1010071910869

Zheng, L. et al. (2023) 'Judging LLM-as-a-judge with MT-Bench and Chatbot Arena', *Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track*. New Orleans, USA. Available at: https://doi.org/10.48550/arXiv.2306.05685

Zhu, L. et al. (2026) 'Designing meaningful human oversight in AI', *AI and Ethics*, 6, article 286. Available at: https://doi.org/10.1007/s43681-026-01147-7
