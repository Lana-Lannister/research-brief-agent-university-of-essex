# Testing and remediation

All outputs referenced here are in `evidence/`, captured from the code in this repository.

## 1. Strategy

| Level | What it checks | Where |
|---|---|---|
| Unit | Each module in isolation: DOI normalisation, JSON extraction, the validate-and-re-prompt loop, API parsing, retry/back-off, Crossref comparison, each agent's rules | `tests/test_models.py`, `test_llm.py`, `test_sources.py`, `test_agents.py` |
| Functional (end-to-end, offline) | Every branch of Unit 6 Diagrams 2 and 3: approve, revise, refine, one scope rejection, second scope rejection stops the run, no evidence, automatic reformulation cap, fail-closed behaviour, CLI exit codes | `tests/test_functional.py` |
| Robustness | Failure paths added after the first green run (Section 4) | `tests/test_robustness.py` |
| Evaluation | The Unit 6 success measures: retrieval relevance (precision/recall vs. hand labels), citation accuracy, goal preservation, storage | `scripts/evaluate.py` |

Tests replay recorded or hand-built responses (`tests/fixtures/`, `demo/`) and inject a scripted LLM, so they are deterministic and need no keys or network, as planned in Unit 6 (Section 6). Network clients are tested through a fake HTTP session, which lets the suite assert on exactly what would be sent (for example, that the Gemini key travels in a header, never in the URL where it could leak into logs).

The demo corpus in `demo/` is **synthetic** (DOIs use the `10.5555` example prefix). It is built to contain one duplicate record, one record without an abstract, one record without a DOI, one DOI that Crossref does not know, one DOI whose Crossref metadata belongs to a different paper, one sub-question that falls below the record threshold, and two off-topic distractors. Every control in the pipeline therefore has something to catch.

## 2. Results

```
python -m pytest            # or: python -m unittest discover -s tests -t . -v
Ran 55 tests ... OK         # evidence/unit_functional_tests.txt
```

Line coverage (`evidence/coverage.txt`): **95 %** of 984 executable lines. The uncovered lines are the interactive `__main__` shim and some console-printing branches. Coverage was measured with the standard-library `trace` module (`scripts/stdlib_coverage.py`) because the build environment had no package-index access; `pytest --cov=research_agent` gives the equivalent figure.

## 3. Evaluation of the offline pipeline (`evidence/evaluation_cutoff_sweep.txt`)

| Relevance cut-off | Selected | Precision | Recall | Citation accuracy | Goal preservation | Stored |
|---|---|---|---|---|---|---|
| 0.3 | 10 | 0.90 | 1.00 | 0.70 | 1.00 | yes |
| 0.5 | 9 | 1.00 | 1.00 | 0.67 | 1.00 | yes |
| 0.7 | 5 | 1.00 | 0.56 | 0.80 | 1.00 | yes |

These figures show the expected precision/recall trade-off around the selection threshold. Citation accuracy is below 1 by construction, because three faulty records were planted. The figures show that the faults are **detected and flagged**. They do not show that faulty papers are removed, because removal is the researcher's decision at checkpoint 2. These numbers come from the lexical baseline on a small synthetic corpus. They validate the pipeline's mechanics, and they make no claim about the quality of the live LLM.

## 4. Remediation log

| ID | Found by | Defect | Fix | Regression test |
|---|---|---|---|---|
| R1 | New robustness test (`evidence/robustness_before_fix.txt`) | When both Semantic Scholar and OpenAlex were unreachable, `SourceUnavailable` escaped `Orchestrator.run` as a raw traceback, and the run log had no failure entry. | `run()` now catches it, logs `run_failed`, returns status `failed` and writes no brief. | `test_all_sources_down_fails_cleanly` |
| R2 | New robustness test | A missing `GEMINI_API_KEY` crashed the CLI with a `ValueError` traceback. | The CLI prints an actionable message (set the key or use `--offline`) and exits with code 2. | `test_missing_api_key_gives_message_not_traceback` |
| R3 | New robustness test | At checkpoint 2, typed paper ids were accepted verbatim. A typo or a missing `S2:` prefix therefore deselected nothing, without any warning. | Ids are resolved against the papers shown, with or without the prefix, and the reviewer is asked again when an id is unknown. | `test_refine_accepts_ids_without_source_prefix`, `test_refine_reprompts_on_unknown_id` |
| R4 | Evaluation script | The first demo corpus contained only relevant papers, so precision was 1.0 at every cut-off and the measure carried no information. | Two off-topic distractors were added. At cut-off 0.3 the baseline now admits one of them (precision 0.90). | `evidence/evaluation_cutoff_sweep.txt` |
| R5 | Manual interactive session | Closing stdin (Ctrl-D) or pressing Ctrl-C at a review prompt produced an `EOFError` traceback. | This case is treated as a researcher cancellation: it is logged as `run_cancelled`, status `cancelled`, and nothing is saved. | `test_closed_input_at_checkpoint_cancels_cleanly` |

**Known limitation (not a defect):** the offline lexical baseline ignores revision feedback at checkpoint 1 and regenerates the same plan (`evidence/interactive_session.txt`). This is inherent to a keyword baseline. With Gemini, the feedback is inserted into the planner prompt, which `test_feedback_is_included_in_revision_prompt` verifies.

## 5. Live end-to-end testing (to be run with network access)

Run the fixed question set below with a Gemini key. Keep the console output and `logs/run_<id>.log` as execution evidence.

1. `How effective are LLM-based agents at automating systematic literature reviews?`
2. `What are the effects of remote work on software developer productivity?`
3. `How is federated learning used to protect patient privacy in healthcare?`

For each run, record the number of selected papers, the Crossref verification rate, and whether any summary states something absent from its abstract (spot-check three). `scripts/record_fixtures.py` captures live API responses into `tests/fixtures/` so that new regression tests can replay them.
