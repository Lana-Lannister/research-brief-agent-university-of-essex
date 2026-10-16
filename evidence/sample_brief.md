# Research brief

**Question:** How effective are LLM-based agents at automating systematic literature reviews?  
**Run id:** `d57313e7`

## Sub-questions

1. What approaches and architectures are used for llm-based agents automating systematic literature reviews?  
   _query:_ `llm-based agents automating systematic literature reviews methods`
2. What evidence exists on the performance of llm-based agents automating systematic literature reviews?  
   _query:_ `llm-based agents automating systematic literature reviews`
3. What risks and limitations affect llm-based agents automating systematic literature reviews?  
   _query:_ `llm-based agents automating systematic literature reviews limitations`

## Selected papers

### 1. Hallucinated references and citation errors in LLM-generated literature reviews (2025)
Synthetic Author 6A, Synthetic Author 6B  
https://doi.org/10.5555/demo.006 — _Crossref-verified_  
Sub-questions: 3

Large language models can fabricate references or corrupt citation details. We analyse limitations and risks of LLM-based agents automating systematic literature reviews, finding that ungrounded generation produced non-existent DOIs and mismatched metadata, while agents retrieving from scholarly indexes and verifying DOIs reduced these errors substantially.

### 2. Multi-agent architectures for automating systematic literature reviews (2025)
Synthetic Author 1A, Synthetic Author 1B  
https://doi.org/10.5555/demo.001 — _Crossref-verified_  
Sub-questions: 1, 2

We describe a multi-agent architecture in which LLM-based agents plan search strategies, screen titles and abstracts, and draft evidence tables for systematic literature reviews. A deterministic controller coordinates the agents and enforces review protocol rules.

### 3. Evaluating LLM-based agents for title and abstract screening in systematic reviews (2025)
Synthetic Author 2A, Synthetic Author 2B  
https://doi.org/10.5555/demo.002 — _Crossref-verified_  
Sub-questions: 1, 2

This evaluation measures how accurately LLM-based agents screen titles and abstracts for systematic literature reviews. Across four review datasets the agents reached high recall but variable precision, and performance depended strongly on prompt design.

### 4. Human oversight in semi-automated systematic reviews (2025)
Synthetic Author 7A, Synthetic Author 7B  
https://doi.org/10.5555/demo.007 — _Crossref-verified_  
Sub-questions: 3

Semi-automated tools for systematic literature reviews keep humans responsible for final inclusion decisions. This study examines limitations of automating review stages with LLM-based agents, including automation bias and reduced reviewer attention, and proposes checkpoints at which human reviewers approve agent outputs before the workflow continues.

### 5. Reproducibility limitations of language-model screening in evidence reviews (2025)
Synthetic Author 8A, Synthetic Author 8B  
https://doi.org/10.5555/demo.008 — _not verified: title_mismatch(0.16)_  
Sub-questions: 3

Language-model outputs vary between runs and model versions. We test the reproducibility of LLM-based agents automating systematic literature reviews and find that screening decisions changed for a notable share of records across repeated runs, a limitation that requires logging prompts, model versions and decisions for audit.

### 6. Retrieval-augmented generation pipelines for literature review automation (2025)
Synthetic Author 3A, Synthetic Author 3B  
https://doi.org/10.5555/demo.003 — _Crossref-verified_  
Sub-questions: 1

Retrieval-augmented generation grounds language model outputs in retrieved documents. We present methods for building retrieval pipelines that support automating systematic literature reviews, including query expansion, abstract retrieval and citation linking, and discuss architectures that keep generated summaries traceable to sources.

### 7. Planning and tool use in LLM-based research agents (2025)
Synthetic Author 4A, Synthetic Author 4B  
https://doi.org/10.5555/demo.004 — _not verified: doi_not_found_  
Sub-questions: 1

LLM-based agents increasingly plan multi-step research tasks and call external tools such as scholarly search engines. This paper surveys planning methods, including task decomposition and reflection, and analyses how agents for automating literature reviews select tools, recover from failed searches and record their decisions for audit.

### 8. Comparative evaluation of agentic and single-prompt approaches to literature reviews (2025)
Synthetic Author 10A, Synthetic Author 10B  
no DOI — _not verified: no_doi_  
Sub-questions: 2

We compare agentic workflows with single-prompt baselines for automating systematic literature reviews. In an evaluation on twelve review topics, agentic workflows improved recall of relevant studies but increased cost and latency, and gains depended on the task and the coordination architecture.

### 9. Benchmarking automated evidence synthesis with large language models (2025)
Synthetic Author 5A, Synthetic Author 5B  
https://doi.org/10.5555/demo.005 — _Crossref-verified_  
Sub-questions: 2

We benchmark large language models on evidence synthesis tasks drawn from published systematic literature reviews. Evaluation covers data extraction, summary faithfulness and agreement with human reviewers.

## Themes

- reviews
- literature
- systematic
- multi-agent

## Gaps

- Lexical baseline cannot infer gaps; use an LLM provider.

## Limitations of this search

- Relevance and summaries are based on titles and abstracts only; methods, results and limitations in full texts were not read.
- Search source: fixture; papers outside its index are missed.
- Relevance scores are LLM judgements pre-selecting evidence for human approval, not a quality appraisal.
- 3 selected paper(s) could not be verified against Crossref; check their citation details manually.
- Run in auto-approve mode: no human reviewed the sub-questions or evidence.
