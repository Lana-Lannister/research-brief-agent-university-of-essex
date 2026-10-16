"""Agent-based academic literature search assistant (Group D design, individual build).

The package mirrors the Unit 6 design: three LLM-backed agents (planning,
retrieval, evaluation/synthesis) sequenced by a deterministic orchestrator that
owns every rule which can be checked exactly (thresholds, DOI validation,
deduplication, storage). See README.md for the rationale and references.
"""

__version__ = "1.0.0"
