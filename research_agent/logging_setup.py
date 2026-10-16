"""Run logging: one key=value line per decision, to console and a per-run file.

key=value lines are human-readable during a demo and trivially grep-able
afterwards, which is what post-run inspection of queries, scores and
decisions (Unit 6, Section 3) needs; a heavier JSON logging stack would add
dependencies without changing what can be audited.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, run_id: str, verbose: bool = True) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run_{run_id}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-34s %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    # Third-party HTTP debug noise would bury the agent decisions.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return path
