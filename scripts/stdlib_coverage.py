"""Line coverage using only the standard library (fallback when pytest-cov is unavailable).

Preferred: `pytest --cov=research_agent`. This script exists because the build
environment used during development had no package-index access; it gives an
equivalent per-module line-coverage table from `trace`.
"""
import sys
import trace
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PKG = ROOT / "research_agent"


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    return unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w")).run(suite)


tracer = trace.Trace(count=True, trace=False)
result = tracer.runfunc(run_tests)
counts = tracer.results().counts
total_exec = total_hit = 0
print(f"{'module':40s} {'lines':>6s} {'hit':>6s} {'cover':>6s}")
for path in sorted(PKG.rglob("*.py")):
    executable = {n for n in trace._find_executable_linenos(str(path)) if isinstance(n, int)}
    # Ignore lines explicitly excluded (same convention as coverage.py).
    src = path.read_text().splitlines()
    executable = {n for n in executable if "pragma: no cover" not in src[n - 1]}
    hit = {ln for (f, ln) in counts if Path(f).resolve() == path.resolve()} & executable
    total_exec += len(executable)
    total_hit += len(hit)
    pct = 100 * len(hit) / len(executable) if executable else 100
    print(f"{str(path.relative_to(ROOT)):40s} {len(executable):6d} {len(hit):6d} {pct:5.0f}%")
print(f"{'TOTAL':40s} {total_exec:6d} {total_hit:6d} {100*total_hit/total_exec:5.0f}%")
print(f"tests run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}")
