"""Download a curated, realistic public corpus for RAG demos.

We use Python Enhancement Proposals (PEPs). They are:
  - real technical docs with dense concepts,
  - stable and public-domain,
  - plain text, no flaky PDF parsing.

A LinkedIn-friendly demo benefits from a corpus anyone can reproduce: a hiring
manager who clones the repo and runs `python scripts/download_corpus.py` gets
the exact same corpus the benchmark numbers were measured on.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

# Curated mix: language semantics, typing, async, packaging, governance, style.
# ~30 PEPs across different concept clusters so hybrid retrieval actually has
# to distinguish between lexically-similar-but-semantically-different topics.
PEPS = [
    8,     # Style Guide for Python Code
    20,    # The Zen of Python
    257,   # Docstring Conventions
    343,   # The "with" Statement
    492,   # Coroutines with async/await
    525,   # Asynchronous Generators
    530,   # Asynchronous Comprehensions
    572,   # Assignment Expressions (walrus)
    585,   # Type Hinting Generics In Standard Collections
    604,   # Allow writing union types as X | Y
    612,   # Parameter Specification Variables
    634,   # Structural Pattern Matching
    646,   # Variadic Generics
    647,   # User-Defined Type Guards
    657,   # Include Fine Grained Error Locations in Tracebacks
    669,   # Low Impact Monitoring for CPython
    673,   # Self Type
    675,   # Arbitrary Literal String Type
    681,   # Data Class Transforms
    695,   # Type Parameter Syntax
    701,   # Syntactic formalization of f-strings
    703,   # Making the Global Interpreter Lock Optional in CPython
    709,   # Inlined comprehensions
    711,   # PyBI: a standard format for distributing Python Binaries
    723,   # Inline script metadata
    3107,  # Function Annotations
    3119,  # Introducing Abstract Base Classes
    3131,  # Supporting Non-ASCII Identifiers
    3156,  # Asynchronous IO Support Rebooted
    484,   # Type Hints
]


def _try_fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-app-demo/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception:
        return None


def download(pep: int, out_dir: Path) -> Path:
    # peps.python.org no longer serves .txt. Fetch raw RST from the canonical repo.
    # Try the current layout (peps/pep-XXXX.rst) then the legacy flat layout.
    candidates = [
        f"https://raw.githubusercontent.com/python/peps/main/peps/pep-{pep:04d}.rst",
        f"https://raw.githubusercontent.com/python/peps/main/pep-{pep:04d}.rst",
        f"https://raw.githubusercontent.com/python/peps/main/pep-{pep:04d}.txt",
    ]
    data = None
    for url in candidates:
        data = _try_fetch(url)
        if data:
            break
    if data is None:
        raise RuntimeError(f"no working URL for PEP {pep}")
    dest = out_dir / f"pep-{pep:04d}.rst"
    dest.write_bytes(data)
    return dest


def main() -> int:
    out_dir = Path("data/docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(PEPS)} PEPs into {out_dir}/ ...")
    ok, fail = 0, 0
    for pep in PEPS:
        try:
            dest = download(pep, out_dir)
            print(f"  PEP {pep:>4} -> {dest.name} ({dest.stat().st_size // 1024} KB)")
            ok += 1
            # Be polite to peps.python.org.
            time.sleep(0.15)
        except Exception as e:
            print(f"  PEP {pep:>4} FAILED: {e}", file=sys.stderr)
            fail += 1
    print(f"\nDone. {ok} downloaded, {fail} failed.")
    print("Next: python cli.py ingest data/docs")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
