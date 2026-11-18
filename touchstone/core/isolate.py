"""Verify one candidate in a fresh interpreter.

`compile_function` screens candidate code and runs it in a restricted
namespace, but it still runs inside the caller's process. `verify_isolated`
runs `handle.verify(candidate)` in a newly spawned Python process (the
`spawn` start method, so nothing is inherited from the parent's memory) and
returns the Verdict. Whatever a candidate does to module state there dies
with the process, so it cannot change the next candidate's score.

The cost is one interpreter start-up plus imports per call (about a second
with numpy/scipy), so use it to re-verify results you intend to report, not
inside a tight search loop.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp

from .verifier import VerificationHandle, Verdict


def _run(handle: VerificationHandle, candidate) -> Verdict:
    return handle.verify(candidate)


def verify_isolated(handle: VerificationHandle, candidate, timeout: float = 300.0) -> Verdict:
    """`handle.verify(candidate)` in a fresh spawned process.

    The handle must be picklable (the shipped handles are). A crash or a
    timeout in the child yields an invalid Verdict rather than an exception.
    """
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        fut = pool.submit(_run, handle, candidate)
        try:
            return fut.result(timeout=timeout)
        except cf.TimeoutError:
            for proc in list(getattr(pool, "_processes", {}).values()):
                proc.kill()
            err = f"isolated verify exceeded {timeout}s"
        except Exception as e:  # child died, or the result could not be unpickled
            err = f"isolated verify failed: {e!r}"[:200]
    return Verdict(score=float("-inf"), valid=False, detail={"error": err},
                   handle=getattr(handle, "name", ""), klass=getattr(handle, "klass", 0))
