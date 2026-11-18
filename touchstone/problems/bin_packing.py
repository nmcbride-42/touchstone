"""Class-2 disposer #1: online bin-packing heuristic discovery.

FunSearch's canonical proving ground. The candidate is a *priority
heuristic*: given an item and the remaining capacity of each open bin, it
scores each bin; the item goes to the highest-scoring bin that fits (open a
new bin if none). The disposer runs the heuristic over a fixed benchmark of
instances and scores it by how few bins it uses vs. the optimal lower bound.

Ground truth here is exact and free: the number of bins used is not a
proxy for anything — it IS the objective. That's why it's the right Phase-0
target. A candidate that beats first-fit/best-fit is a real, checkable win.

For Phase-0 (deterministic spine proof) the heuristic is *parameterized*
(a small genome the cheap mutation proposer can search). Phase-1 swaps in an
LLM proposer that writes the heuristic as actual code — same disposer.
"""
from __future__ import annotations

import random
from typing import Any

from ..core.verifier import VerificationHandle, Verdict
from ..core.sandbox import compile_priority, time_limit, UnsafeCode, CodeTimeout

CAPACITY = 1.0


def _make_instances(n_instances: int = 20, n_pairs: int = 16, seed: int = 1337):
    """Fixed benchmark of large/complement/filler item streams.

    HONEST STATUS (measured 2026-06-18): this distribution is NOT best-fit-
    beatable. The original premise — that best-fit fragments the ~0.45
    complement gaps by stuffing fillers into them — is WRONG: best-fit's
    tightest-fit rule fills the *smallest* gaps first, so it preserves big
    gaps for their complements and is already near-optimal here. A hand-built
    complement-aware heuristic scored identically (margin 0.0). Kept as a
    harder-than-uniform benchmark, but reproducing a real best-fit BEAT needs
    the established OR-Library / Weibull instances (where best-fit is known
    suboptimal) AND a far larger eval budget than a few batched LLM calls —
    FunSearch used thousands. Same seed -> same instances -> replayable."""
    rng = random.Random(seed)
    insts = []
    for _ in range(n_instances):
        stream = []
        for _ in range(n_pairs):
            large = rng.gauss(0.55, 0.015)
            comp = rng.gauss(0.45, 0.015)
            stream.append(max(0.30, min(0.69, large)))
            for _ in range(rng.choice([2, 3])):                 # fillers in the gap window
                stream.append(max(0.04, min(0.12, rng.gauss(0.08, 0.02))))
            stream.append(max(0.30, min(0.60, comp)))           # complement arrives late
        insts.append(stream)
    return insts


_INSTANCES = _make_instances()


def _priority(genome: dict, item: float, remaining: float) -> float:
    """Parameterized heuristic family. Genome interpolates between the
    classic heuristics so the search space provably CONTAINS a good answer:
      best-fit  ~ maximize -(remaining-item)  (tightest fit)
      worst-fit ~ maximize  (remaining-item)
    plus a nonlinearity exponent. The proposer's job is to find the mix."""
    slack = remaining - item                      # >=0 guaranteed by caller
    a = genome.get("a", 0.0)                       # weight on tightness
    b = genome.get("b", 0.0)                       # weight on raw remaining
    p = genome.get("p", 1.0)                       # slack exponent
    p = max(0.1, min(4.0, p))
    return -a * (slack ** p) + b * remaining


def _run_heuristic(genome: dict, items: list[float]) -> int:
    bins: list[float] = []                         # remaining capacity per open bin
    for item in items:
        best_idx, best_pri = -1, float("-inf")
        for i, rem in enumerate(bins):
            if rem + 1e-9 >= item:                 # fits
                pri = _priority(genome, item, rem)
                if pri > best_pri:
                    best_pri, best_idx = pri, i
        if best_idx == -1:
            bins.append(CAPACITY - item)           # open new bin
        else:
            bins[best_idx] -= item
    return len(bins)


def _lower_bound(items: list[float]) -> float:
    return sum(items) / CAPACITY                   # continuous LB on bins needed


class BinPackingHandle(VerificationHandle):
    name = "bin-packing"
    klass = 2

    def verify(self, candidate: dict) -> Verdict:
        try:
            total_bins = total_lb = 0.0
            for items in _INSTANCES:
                total_bins += _run_heuristic(candidate, items)
                total_lb += _lower_bound(items)
            # score: negative mean excess bins over the lower bound.
            # higher (closer to 0) = better. perfect packing -> ~0.
            excess = (total_bins - total_lb) / len(_INSTANCES)
            return Verdict(score=-excess, valid=True, handle=self.name, klass=self.klass,
                           detail={"mean_bins": total_bins / len(_INSTANCES),
                                   "mean_lb": total_lb / len(_INSTANCES),
                                   "mean_excess": excess})
        except Exception as e:  # a malformed genome must score -inf, never crash the loop
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)})


def _run_compiled(priority, items: list[float]) -> int:
    """Same greedy driver as _run_heuristic but the per-bin priority is an
    arbitrary compiled function. NaN/inf scores are treated as 'never pick'."""
    import math as _m
    bins: list[float] = []
    for item in items:
        best_idx, best_pri = -1, float("-inf")
        for i, rem in enumerate(bins):
            if rem + 1e-9 >= item:
                pri = priority(item, rem)
                if pri == pri and pri != float("inf") and pri != float("-inf") \
                        and pri > best_pri:  # pri==pri rejects NaN
                    best_pri, best_idx = pri, i
        if best_idx == -1:
            bins.append(CAPACITY - item)
        else:
            bins[best_idx] -= item
    return len(bins)


class CodeBinPackingHandle(VerificationHandle):
    """Phase-1 disposer: the candidate is LLM-written `priority` source code.
    Identical scoring to BinPackingHandle, so scores are directly comparable
    to the named baselines and the param-search results."""
    name = "bin-packing"   # same problem key -> same ledger/leaderboard
    klass = 2

    def verify(self, candidate: str) -> Verdict:
        try:
            priority = compile_priority(candidate)
            total_bins = total_lb = 0.0
            with time_limit(5.0):
                for items in _INSTANCES:
                    total_bins += _run_compiled(priority, items)
                    total_lb += _lower_bound(items)
            excess = (total_bins - total_lb) / len(_INSTANCES)
            return Verdict(score=-excess, valid=True, handle=self.name, klass=self.klass,
                           detail={"mean_bins": total_bins / len(_INSTANCES),
                                   "mean_lb": total_lb / len(_INSTANCES),
                                   "mean_excess": excess})
        except (UnsafeCode, CodeTimeout, SyntaxError, NameError, TypeError,
                ValueError, ZeroDivisionError, OverflowError, RecursionError) as e:
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)[:200]})


# Reference baselines so we can SEE the loop climb past known heuristics.
BEST_FIT = {"a": 1.0, "b": 0.0, "p": 1.0}
WORST_FIT = {"a": -1.0, "b": 0.0, "p": 1.0}
FIRST_FIT = {"a": 0.0, "b": 0.0, "p": 1.0}      # ties broken by first index = first-fit


# ---- proposer glue (code-writing proposers) -------------------------------
_CONTRACT = '''\
def priority(item, remaining):
    # item:      float, size of the item to place, 0 < item <= 1
    # remaining: float, remaining capacity of ONE candidate bin, item <= remaining <= 1
    # return:    float, higher == this bin is a better place for the item
    ...'''

SYS = (
    "You are a combinatorial-optimization researcher inventing online "
    "bin-packing heuristics. You write small, fast, correct Python. You only "
    "use the parameters given and the `math` module. No imports, no I/O, no "
    "global state. You favor clever, non-obvious scoring rules over textbook ones."
)


def build_prompt(examples: list[dict], n: int) -> str:
    lines = [
        "ONLINE BIN PACKING. Items arrive one at a time and must be placed "
        "immediately into an open bin that fits, or a new bin. Bin capacity = 1.0.",
        "Items go to the bin your `priority` function scores HIGHEST among "
        "those that fit. Goal: use as FEW bins as possible (score = negative "
        "mean excess bins over the continuous lower bound; closer to 0 is better).",
        "",
        "Write a `priority(item, remaining)` function. Contract:",
        "```python", _CONTRACT, "```",
        "",
        "Known reference: best-fit (`priority = -(remaining - item)`, i.e. "
        "tightest fit) is a strong baseline. Your job is to BEAT it.",
    ]
    if examples:
        lines += ["", "CURRENT BEST PROGRAMS (score: higher is better) — beat these:"]
        for e in examples:
            lines += [f"# score = {e['score']:+.4f}", "```python", e["code"].strip(), "```"]
    lines += [
        "",
        f"Output EXACTLY {n} NEW, DISTINCT candidate functions, each in its own "
        "```python``` block, each a complete `def priority(item, remaining):`. "
        "No prose outside the code blocks.",
    ]
    return "\n".join(lines)


def seed_programs() -> list[str]:
    return [
        "def priority(item, remaining):\n    return -(remaining - item)",  # best-fit
        "def priority(item, remaining):\n    return 0.0",                  # first-fit
    ]
