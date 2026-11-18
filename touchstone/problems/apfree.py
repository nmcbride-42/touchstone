"""Class-1 disposer: largest 3-term-AP-free subset of {0, ..., N-1}.

This is the FIRST handle with ABSOLUTE, EXACT verification — no held-out, no
R², no statistics. The candidate is a CONSTRUCTOR `construct(N) -> set`; the
verifier exhaustively checks the set contains no 3-term arithmetic
progression (no a, b, c with a + c = 2b) and that every element is a valid
integer in range. Correctness is BINARY and machine-checked. The score is the
set size — an optimization over the space of provably-correct objects.

This is FunSearch's actual territory (cap sets / AP-free sets). We demonstrate
the class-1 loop end-to-end; matching/finding a strong known construction is
the bar (beating world-record bounds is a future compute-heavy frontier, not
a one-arc demo — stated plainly so the result isn't oversold).
"""
from __future__ import annotations

from ..core.sandbox import compile_function, time_limit
from ..core.verifier import VerificationHandle, Verdict

N = 1000


def is_ap_free(s: set[int]) -> tuple[bool, tuple]:
    """EXACT check: no 3-term AP. Returns (ok, witness). O(|s|^2)."""
    arr = sorted(s)
    members = set(arr)
    for i, a in enumerate(arr):
        for c in arr[i + 1:]:
            if (a + c) % 2 == 0 and (a + c) // 2 in members:
                b = (a + c) // 2
                if b != a and b != c:
                    return False, (a, b, c)         # a, b, c is an AP
    return True, ()


def _validate(raw, n) -> set[int]:
    """Coerce a constructor's output to a clean set of in-range ints, or raise."""
    s = set()
    for v in raw:
        iv = int(v)
        if iv != v or iv < 0 or iv >= n:
            raise ValueError(f"element {v!r} not an int in [0,{n})")
        s.add(iv)
    return s


class ApFreeHandle(VerificationHandle):
    name = "apfree"
    klass = 1

    def __init__(self, n: int = N):
        self.n = n

    def verify(self, candidate: str) -> Verdict:
        try:
            construct = compile_function(candidate, "construct")
            with time_limit(5.0):
                raw = construct(self.n)
                # inside the alarm: `raw` may be a generator, so iterating it
                # still runs candidate code
                s = _validate(raw, self.n)
            ok, witness = is_ap_free(s)
            if not ok:
                # EXACT correctness gate: any AP -> invalid, with the witness.
                return Verdict(score=float("-inf"), valid=False, handle=self.name,
                               klass=self.klass,
                               detail={"error": f"contains AP {witness}", "size": len(s)})
            return Verdict(score=float(len(s)), valid=True, handle=self.name,
                           klass=self.klass,
                           detail={"size": len(s), "ap_free_proven": True,
                                   "n": self.n})
        except Exception as e:
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)[:200]})


# Reference constructions (scored by the same exact verifier).
POWERS_OF_TWO = (
    "def construct(N):\n"
    "    s, k = [], 1\n"
    "    while k < N:\n        s.append(k); k *= 2\n    return s")  # weak: ~log2(N)

GREEDY = (
    "def construct(N):\n"
    "    s = []\n    sset = set()\n"
    "    for k in range(N):\n"
    "        ok = True\n"
    "        for b in s:\n"
    "            a = 2*b - k\n"            # a, b, k is an AP (a<b<k) if a in set
    "            if a >= 0 and a in sset:\n                ok = False; break\n"
    "        if ok:\n            s.append(k); sset.add(k)\n"
    "    return s")                                                  # strong: Stanley ~N^0.63


# ---- proposer glue -------------------------------------------------------
SYS = (
    "You are a combinatorialist. You write a Python `construct(N)` that returns "
    "a LARGE subset of {0,...,N-1} with NO three-term arithmetic progression "
    "(no a,b,c in the set with a+c=2b). Use only `math`. Correctness is checked "
    "exactly and a single AP disqualifies the whole set — so be certain it is "
    "AP-free. Larger valid sets score higher. Known strong idea: integers whose "
    "base-3 representation avoids the digit 2 are AP-free."
)


def build_prompt(examples, n):
    lines = [
        f"Construct the LARGEST subset of {{0,...,{N-1}}} with no 3-term "
        "arithmetic progression (no a,b,c with a+c=2b).",
        "Write `def construct(N):` returning a list/set of ints in [0,N). Use "
        "only `math`. Correctness is verified EXACTLY — any single AP makes the "
        "whole candidate invalid, so guarantee AP-freeness by construction.",
        "Score = size of the set. Beat the forms below.",
    ]
    if examples:
        lines += ["", "BEST SO FAR (size) — beat these:"]
        for e in examples:
            lines += [f"# size = {e['score']:.0f}", "```python", e["code"].strip(), "```"]
    lines += ["", f"Output EXACTLY {n} DISTINCT `def construct(N):` strategies, "
              "each in its own ```python``` block. No prose outside the blocks."]
    return "\n".join(lines)


def seed_programs():
    return [POWERS_OF_TWO]      # deliberately weak start; clear room to climb


# ---- harder variant: start FROM the base-3 construction and try to beat it --
BASE3 = (
    "def construct(N):\n"
    "    out = []\n"
    "    for x in range(N):\n"
    "        y, ok = x, True\n"
    "        while y > 0:\n"
    "            if y % 3 == 2:\n"
    "                ok = False\n"
    "                break\n"
    "            y //= 3\n"
    "        if ok:\n"
    "            out.append(x)\n"
    "    return out\n")       # integers with no digit 2 in base 3: size 105 at N=1000

SYS_BEAT_BASE3 = (
    "You are a combinatorial number theorist and algorithm designer writing "
    "pure Python (`math` only, no other imports, < 3.5 seconds runtime) "
    "to construct the largest possible 3-AP-free subset S of {0, ..., N-1}."
)


def build_prompt_beat_base3(examples, n):
    """Prompt used for the island-swarm run: it names the base-3 baseline and
    suggests fast greedy-with-forbidden-array search over many orderings."""
    lines = [
        f"Construct a 3-term-arithmetic-progression-free subset of {{0, ..., {N - 1}}}.",
        "No three distinct elements a < b < c in S may satisfy a + c == 2*b.",
        "",
        "GOAL: the base-3 construction (integers with no digit 2 in base 3) gives",
        f"size 105 for N={N}. STRICTLY BEAT 105.",
        "",
        "Useful ideas:",
        "1. Incremental greedy with a `forbidden = [False] * N` array: when adding x,",
        "   for each prior s mark 2*x - s, 2*s - x and (x + s)/2 (when integral) as forbidden.",
        "   One pass over 1000 numbers is about a millisecond in pure Python.",
        "2. So `construct(N)` can try dozens of deterministic orderings of range(N)",
        "   (digit-penalty keys in several bases, modular strides, distance from the",
        "   center, reflections) and return the largest valid set.",
        "3. Small exact optima can seed larger sets via block constructions; local",
        "   search (remove one element, re-add freed ones) can add a few more.",
        "",
        "SANDBOX RULES: define `def construct(N):` returning a list of ints in [0, N-1].",
        "Import nothing except `math` (no random, numpy, itertools). For tie-breaking",
        "use deterministic arithmetic hashes.",
    ]
    if examples:
        lines += ["", "CURRENT ELITE CONSTRUCTIONS (beat these sizes):"]
        for e in examples:
            lines += [f"# verified 3-AP-free size = {int(e['score'])}", "```python",
                      e["code"].strip(), "```"]
    lines += ["", f"Output EXACTLY {n} DISTINCT `def construct(N):` functions, each in "
              "its own ```python``` block. No prose outside the blocks."]
    return "\n".join(lines)
