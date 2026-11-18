"""Class-2 disposer #2: online bin packing on a *trap* distribution, with a
held-out instance set.

The candidate is LLM-written `def priority(item, remaining):` source code. Items
are packed online (capacity 1.0) into the open bin whose priority is highest;
the score is the negative mean relative excess over the trivial lower bound
ceil(sum(items)), on TRAIN instances. The same candidate is also run on
HELD-OUT instances (same generator, different seed) and both numbers are
reported, so a heuristic that only memorized the training streams shows up.

The distribution is deliberately adversarial for best-fit. Item sizes are a
three-mode mixture (Gaussians around 0.58, 0.32 and 0.205, clipped to
[0.16, 0.68]). Because no item is smaller than 0.16, any bin whose slack lands
in roughly (0.02, 0.16) is wasted forever, and best-fit's "tightest fit" rule
walks straight into that dead zone.

Honest scope: the instances are synthetic and were *designed* so best-fit is
beatable; the prompt (`build_prompt`) tells the proposer the item modes and the
dead-zone mechanism. A heuristic that wins here is tuned to this distribution.
It is not evidence of a better general-purpose bin-packing heuristic, and the
held-out set tests generalization across seeds of the same generator only.
(Earlier drafts of this module called the generator "Weibull"; it is not.)
"""
from __future__ import annotations

import functools
import math

import numpy as np

from ..core.sandbox import compile_function, time_limit
from ..core.verifier import VerificationHandle, Verdict

TRAIN_SEED = 20260618
HELDOUT_SEED = 20261029
N_INSTANCES = 20
N_ITEMS = 120


def make_instances(n_inst: int, n_items: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(n_inst):
        items = []
        for _ in range(n_items):
            u = rng.random()
            if u < 0.36:
                w = 0.58 + rng.normal(0.0, 0.025)   # anchor: leaves a ~0.42 gap
            elif u < 0.68:
                w = 0.32 + rng.normal(0.0, 0.02)    # trap: 0.42 - 0.32 = dead space
            else:
                w = 0.205 + rng.normal(0.0, 0.015)  # pair: two fill a ~0.42 gap
            items.append(float(np.clip(w, 0.16, 0.68)))
        instances.append(np.round(np.array(items, dtype=float), 4))
    return instances


@functools.lru_cache(maxsize=1)
def instances() -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (make_instances(N_INSTANCES, N_ITEMS, TRAIN_SEED),
            make_instances(N_INSTANCES, N_ITEMS, HELDOUT_SEED))


def _pack_stream(items: np.ndarray, priority_fn) -> int:
    rem: list[float] = []
    for item in items:
        w = float(item)
        best_idx = -1
        best_score = float("-inf")
        for i, r in enumerate(rem):
            if r + 1e-9 >= w:
                s = float(priority_fn(w, r))
                if not math.isfinite(s):
                    raise ValueError("priority returned non-finite value")
                if s > best_score:          # strict: ties go to the earliest bin
                    best_score = s
                    best_idx = i
        if best_idx >= 0 and best_score > -1e9:
            rem[best_idx] -= w
        else:
            rem.append(1.0 - w)
    return len(rem)


def mean_excess(instance_set: list[np.ndarray], priority_fn) -> float:
    """Mean of (bins_used - LB) / LB with LB = ceil(sum(items))."""
    excess = []
    for items in instance_set:
        lb = math.ceil(float(np.sum(items)) - 1e-9)
        excess.append((_pack_stream(items, priority_fn) - lb) / lb)
    return float(np.mean(excess))


FIRST_FIT = "def priority(item, remaining):\n    return 0.0\n"
BEST_FIT = "def priority(item, remaining):\n    return -(remaining - item)\n"


@functools.lru_cache(maxsize=1)
def baselines() -> dict[str, float]:
    tr, ho = instances()
    ff = lambda w, r: 0.0          # noqa: E731
    bf = lambda w, r: -(r - w)     # noqa: E731
    return {"ff_train": mean_excess(tr, ff), "ff_heldout": mean_excess(ho, ff),
            "bf_train": mean_excess(tr, bf), "bf_heldout": mean_excess(ho, bf)}


class TrimodalBinPackingHandle(VerificationHandle):
    name = "bin-packing-trimodal-heldout"
    klass = 2

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def verify(self, candidate: str) -> Verdict:
        try:
            fn = compile_function(candidate, "priority")
            tr, ho = instances()
            b = baselines()
            with time_limit(self.timeout_s):
                train_exc = mean_excess(tr, fn)
                heldout_exc = mean_excess(ho, fn)
            return Verdict(
                score=-train_exc, valid=True, handle=self.name, klass=self.klass,
                detail={
                    "train_excess": round(train_exc, 5),
                    "heldout_excess": round(heldout_exc, 5),
                    "bf_train_excess": round(b["bf_train"], 5),
                    "bf_heldout_excess": round(b["bf_heldout"], 5),
                    "ff_train_excess": round(b["ff_train"], 5),
                    "ff_heldout_excess": round(b["ff_heldout"], 5),
                    "beats_best_fit_train": train_exc < b["bf_train"] - 1e-4,
                    "beats_best_fit_heldout": heldout_exc < b["bf_heldout"] - 1e-4,
                },
            )
        except Exception as e:
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)[:180]})


# ---- proposer glue -------------------------------------------------------
SYS = (
    "You are an operations-research scientist evolving online bin-packing "
    "priority functions in pure Python (`math` only) for capacity-1.0 bins."
)


def build_prompt(examples: list[dict], n: int) -> str:
    b = baselines()
    lines = [
        "Design `def priority(item, remaining):` for online 1D bin packing (bin capacity = 1.0).",
        "Each arriving `item` in [0.16, 0.68] is placed in the open bin with `remaining >= item`",
        "that maximizes `priority(item, remaining)`. (If all open bins return <= -1e9 or none fit, a new bin opens.)",
        "",
        "WHY BEST-FIT IS SUBOPTIMAL HERE:",
        f"- On this distribution, First-Fit has {100 * b['ff_train']:.2f}% excess bins and "
        f"Best-Fit has {100 * b['bf_train']:.2f}% excess bins.",
        "- All items have size >= 0.16. When `slack = remaining - item` lands in the DEAD ZONE (0.02, 0.155),",
        "  that slack can never fit another item. Best-Fit prefers slack 0.09 over slack 0.21 and wastes it;",
        "  leaving slack ~0.20-0.23, ~0.40-0.44 or < 0.025 lets future items (~0.205, ~0.32, ~0.58) pack tightly.",
        "",
        "Score is `-mean_excess_bins` (closer to 0.0 is better; beat Best-Fit's "
        f"{-b['bf_train']:+.4f}).",
    ]
    if examples:
        lines += ["", "BEST HEURISTICS SO FAR (higher score = fewer bins):"]
        for e in examples:
            lines += [f"# score = {e['score']:+.5f}", "```python", e["code"].strip(), "```"]
    lines += [
        "",
        f"Output EXACTLY {n} DISTINCT `def priority(item, remaining):` functions, "
        "each in its own ```python``` block. No prose outside the blocks.",
    ]
    return "\n".join(lines)


def seed_programs() -> list[str]:
    return [FIRST_FIT, BEST_FIT]
