"""The disposer. The whole game lives here.

A VerificationHandle is a deterministic external scorer. The proposer
(an LLM, a mutation, anything) never needs to be correct — the handle
enforces correctness. Candidate code reaches the handle only through
`core.sandbox`, which blocks every route we know of from the candidate to the
scorer's state; `core.isolate` re-verifies a result in a fresh process. This is the FunSearch/AlphaEvolve principle and it
is the one non-negotiable architectural choice of the engine.

Handles are ranked by verification CLASS, cheapest+most-trustworthy first:

  1 formal-proof-checkable   (Lean / SMT kernel)        — binary, deterministic
  2 simulation/scorer        (fitness function)         — cheap, ours, absolute*
  3 held-out-data-predict    (OOD / holdout fit)        — cheap; good fit != truth
  4 reproducible-anomaly     (re-run + provenance)      — self-certifies, not causal
  5 real-world-experiment    (wet lab / cloud lab)      — only real ground truth, slow+$$$
  6 robotic-synth+char       (materials, hours-days)    — the real frontier today

A solo compute agent with no wet lab can FULLY close 1-3, partially
self-certify 4, and CANNOT close 5-6 alone (capital-gated). The honest
output for 4-6 is a ranked, falsifiable, citation-grounded candidate
packet awaiting someone else's experiment — never a "discovery."

*class 2 is "absolute" only w.r.t. the scorer: a high score is ground
 truth about the objective, not necessarily about the real-world question
 the objective is a proxy for. Choosing scorers whose objective IS the
 question is Phase 0's discipline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Verdict:
    """What the disposer returns. score is the only thing evolution optimizes;
    everything else is provenance so the ledger can prove the result later."""
    score: float                      # higher = better; -inf = invalid candidate
    valid: bool                       # did the candidate even run / typecheck?
    detail: dict[str, Any] = field(default_factory=dict)
    handle: str = ""                  # which disposer produced this
    klass: int = 0                    # verification class 1-6


class VerificationHandle(ABC):
    """Plug a new disposer in to point the engine at a new domain.
    'Domain' reduces to 'which handle you register.'"""

    name: str = "abstract"
    klass: int = 0

    @abstractmethod
    def verify(self, candidate: Any) -> Verdict:
        """Score a candidate against ground truth the proposer does not control.
        Must be deterministic for a given candidate (so results replay)."""
        ...


class VerifierRegistry:
    def __init__(self) -> None:
        self._handles: dict[str, VerificationHandle] = {}

    def register(self, handle: VerificationHandle) -> VerificationHandle:
        self._handles[handle.name] = handle
        return handle

    def get(self, name: str) -> VerificationHandle:
        return self._handles[name]

    def __contains__(self, name: str) -> bool:
        return name in self._handles
