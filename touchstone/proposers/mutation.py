"""Cheap proposer #0: parameter mutation. NO LLM cost.

Its only job is to prove the spine — proposer -> disposer -> ledger ->
artifact — climbs against a deterministic scorer. If a blind mutation
search beats the named baselines, the harness is sound and we've earned the
right to pay for the expensive proposer (LLM code-synthesis) in Phase 1.
"""
from __future__ import annotations

import random
from typing import Any


class MutationProposer:
    name = "mutation"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def seed_genome(self) -> dict:
        return {"a": self.rng.uniform(-1, 1),
                "b": self.rng.uniform(-1, 1),
                "p": self.rng.uniform(0.5, 2.0)}

    def mutate(self, parent: dict) -> dict:
        child = dict(parent)
        key = self.rng.choice(["a", "b", "p"])
        child[key] = child[key] + self.rng.gauss(0, 0.3)
        return child
