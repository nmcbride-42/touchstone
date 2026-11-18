"""The loop. Generate -> verify -> keep the fit -> repeat.

Deliberately minimal: a (mu+lambda)-style evolutionary search over a ledger.
The proposer and the disposer are both injected, so the same loop drives a
mutation search over heuristic params OR an LLM writing code, against a
proof checker OR a simulator. That injectability IS the spine.

Adversarial gate: before a NEW best is *blessed* as a result (not just kept
as a stepping stone), it can be sent through an external refutation pass.
That hook is left as `on_new_best` so callers can plug in whatever review
they trust. For class-1/2 self-validating scores the gate is a formality
(the scorer is ground truth); it matters for classes 4-6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class EvolveConfig:
    problem: str
    generations: int = 40
    pop_size: int = 12          # mu: survivors
    offspring: int = 24         # lambda: children per generation
    seed_pop: int = 12


def evolve(proposer, handle, ledger, cfg: EvolveConfig,
           on_new_best: Optional[Callable[[dict], None]] = None) -> dict:
    population: list[tuple[float, str, dict]] = []  # (score, key, genome)
    best_score = float("-inf")

    def consider(genome: dict, generation: int, parent_key: str | None):
        nonlocal best_score
        if ledger.seen(cfg.problem, genome):
            return None
        verdict = handle.verify(genome)
        key = ledger.record(cfg.problem, genome, verdict, generation, parent_key)
        if verdict.valid:
            population.append((verdict.score, key, genome))
            if verdict.score > best_score:
                best_score = verdict.score
                if on_new_best:
                    on_new_best({"genome": genome, "score": verdict.score,
                                 "detail": verdict.detail, "generation": generation})
        return key

    # seed
    for _ in range(cfg.seed_pop):
        consider(proposer.seed_genome(), 0, None)

    for gen in range(1, cfg.generations + 1):
        population.sort(key=lambda t: t[0], reverse=True)
        survivors = population[:cfg.pop_size]
        population = list(survivors)
        for _ in range(cfg.offspring):
            if not survivors:
                break
            _, pkey, pgenome = proposer.rng.choice(survivors) \
                if hasattr(proposer, "rng") else survivors[0]
            consider(proposer.mutate(pgenome), gen, pkey)

    top = ledger.best(cfg.problem, 1)
    return top[0] if top else {}
