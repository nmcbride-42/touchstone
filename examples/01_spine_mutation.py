"""Example 1 — the spine, with no LLM.

Does the generate -> verify -> ledger loop climb past named baselines on an
deterministic scorer, using only a blind parameter-mutation proposer? The
search space is a 3-parameter family that contains best-fit, first-fit and
worst-fit, and best-fit is that family's ceiling on this benchmark, so the
expected outcome is MATCHING best-fit, not beating it. Many genomes in the
family pack identically to best-fit, so a random seed often matches it
immediately: this example exercises the plumbing (proposer, sandbox-free
scorer, dedup ledger, baselines, new-best hook), not search power.
Deterministic (seeded), offline, runs in a few seconds.

    python examples/01_spine_mutation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from touchstone.core.evolve import EvolveConfig, evolve
from touchstone.core.ledger import Ledger
from touchstone.problems import bin_packing as bp
from touchstone.proposers.mutation import MutationProposer

PROBLEM = "bin-packing-v0"


def main():
    handle = bp.BinPackingHandle()

    # 1. Establish the bar: score the classic named heuristics.
    baselines = {name: handle.verify(g).score for name, g in
                 [("first-fit", bp.FIRST_FIT), ("best-fit", bp.BEST_FIT),
                  ("worst-fit", bp.WORST_FIT)]}
    best_baseline = max(baselines.values())
    print("baselines (score = -mean excess bins over lower bound; higher=better):")
    for n, s in baselines.items():
        print(f"  {n:10s} {s:+.4f}")
    print(f"  bar to beat: best-named = {best_baseline:+.4f}\n")

    # 2. Run the spine.
    ledger = Ledger(":memory:")
    proposer = MutationProposer(seed=7)
    new_bests = []
    result = evolve(proposer, handle, ledger,
                    EvolveConfig(problem=PROBLEM, generations=40),
                    on_new_best=lambda b: new_bests.append(b))

    print(f"evaluated {ledger.count(PROBLEM)} candidates, "
          f"{len(new_bests)} new-best events")
    print(f"discovered: score={result['score']:+.4f} genome={result['genome']}")
    print(f"  mean bins/instance: {result['detail'].get('mean_bins'):.3f} "
          f"(lower bound {result['detail'].get('mean_lb'):.3f})\n")

    delta = result["score"] - best_baseline
    if delta > 1e-6:
        print(f"LOOP CLOSES: spine beat best-named baseline by {delta:+.4f}. "
              f"Harness sound — earned the LLM proposer.")
        return 0
    elif delta > -1e-6:
        print(f"LOOP CLOSES: spine MATCHED best-named baseline ({delta:+.4f}). "
              f"Search found the known optimum region — harness sound.")
        return 0
    else:
        print(f"LOOP DID NOT CLIMB: {delta:+.4f} short of baseline. Fix before LLM.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
