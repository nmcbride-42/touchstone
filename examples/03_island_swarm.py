"""Example 3 — LLM island swarm against an exact or held-out verifier.

Several islands (each an LLM proposer with its own strategy hint, optionally a
different model family) write candidate programs in parallel; every candidate
is compiled in the sandbox and scored by the problem's VerificationHandle;
each island's best migrates around a ring after every generation. Everything
is recorded in a SQLite ledger plus a JSON trajectory, and the best program is
written out as an artifact you can re-verify later.

    python examples/03_island_swarm.py --problem apfree   [--generations 2 --per-gen 2]
    python examples/03_island_swarm.py --problem binpack
    python examples/03_island_swarm.py --problem lightcurve
    python examples/03_island_swarm.py --problem symreg

Islands come from TOUCHSTONE_SWARM, a comma-separated list of providers, one
per island (default: "anthropic,anthropic"). Mixing families is the point, e.g.
TOUCHSTONE_SWARM=anthropic,openai,anthropic,openai. Provider settings are in
.env.example. This example makes real (paid) LLM calls: islands x generations
calls in total.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from touchstone.config import env, output_dir
from touchstone.core.ledger import Ledger
from touchstone.problems import apfree, binpack_trimodal, lightcurve, symbolic_regression
from touchstone.proposers.llm import client_from_env
from touchstone.proposers.swarm import evolve_swarm, make_islands

# problem -> (handle factory, seeds, prompt builder, system prompt, fn name,
#             stop score, strategy hints cycled across islands)
PROBLEMS = {
    "apfree": (
        apfree.ApFreeHandle, lambda: [apfree.POWERS_OF_TWO, apfree.BASE3],
        apfree.build_prompt_beat_base3, apfree.SYS_BEAT_BASE3, "construct", None, [
            "Fast forbidden-array greedy passes over many deterministic orderings "
            "(digit penalties in several bases, modular strides, shell distances); "
            "return the largest set.",
            "Rank integers by how few new 3-AP conflicts they create, with "
            "multi-pass deterministic sweeps.",
            "Combine small exact optima into larger sets with block constructions, "
            "then improve with remove-and-refill local search.",
            "Synthesize the best ordering families from the elite examples and migrants.",
        ]),
    "binpack": (
        binpack_trimodal.TrimodalBinPackingHandle, binpack_trimodal.seed_programs,
        binpack_trimodal.build_prompt, binpack_trimodal.SYS, "priority", None, [
            "Penalize dead-zone slack in (0.025, 0.165) where no future item fits; "
            "reward tight fits and reusable gaps (~0.205, ~0.32, ~0.41).",
            "Score slack by how well it matches sums of the dominant item modes.",
            "Use clean piecewise priority bands: tight fit, one-item residual, "
            "two-item residual, dead-space avoidance.",
            "Refine thresholds and penalties from the best migrants.",
        ]),
    "lightcurve": (
        lightcurve.LightCurveHandle, lightcurve.seed_programs,
        lightcurve.build_prompt, lightcurve.SYS, "f", None, [
            "Prefer the fewest parameters that capture both eclipses.",
            "Explore flat-bottomed eclipse profiles (super-Gaussian, sech², trapezoid).",
        ]),
    "symreg": (
        symbolic_regression.SymbolicRegressionHandle, symbolic_regression.seed_programs,
        symbolic_regression.build_prompt, symbolic_regression.SYS, "f", 0.95, [
            "Try decaying, oscillating and growth families.",
            "Prefer compact physically plausible forms over polynomials.",
        ]),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problem", choices=sorted(PROBLEMS), default="apfree")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--per-gen", type=int, default=2)
    a = ap.parse_args()

    handle_cls, seeds, build_prompt, system, fn_name, stop, hints = PROBLEMS[a.problem]
    providers = [p.strip() for p in env("TOUCHSTONE_SWARM", "anthropic,anthropic").split(",")
                 if p.strip()]
    islands = make_islands([(client_from_env(p), f"{p}_{i}", hints[i % len(hints)])
                            for i, p in enumerate(providers)])

    out = output_dir(); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ledger = Ledger(out / f"{a.problem}_{stamp}.db")
    handle = handle_cls()

    print(f"problem={a.problem} islands={providers} generations={a.generations} "
          f"per_gen={a.per_gen}")
    for code in seeds():
        print(f"  seed score: {handle.verify(code).score:+.4f}")

    t0 = time.time()
    best = evolve_swarm(problem=a.problem, handle=handle, ledger=ledger, seeds=seeds(),
                        build_prompt=build_prompt, system=system, fn_name=fn_name,
                        islands=islands, generations=a.generations, per_gen=a.per_gen,
                        stop_score=stop)
    elapsed = round(time.time() - t0, 1)

    if best["code"] is None:
        print("no valid candidate")
        return 1
    print(f"\nbest score {best['score']:+.4f} (island {best['island']}, "
          f"{best['role']}, generation {best['generation']}) in {elapsed}s")
    print(f"detail: {best['detail']}")

    art = out / f"best_{a.problem}_{stamp}.py"
    art.write_text(f"# touchstone {a.problem}: score {best['score']:+.4f}\n"
                   f"# detail: {json.dumps(best['detail'], default=str)}\n"
                   f"# island {best['island']} ({best['role']}), generation "
                   f"{best['generation']}\n\n{best['code']}\n")
    (out / f"trajectory_{a.problem}_{stamp}.json").write_text(json.dumps(
        {"problem": a.problem, "islands": providers, "generations": a.generations,
         "per_gen": a.per_gen, "elapsed_s": elapsed,
         "best": {k: v for k, v in best.items() if k != "history"},
         "history": best["history"]}, indent=2, default=str))
    print(f"artifact: {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
