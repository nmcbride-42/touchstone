"""Example 2 — the discriminating verifier (offline; LLM loop optional). Demonstrate that composing
extrapolation + parsimony with interpolation catches overfits that plain
held-out R² waves through — the leap from "it fits" to "it's probably true".

Three hand-crafted forms run through the robust verdict:
  - TRUE damped cosine          -> should be ACCEPTED
  - OVERFIT degree-9 polynomial -> naive interp passes; robust REJECTS (extrap)
  - SIMPLE WRONG linear         -> rejected everywhere

Then (optionally) the LLM loop runs under the robust verifier.

    python examples/02_discriminating_verifier.py [--llm GENERATIONS]

The --llm option needs a configured provider (see README, .env.example).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from touchstone.problems import symbolic_regression as sr

FORMS = {
    "TRUE damped-cosine (4p)":
        "import math\ndef f(x, a, b, w, p):\n    return a*math.exp(-b*x)*math.cos(w*x+p)",
    "OVERFIT degree-9 poly (10p)":
        ("def f(x, a, b, c, d, e, g, h, i, j, k):\n"
         "    return a*x**9+b*x**8+c*x**7+d*x**6+e*x**5+g*x**4+h*x**3+i*x**2+j*x+k"),
    "SIMPLE WRONG linear (2p)":
        "def f(x, a, b):\n    return a*x + b",
}


def main():
    h = sr.RobustSymRegHandle()
    print("DISCRIMINATING VERIFIER")
    print("fit on interior x<=4.5; test interpolation (held-out interior) AND")
    print(f"extrapolation (x>4.5, out of range); gate extrap_r2 > {sr._EXTRAP_GATE}\n")
    print(f"{'form':<30} {'interp':>8} {'extrap':>8} {'cx_dl':>7} {'naive?':>7} {'ROBUST':>8}")
    print("-" * 74)
    for name, code in FORMS.items():
        v = h.verify(code)
        d = v.detail
        if not v.valid:
            print(f"{name:<30} INVALID: {d.get('error')}")
            continue
        naive = "PASS" if d["naive_would_pass"] else "fail"
        robust = "ACCEPT" if d["accepted"] else "REJECT"
        print(f"{name:<30} {d['interp_r2']:>8.3f} {d['extrap_r2']:>8.3f} "
              f"{d['complexity_dl']:>7.1f} {naive:>7} {robust:>8}")

    print("\nThe point: the overfit polynomial PASSES naive held-out R² but the")
    print("robust verifier REJECTS it on extrapolation — plain R² cannot tell an")
    print("overfit from the real law; composing extrapolation + parsimony can.")

    if len(sys.argv) > 2 and sys.argv[1] == "--llm":
        _run_llm(int(sys.argv[2]))
    return 0


def _run_llm(gens):
    from touchstone.config import output_dir
    from touchstone.core.ledger import Ledger
    from touchstone.proposers.llm import CodeProposer, client_from_env
    out = output_dir(); out.mkdir(parents=True, exist_ok=True)
    h = sr.RobustSymRegHandle()
    ledger = Ledger(out / "symreg_robust.db")
    proposer = CodeProposer(client_from_env())
    best = {"score": float("-inf"), "code": None, "accepted": False}
    print("\n" + "=" * 60 + "\nLLM under the robust verifier:")
    for code in sr.seed_programs():
        if not ledger.seen("symreg-robust", code):
            ledger.record("symreg-robust", code, h.verify(code), 0)
    for gen in range(1, gens + 1):
        ex = [{"code": e["genome"], "score": e["score"]} for e in ledger.best("symreg-robust", 3)]
        codes = proposer.propose_raw(sr.build_prompt(ex, 4), sr.SYS, fn_name="f")
        print(f"  gen{gen}: {proposer.last_meta} -> {len(codes)} forms")
        for code in codes:
            if ledger.seen("symreg-robust", code):
                continue
            v = h.verify(code); ledger.record("symreg-robust", code, v, gen)
            if v.valid and v.score > best["score"]:
                best.update(score=v.score, code=code, accepted=v.detail["accepted"])
                print(f"    NEW BEST composite={v.score:+.3f} "
                      f"interp={v.detail['interp_r2']} extrap={v.detail['extrap_r2']} "
                      f"accepted={v.detail['accepted']}")
        if best["accepted"] and best["score"] > 0.4:
            break
    print(f"\nbest model (accepted={best['accepted']}):\n{best['code']}")


if __name__ == "__main__":
    raise SystemExit(main())
