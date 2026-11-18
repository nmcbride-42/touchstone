from touchstone.core.complexity import complexity
from touchstone.core.evolve import EvolveConfig, evolve
from touchstone.core.ledger import Ledger
from touchstone.core.verifier import Verdict, VerificationHandle, VerifierRegistry
from touchstone.problems import apfree, bin_packing
from touchstone.proposers.mutation import MutationProposer
from touchstone.proposers.swarm import Island, evolve_swarm


class Const(VerificationHandle):
    name, klass = "const", 2

    def verify(self, candidate):
        return Verdict(score=float(candidate["x"]), valid=True, handle=self.name, klass=2)


def test_ledger_dedup_and_best():
    led = Ledger(":memory:")
    h = Const()
    for x in (1, 3, 2, 3):
        g = {"x": x}
        if not led.seen("p", g):
            led.record("p", g, h.verify(g), 0)
    assert led.count("p") == 3
    assert led.best("p", 1)[0]["score"] == 3


def test_registry():
    reg = VerifierRegistry()
    reg.register(Const())
    assert "const" in reg and reg.get("const").klass == 2


def test_evolve_matches_best_fit_offline():
    handle = bin_packing.BinPackingHandle()
    bar = handle.verify(bin_packing.BEST_FIT).score
    events = []
    best = evolve(MutationProposer(seed=7), handle, Ledger(":memory:"),
                  EvolveConfig(problem="bp", generations=5, offspring=8),
                  on_new_best=events.append)
    assert events and best["score"] >= bar - 1e-9


def test_complexity_prefers_simpler_form():
    simple = complexity("def f(x, a, b):\n    return a*x + b", 2)["dl"]
    big = complexity("def f(x, a, b, c, d):\n    return a*x**3 + b*x**2 + c*x + d", 4)["dl"]
    assert simple < big


class ScriptedProposer:
    """Deterministic stand-in for an LLM: returns pre-written candidates."""

    def __init__(self, role, batches):
        self.role, self.batches, self.last_meta = role, list(batches), {}
        self.prompts = []

    def propose_raw(self, prompt, system, fn_name):
        self.prompts.append(prompt)
        self.last_meta = {"driver": "scripted", "role": self.role, "ok": True, "elapsed_s": 0}
        return self.batches.pop(0) if self.batches else []


def test_swarm_tracks_best_and_migrates():
    base3 = apfree.BASE3
    broken = "def construct(N):\n    return [0, 1, 2]"           # contains an AP
    isl0 = Island(0, ScriptedProposer("a", [[broken], []]))
    isl1 = Island(1, ScriptedProposer("b", [[base3], []]))
    best = evolve_swarm("apfree", apfree.ApFreeHandle(), Ledger(":memory:"),
                        seeds=[apfree.POWERS_OF_TWO], build_prompt=apfree.build_prompt,
                        system=apfree.SYS, fn_name="construct", islands=[isl0, isl1],
                        generations=2, per_gen=1, log=lambda *_: None)
    assert best["score"] == 105 and best["island"] == 1 and best["generation"] == 1
    assert any(not h["valid"] for h in best["history"])     # the broken one was recorded
    # island 1's best migrated to island 0 and was shown in its generation-2 prompt
    assert "size = 105" in isl0.proposer.prompts[1]


def test_swarm_stop_score():
    isl = Island(0, ScriptedProposer("a", [[apfree.BASE3], [apfree.GREEDY]]))
    best = evolve_swarm("apfree", apfree.ApFreeHandle(), Ledger(":memory:"), seeds=[],
                        build_prompt=apfree.build_prompt, system=apfree.SYS,
                        fn_name="construct", islands=[isl], generations=5, per_gen=1,
                        stop_score=100, log=lambda *_: None)
    assert best["generation"] == 1 and len(isl.proposer.prompts) == 1
