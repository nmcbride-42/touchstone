import math

from touchstone.problems import (apfree, bin_packing, binpack_trimodal, lightcurve,
                                 symbolic_regression as sr)


# --- class 1: exact -----------------------------------------------------------
def test_apfree_rejects_set_with_progression_and_names_witness():
    v = apfree.ApFreeHandle(n=100).verify("def construct(N):\n    return [0, 5, 10]")
    assert not v.valid and "(0, 5, 10)" in v.detail["error"]


def test_apfree_rejects_out_of_range_and_non_integers():
    h = apfree.ApFreeHandle(n=10)
    assert not h.verify("def construct(N):\n    return [0, 10]").valid
    assert not h.verify("def construct(N):\n    return [0.5]").valid


def test_apfree_reference_constructions():
    h = apfree.ApFreeHandle()
    assert h.verify(apfree.POWERS_OF_TWO).score == 10
    assert h.verify(apfree.BASE3).score == 105
    assert h.verify(apfree.GREEDY).score == 105   # Stanley sequence == base-3 set


def test_apfree_checker_matches_brute_force():
    import itertools
    import random
    rng = random.Random(0)
    for _ in range(200):
        s = set(rng.sample(range(30), rng.randint(0, 8)))
        brute = not any(a + c == 2 * b for a, b, c in itertools.permutations(s, 3))
        assert apfree.is_ap_free(s)[0] == brute


def test_apfree_constructor_timeout_is_invalid():
    v = apfree.ApFreeHandle().verify("def construct(N):\n    while True:\n        pass")
    assert not v.valid and "CodeTimeout" in v.detail["error"]


# --- class 2: simulation scorers -----------------------------------------------
def test_param_bin_packing_baselines_ordered():
    h = bin_packing.BinPackingHandle()
    bf = h.verify(bin_packing.BEST_FIT).score
    ff = h.verify(bin_packing.FIRST_FIT).score
    wf = h.verify(bin_packing.WORST_FIT).score
    assert bf >= ff > wf


def test_code_bin_packing_agrees_with_param_version():
    code_bf = bin_packing.CodeBinPackingHandle().verify(bin_packing.seed_programs()[0]).score
    param_bf = bin_packing.BinPackingHandle().verify(bin_packing.BEST_FIT).score
    assert math.isclose(code_bf, param_bf)


def test_trimodal_baselines_and_detail():
    h = binpack_trimodal.TrimodalBinPackingHandle()
    bf = h.verify(binpack_trimodal.BEST_FIT)
    ff = h.verify(binpack_trimodal.FIRST_FIT)
    assert bf.valid and ff.valid
    assert bf.detail["beats_best_fit_train"] is False
    assert math.isclose(-bf.score, binpack_trimodal.baselines()["bf_train"])
    assert math.isclose(-ff.score, binpack_trimodal.baselines()["ff_train"])


def test_trimodal_rejects_nan_priority():
    v = binpack_trimodal.TrimodalBinPackingHandle().verify(
        "def priority(item, remaining):\n    return float('nan')")
    assert not v.valid


def test_trimodal_instances_are_deterministic():
    a = binpack_trimodal.make_instances(2, 10, 1)
    b = binpack_trimodal.make_instances(2, 10, 1)
    assert all((x == y).all() for x, y in zip(a, b))


# --- class 3: held-out fit ----------------------------------------------------
def test_symreg_true_form_beats_polynomial_baseline():
    true_form = ("import math\ndef f(x, a, b, w, p):\n"
                 "    return a*math.exp(-b*x)*math.cos(w*x+p)")
    v = sr.SymbolicRegressionHandle().verify(true_form)
    assert v.valid and v.score > 0.95 > sr.BASELINE_R2


def test_robust_verifier_rejects_overfit_accepts_truth():
    h = sr.RobustSymRegHandle()
    poly = ("def f(x, a, b, c, d, e, g, h, i, j, k):\n"
            "    return a*x**9+b*x**8+c*x**7+d*x**6+e*x**5+g*x**4+h*x**3+i*x**2+j*x+k")
    true_form = ("import math\ndef f(x, a, b, w, p):\n"
                 "    return a*math.exp(-b*x)*math.cos(w*x+p)")
    over, truth = h.verify(poly), h.verify(true_form)
    assert over.detail["naive_would_pass"] and not over.detail["accepted"]
    assert truth.detail["accepted"] and truth.score > over.score


def test_lightcurve_seeds_score_and_baseline():
    h = lightcurve.LightCurveHandle()
    flat, one_dip = (h.verify(c) for c in lightcurve.seed_programs())
    assert flat.valid and one_dip.valid
    assert one_dip.score > flat.score
    assert math.isclose(one_dip.score, lightcurve.baseline()[0], abs_tol=0.02)


def test_lightcurve_rejects_bad_arity():
    args = ", ".join(f"p{i}" for i in range(11))
    v = lightcurve.LightCurveHandle().verify(f"def f(x, {args}):\n    return p0")
    assert not v.valid
