"""Class-3 disposer on REAL data, genuinely-unknown form: discover a compact
analytic model of a phase-folded eclipsing-binary light curve.

There is no canonical closed form for an eclipsing-binary (EB) folded light
curve: it is a flat baseline punctuated by a deep primary eclipse (phase 0)
and a shallower secondary (phase 0.5). The proposer must DISCOVER a form
(e.g. baseline minus two localized dips). The disposer is held-out phase
points: a model that fits the training phases but fails unseen ones is
rejected.

Data: TESS 2-minute SPOC photometry of TIC 329089161 (a detached EB,
P ~ 2.537 d), fetched from MAST, phase-folded and binned to 140 points by
`touchstone.tools.fetch_lightcurve`, cached as data/lc_329089161.npz.
Phase is re-centered to [-0.25, 0.75] so BOTH eclipses are interior.

Honest scope: a parsimonious *empirical* model of one binned curve, scored by
held-out R² on a random 50/50 split of 140 bins. Not a physical (Roche
geometry) model, and adjacent bins are correlated, so held-out R² here is an
optimistic generalization estimate.
"""
from __future__ import annotations

import functools
import inspect
import warnings

import numpy as np
from scipy.optimize import curve_fit

from ..config import data_dir
from ..core.sandbox import compile_function
from ..core.verifier import VerificationHandle, Verdict

TIC = 329089161
SPLIT_SEED = 7


@functools.lru_cache(maxsize=4)
def _dataset(tic: int = TIC):
    """(phase, flux, train_idx, test_idx), loaded lazily and cached."""
    path = data_dir() / f"lc_{tic}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The repo ships data/lc_{TIC}.npz; set "
            "TOUCHSTONE_DATA_DIR or run touchstone.tools.fetch_lightcurve.")
    d = np.load(path)
    phase, flux = d["phase"].astype(float), d["flux"].astype(float)
    # re-center: primary at 0, secondary at 0.5, both interior in [-0.25, 0.75]
    phase = np.where(phase >= -0.25, phase, phase + 1.0)
    order = np.argsort(phase)
    phase, flux = phase[order], flux[order]
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(phase))
    n_tr = len(phase) // 2
    return phase, flux, np.sort(perm[:n_tr]), np.sort(perm[n_tr:])


def curve():
    ph, fl, _, _ = _dataset()
    return ph, fl


def _r2(y, yp):
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")


def _fit_score(f, n_params):
    ph, fl, tr, te = _dataset()
    Ptr, Ftr, Pte, Fte = ph[tr], fl[tr], ph[te], fl[te]

    def model(X, *p):
        return np.array([float(f(float(x), *p)) for x in X])

    if n_params == 0:
        popt = []
    else:
        popt, _ = curve_fit(model, Ptr, Ftr, p0=np.full(n_params, 0.1), maxfev=20000)
    return _r2(Fte, model(Pte, *popt)), [round(float(c), 5) for c in popt]


@functools.lru_cache(maxsize=1)
def baseline():
    """Best of {flat line, SINGLE primary dip}. The proposer must add the
    secondary eclipse to win."""
    flat, _ = _fit_score(lambda x, c: c, 1)
    one_dip, _ = _fit_score(lambda x, a, w, c: c - a * np.exp(-((x) / w) ** 2), 3)
    return max(flat, one_dip), {"flat": round(flat, 4),
                                "single_primary_dip": round(one_dip, 4)}


class LightCurveHandle(VerificationHandle):
    name = "eb-lightcurve"
    klass = 3

    def verify(self, candidate: str) -> Verdict:
        try:
            f = compile_function(candidate, "f")
            n_params = len(inspect.signature(f).parameters) - 1
            if n_params < 0 or n_params > 10:
                raise ValueError(f"bad arity: {n_params + 1} args")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r2, popt = _fit_score(f, n_params)
            if not np.isfinite(r2):
                raise ValueError("non-finite score")
            return Verdict(score=r2, valid=True, handle=self.name, klass=self.klass,
                           detail={"heldout_r2": r2, "n_params": n_params,
                                   "fitted_params": popt,
                                   "baseline_r2": round(baseline()[0], 4)})
        except Exception as e:
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)[:200]})


# ---- proposer glue -------------------------------------------------------
SYS = (
    "You are an astronomer modeling a phase-folded eclipsing-binary light "
    "curve. You write a compact analytic flux model f(phase, ...) in Python "
    "using only `math`, leaving numeric constants as fittable PARAMETERS. You "
    "recognize eclipse structure (flat baseline near 1.0, localized dips) and "
    "favor parsimonious smooth forms (Gaussian/sech² dips) over polynomials."
)


def _sample(k=22):
    ph, fl, _, _ = _dataset()
    step = max(1, len(ph) // k)
    return [(round(float(ph[i]), 3), round(float(fl[i]), 4))
            for i in range(0, len(ph), step)]


def build_prompt(examples, n):
    pts = "; ".join(f"({x},{y})" for x, y in _sample())
    lines = [
        "Model this phase-folded eclipsing-binary light curve: flux y vs "
        f"orbital phase x in [-0.25, 0.75]. Sample (x, y) points: {pts}",
        "",
        "Baseline flux is ~1.0. There is a DEEP primary eclipse near phase 0 "
        "and a SHALLOWER secondary eclipse near phase 0.5. Write "
        "`def f(x, ...):` (args after x are fittable constants). Use only "
        "`math`. Capture BOTH eclipses.",
        "Example shape (NOT the answer): "
        "`def f(x, c, a, w): return c - a*math.exp(-(x/w)**2)`  (this is only "
        "the primary — you must also model the secondary near 0.5).",
        "Real eclipses are FLAT-BOTTOMED (totality), not bell-shaped — a "
        "Gaussian rounds the bottom. Profiles worth trying: raised-cosine with "
        "a flat floor, super-Gaussian exp(-(x/w)**p) with p>2, trapezoid/"
        "sigmoid edges. The better the eclipse shape, the higher the R².",
        "You are scored by R² on HELD-OUT phases you cannot see.",
    ]
    if examples:
        lines += ["", "BEST MODELS SO FAR (held-out R²) — beat these:"]
        for e in examples:
            lines += [f"# R2 = {e['score']:+.4f}", "```python", e["code"].strip(), "```"]
    lines += ["", f"Output EXACTLY {n} DISTINCT `def f(x, ...):` models, each in "
              "its own ```python``` block. No prose outside the blocks."]
    return "\n".join(lines)


def seed_programs():
    return [
        "def f(x, c):\n    return c",                                           # flat
        "import math\ndef f(x, c, a, w):\n    return c - a*math.exp(-(x/w)**2)",  # primary only
    ]
