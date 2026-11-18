"""Class-3 disposer: symbolic regression with held-out verification.

The candidate is an LLM-written functional FORM `f(x, *params)`. We fit the
params on TRAIN data (scipy least-squares) and score on HELD-OUT data (R²).
The held-out split is the disposer: a form that fits train but fails held-out
is rejected. This is the LLM-SR design, and it's the cleanest fast win — a
linear/polynomial baseline is hopeless on a genuinely nonlinear law, and the
LLM recognizes the form (a damped oscillation) instantly.

Ground truth here is *held-out predictive accuracy* — class 3. Caveat the
research flagged: good fit ≠ true law. A high held-out R² says the form
predicts unseen data, not that it's THE generating law. For a demonstrator
that's exactly the bar we want; for a real claim we'd add parsimony +
out-of-RANGE extrapolation tests.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
from scipy.optimize import curve_fit

from ..core.sandbox import compile_function, UnsafeCode
from ..core.verifier import VerificationHandle, Verdict

# --- the hidden generating law: damped oscillation (+ noise) ---------------
# Linear and even cubic fits are hopeless across ~2 periods. The LLM must
# recognize A*exp(-b x)*cos(w x + p) to win. Deterministic (seeded).
_A, _B, _W, _P = 4.0, 0.40, 2.0, 0.6


def _generate(n=80, seed=20260618):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, n)
    y = _A * np.exp(-_B * x) * np.cos(_W * x + _P)
    y = y + rng.normal(0, 0.10, size=n)          # observation noise
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]          # 50/50 train / held-out
    return x[tr], y[tr], x[te], y[te]


_XTR, _YTR, _XTE, _YTE = _generate()


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")


def _poly_baseline(deg):
    coeffs = np.polyfit(_XTR, _YTR, deg)
    return _r2(_YTE, np.polyval(coeffs, _XTE))


# The bar: best of linear and cubic held-out R². Beat THIS, not just a constant.
BASELINE_R2 = max(_poly_baseline(1), _poly_baseline(3))


# --- Phase 3: range-based splits for extrapolation testing -----------------
# Fit on the INTERIOR range only; test interpolation on held-out interior
# points AND extrapolation on OUT-OF-RANGE points. Overfit forms ace
# interpolation and explode on extrapolation — that's the whole point.
def _generate_ranges(n=90, seed=20260618):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, n)
    y = _A * np.exp(-_B * x) * np.cos(_W * x + _P) + rng.normal(0, 0.10, n)
    interior = x <= 4.5
    xi, yi = x[interior], y[interior]
    perm = rng.permutation(len(xi))
    half = len(xi) // 2
    tr, it = np.sort(perm[:half]), np.sort(perm[half:])
    return (xi[tr], yi[tr]), (xi[it], yi[it]), (x[~interior], y[~interior])


(_RX_TR, _RY_TR), (_RX_INT, _RY_INT), (_RX_EXT, _RY_EXT) = _generate_ranges()

_EXTRAP_GATE = 0.5      # extrap R² below this => the form does not capture the law
_LAMBDA = 0.01          # parsimony weight on description length


def _fit_on(xtr, ytr, f, n_params):
    def model(X, *p):
        return np.array([float(f(float(xi), *p)) for xi in X])
    if n_params == 0:
        return model, []
    popt, _ = curve_fit(model, xtr, ytr, p0=np.ones(n_params), maxfev=8000)
    return model, popt


class RobustSymRegHandle(VerificationHandle):
    """Phase-3 composite disposer: interpolation + EXTRAPOLATION + parsimony.

    A high score now means 'probably the real structure', not merely 'fits
    held-out interior'. This is the symbolic-model analog of the gauntlet:
    several independent checks composed into one verdict. The `accepted` flag
    gates on out-of-range generalization; the score is parsimony-adjusted so
    the loop climbs toward the simplest form that truly generalizes."""
    name = "symreg-robust"
    klass = 3

    def verify(self, candidate: str) -> Verdict:
        from ..core.complexity import complexity
        try:
            f = compile_function(candidate, "f")
            n_params = len(inspect.signature(f).parameters) - 1
            if n_params < 0 or n_params > 12:
                raise ValueError(f"bad arity: {n_params + 1}")
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                model, popt = _fit_on(_RX_TR, _RY_TR, f, n_params)
            interp = _r2(_RY_INT, model(_RX_INT, *popt))
            extrap = _r2(_RY_EXT, model(_RX_EXT, *popt))
            cx = complexity(candidate, n_params)
            if not (np.isfinite(interp) and np.isfinite(extrap)):
                raise ValueError("non-finite score")
            extrap_c = float(np.clip(extrap, -1.0, 1.0))
            composite = 0.5 * interp + 0.5 * extrap_c - _LAMBDA * cx["dl"]
            accepted = (extrap > _EXTRAP_GATE) and (interp > 0.8)
            return Verdict(score=composite, valid=True, handle=self.name, klass=self.klass,
                           detail={"interp_r2": round(interp, 4),
                                   "extrap_r2": round(extrap, 4),
                                   "complexity_dl": cx["dl"], "n_params": n_params,
                                   "accepted": accepted,
                                   "naive_would_pass": interp > 0.8,
                                   "params": [round(float(c), 4) for c in popt]})
        except Exception as e:
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)[:200]})


SYS = (
    "You are a physicist doing symbolic regression. Given sample (x, y) points "
    "you infer the underlying functional FORM, expressed as Python using only "
    "the `math` module. You leave numeric constants as fittable PARAMETERS "
    "(extra function args) rather than hard-coding them — a downstream "
    "least-squares fitter sets them. You favor compact, physically-plausible "
    "forms (decay, oscillation, growth, power laws) over high-degree polynomials."
)


def sample_points(k: int = 14) -> list[tuple[float, float]]:
    """A readable sample of TRAIN points to show the model (never held-out)."""
    order = np.argsort(_XTR)[:: max(1, len(_XTR) // k)]
    return [(round(float(_XTR[i]), 3), round(float(_YTR[i]), 3)) for i in order]


def build_prompt(examples: list[dict], n: int) -> str:
    pts = ", ".join(f"({x}, {y})" for x, y in sample_points())
    lines = [
        "SYMBOLIC REGRESSION. Infer the function y = f(x) generating this data.",
        f"Sample training points (x, y): {pts}",
        "",
        "Write `def f(x, ...):` where x is a scalar float and the remaining "
        "args are constants to be fitted by least-squares (don't hard-code "
        "numbers — make them parameters). Use only the `math` module.",
        "Example shape (NOT the answer): "
        "`def f(x, a, b): return a*math.exp(b*x)`",
        "",
        "The data is noisy; look for the underlying smooth structure (it is "
        "not a polynomial). You are scored by R² on HELD-OUT points you "
        "cannot see, so a form that merely interpolates these points will lose.",
    ]
    if examples:
        lines += ["", "BEST FORMS SO FAR (held-out R²; higher better) — beat these:"]
        for e in examples:
            lines += [f"# R2 = {e['score']:+.4f}", "```python", e["code"].strip(), "```"]
    lines += [
        "",
        f"Output EXACTLY {n} DISTINCT candidate forms, each a complete "
        "`def f(x, ...):` in its own ```python``` block. Try genuinely "
        "different functional families. No prose outside the code blocks.",
    ]
    return "\n".join(lines)


def seed_programs() -> list[str]:
    """Grounded, valid, deliberately-WEAK seeds so the population starts but
    there's clear room to climb."""
    return [
        "import math\ndef f(x, a, b):\n    return a*x + b",            # linear
        "import math\ndef f(x, a, b):\n    return a*math.exp(-b*x)",   # plain decay
    ]


class SymbolicRegressionHandle(VerificationHandle):
    name = "symreg-damped"
    klass = 3

    def verify(self, candidate: str) -> Verdict:
        try:
            f = compile_function(candidate, "f")
            n_params = len(inspect.signature(f).parameters) - 1   # minus x
            if n_params < 0 or n_params > 8:
                raise ValueError(f"bad arity: {n_params + 1} args")

            # pointwise -> vectorized wrapper (LLM writes scalar math)
            def model(X, *p):
                return np.array([float(f(float(xi), *p)) for xi in X])

            p0 = np.ones(n_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if n_params == 0:
                    popt = []
                else:
                    popt, _ = curve_fit(model, _XTR, _YTR, p0=p0, maxfev=4000)
            pred = model(_XTE, *popt)
            if not np.all(np.isfinite(pred)):
                raise ValueError("non-finite predictions")
            r2 = _r2(_YTE, pred)
            return Verdict(score=r2, valid=True, handle=self.name, klass=self.klass,
                           detail={"heldout_r2": r2, "n_params": n_params,
                                   "params": [round(float(c), 4) for c in popt],
                                   "baseline_r2": round(BASELINE_R2, 4)})
        except Exception as e:
            return Verdict(score=float("-inf"), valid=False, handle=self.name,
                           klass=self.klass, detail={"error": repr(e)[:200]})
