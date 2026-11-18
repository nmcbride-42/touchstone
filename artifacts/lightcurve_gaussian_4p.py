# Analytic model of the folded TESS light curve of eclipsing binary
# TIC 329089161 (class 3, held-out R² on 70 of 140 phase bins).
# Two Gaussian dips with a shared width and a cross-term, 4 fitted parameters.
# An empirical shape model, not Roche-geometry physics.
# Provenance (historical): proposed by an LLM against an earlier fetch of this
# star's light curve. Re-verified on the curve shipped in data/ (see manifest).

import math
def f(x, c, a1, a2, w):
    g1 = math.exp(-(x/w)**2)
    g2 = math.exp(-((x-0.5)/w)**2)
    return c - a1*g1 - a2*g2 - a1*a2*(g1*g2)
