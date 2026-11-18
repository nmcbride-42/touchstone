# Symbolic regression (class 3, held-out R²): LLM-proposed functional FORM.
# Target: a hidden damped oscillation 4.0*exp(-0.4x)*cos(2x+0.6) + N(0, 0.1) noise,
# 80 points on [0, 6], random 50/50 train/held-out split (seeded).
# The LLM proposed the form from 14 training points; scipy fit the constants on
# train; held-out R² is the score. sin(cx+d) is the same family as cos(wx+p).
# Re-verified score: see artifacts/manifest.json and examples/reverify_artifacts.py.
# Provenance (historical): a Claude model, won in the first generation.

import math
def f(x, a, b, c, d):
    return a * math.exp(-b * x) * math.sin(c * x + d)
