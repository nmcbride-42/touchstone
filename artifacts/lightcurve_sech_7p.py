# Analytic model of the folded TESS light curve of eclipsing binary
# TIC 329089161: two flat-floored sech² eclipse profiles, 7 fitted parameters.
# Historical: on an earlier fetch of the curve it scored held-out R² 0.970 and
# beat the 4-parameter Gaussian model (0.958). On the curve shipped in data/
# (re-fetched with more TESS sectors) the ranking REVERSES: the 4-parameter
# model scores higher. Kept as an honest example of a result that did not
# survive re-verification on new data.
# Provenance (historical): 2-island loop (single model family), 3 generations.

def f(x, c, a1, w1, f1, a2, w2, f2):
    def sech_flat(dx, amp, width, flat):
        z = abs(dx) / width
        if z > 1.0:
            return 0.0
        if z <= flat:
            return amp
        z_norm = (z - flat) / (1.0 - flat)
        cosh_z = math.cosh(math.pi * z_norm)
        return amp / (cosh_z * cosh_z)
    
    return c - sech_flat(x, a1, w1, f1) - sech_flat(x - 0.5, a2, w2, f2)
