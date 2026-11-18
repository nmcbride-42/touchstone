# Online bin-packing priority heuristic for the synthetic trimodal "trap"
# distribution in touchstone/problems/binpack_trimodal.py (class 2).
# Mean excess bins over ceil(sum(items)), 20 instances x 120 items each:
#   train    9.87%  (best-fit 10.87%, first-fit 11.09%)
#   held-out 9.57%  (best-fit 10.35%, first-fit 10.33%)   [different seed]
# The proposer was TOLD the item modes (0.205 / 0.32 / 0.58) and the dead-zone
# mechanism; this heuristic is tuned to this distribution, not a general
# improvement over best-fit.
# Provenance (historical): 4-island swarm (2 Claude, 2 GPT islands),
# 2 generations x 2 candidates; this one came from a GPT island in generation 1.

def priority(item, remaining):
    import math

    slack = remaining - item
    if slack < -1e-12:
        return -1e12
    slack = max(0.0, slack)

    factorial = (1, 1, 2, 6, 24)
    completion_mass = 0.0

    for count in range(1, 5):
        if 0.16 * count > slack + 1e-12:
            continue
        for a in range(count + 1):
            for b in range(count - a + 1):
                c = count - a - b
                target = 0.205 * a + 0.32 * b + 0.58 * c
                variance = (
                    0.012 ** 2 * a
                    + 0.018 ** 2 * b
                    + 0.024 ** 2 * c
                )
                scale = math.sqrt(2.0 * variance)
                probability = 0.5 * (
                    math.erf((slack - target) / scale)
                    - math.erf((slack - 0.025 - target) / scale)
                )
                multiplicity = factorial[count] / (
                    factorial[a] * factorial[b] * factorial[c]
                )
                completion_mass += (
                    multiplicity
                    * probability
                    / (3.0 ** count)
                    * math.exp(-0.18 * (count - 1))
                )

    score = (
        3.0 * math.log1p(24.0 * completion_mass)
        - 2.4 * slack
        + 9.0 * math.exp(-(slack / 0.033) ** 2)
    )

    if 0.025 < slack < 0.16:
        severity = min(1.0, (slack - 0.025) / 0.035)
        score -= severity * (5.0 + 2.0 * (remaining > 0.68))

    return score
