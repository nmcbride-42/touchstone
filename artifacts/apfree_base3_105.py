# 3-term-AP-free subset of {0..999} (class 1, EXACT verification).
# LLM-written constructor; the verifier checks every pair exhaustively, so
# "valid" means proven AP-free, not "probably". Size 105: this is the classic
# base-3 construction (subset sums of powers of 3), i.e. the LLM rediscovered a
# known strong construction starting from a powers-of-two seed (size 10).
# Provenance (historical): Claude Haiku 4.5, a few batched calls, 2026.

def construct(N):
    result = []
    powers = []
    p = 1
    while p < N:
        powers.append(p)
        p *= 3
    
    for mask in range(1 << len(powers)):
        s = 0
        for i in range(len(powers)):
            if mask & (1 << i):
                s += powers[i]
        if s < N:
            result.append(s)
    
    return result
