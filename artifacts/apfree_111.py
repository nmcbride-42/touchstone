# 3-term-AP-free subset of {0..999} (class 1, EXACT verification), size 111.
# Strictly larger than the base-3 construction (105). Strategy: exact small
# optima by DFS for m <= 22, a block recurrence to build a size-N set, greedy
# completion over several orderings, then a deterministic remove-and-refill
# local search; it re-checks its own output before returning.
# NOT a record claim: we did not compare against the best-known lower bounds
# for r_3(1000) from dedicated computational searches.
# Provenance (historical): 4-island swarm (2 Claude islands, 2 GPT islands),
# 2 generations x 2 candidates. The GPT islands returned no parseable code in
# this run; all 8 LLM-written candidates came from the Claude islands.

def construct(N):
    # Block recurrence with exact small optima, then multi-order greedy, then swap local search
    M = min(N, 22)
    best = [[]]
    for m in range(1, M + 1):
        r = [len(b) for b in best]
        need = r[m - 1] + 1
        found = [None]

        def ok(y, T):
            for t in T:
                if (2 * y - t) in T or (2 * t - y) in T:
                    return False
                if (y + t) % 2 == 0 and (y + t) // 2 in T:
                    return False
            return True

        def dfs(x, T, lst):
            if found[0] is not None:
                return
            if len(lst) >= need:
                found[0] = sorted(lst)
                return
            if x >= m - 1:
                return
            if len(lst) + r[m - 1 - x] < need:
                return
            if ok(x, T):
                T.add(x); lst.append(x)
                dfs(x + 1, T, lst)
                T.discard(x); lst.pop()
            dfs(x + 1, T, lst)

        dfs(0, {m - 1}, [m - 1])
        best.append(found[0] if found[0] is not None else best[m - 1])

    gs = [0] * (N + 1)
    ch = [None] * (N + 1)
    for n in range(1, N + 1):
        if n <= M:
            gs[n] = len(best[n]); ch[n] = (0,)
            continue
        gs[n] = gs[n - 1]; ch[n] = (1,)
        for a in range(1, n):
            bb = -1
            b1 = min(a, n - 2 * a + 1)
            if b1 >= 1:
                bb = b1
            b2 = (n - a + 1) // 2
            if b2 > a and b2 > bb:
                bb = b2
            if bb < 1:
                continue
            v = gs[a] + gs[bb]
            if v > gs[n]:
                gs[n] = v; ch[n] = (2, a, bb)

    memo = {}
    def build(n):
        if n == 0:
            return []
        if n in memo:
            return memo[n]
        c = ch[n]
        if c[0] == 0:
            res = list(best[n])
        elif c[0] == 1:
            res = build(n - 1)
        else:
            a, b = c[1], c[2]
            sh = max(2 * a - 1, a + b - 1)
            res = build(a) + [y + sh for y in build(b)]
        memo[n] = res
        return res

    def greedy(seed, order):
        inS = [False] * N; forb = [False] * N; L = []
        def add(x):
            for s in L:
                k = 2 * x - s
                if 0 <= k < N: forb[k] = True
                k = 2 * s - x
                if 0 <= k < N: forb[k] = True
                if (x + s) % 2 == 0: forb[(x + s) // 2] = True
            L.append(x); inS[x] = True
        for x in seed:
            if 0 <= x < N and not inS[x] and not forb[x]:
                add(x)
        for x in order:
            if not inS[x] and not forb[x]:
                add(x)
        return L

    def h(x, c):
        return ((x * 6364136223846793005 + c * 1442695040888963407) >> 17) & 0xFFFFFFFF

    base = build(N)
    seeds = [base, [N - 1 - x for x in base]]
    orders = [list(range(N)), list(range(N - 1, -1, -1)),
              sorted(range(N), key=lambda x: abs(2 * x - (N - 1))),
              sorted(range(N), key=lambda x: -abs(2 * x - (N - 1)))]
    cands = []
    for sd in seeds:
        for od in orders:
            cands.append(greedy(sd, od))
    start = max(cands, key=len)

    # local search with conflict counts
    cnt = [0] * N
    inS = [False] * N
    mem = []

    def add(x):
        for s in mem:
            k = 2 * x - s
            if 0 <= k < N: cnt[k] += 1
            k = 2 * s - x
            if 0 <= k < N: cnt[k] += 1
            t = x + s
            if not (t & 1): cnt[t >> 1] += 1
        mem.append(x); inS[x] = True

    def rem(x, out):
        mem.remove(x); inS[x] = False
        for s in mem:
            k = 2 * x - s
            if 0 <= k < N:
                cnt[k] -= 1
                if cnt[k] == 0: out.append(k)
            k = 2 * s - x
            if 0 <= k < N:
                cnt[k] -= 1
                if cnt[k] == 0: out.append(k)
            t = x + s
            if not (t & 1):
                k = t >> 1
                cnt[k] -= 1
                if cnt[k] == 0: out.append(k)

    for x in start:
        add(x)
    bestS = list(mem)
    state = 88172645463325252
    for it in range(4000):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        r = state >> 20
        freed = []
        removed = []
        k = 2 if (r % 9 == 0) else 1
        for j in range(k):
            if not mem: break
            e = mem[(r >> (7 * j)) % len(mem)]
            removed.append(e)
            rem(e, freed)
        freed.sort(key=lambda y: h(y, it))
        added = 0
        for y in freed:
            if not inS[y] and cnt[y] == 0 and y not in removed:
                add(y); added += 1
        for e in removed:
            if not inS[e] and cnt[e] == 0 and added < len(removed):
                add(e); added += 1
        if len(mem) > len(bestS):
            bestS = list(mem)
        elif len(mem) < len(bestS) - 2:
            for x in list(mem):
                rem(x, [])
            for x in bestS:
                add(x)

    def valid(S):
        T = set(S)
        if len(T) != len(S): return False
        Ls = sorted(S)
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                t = Ls[i] + Ls[j]
                if not (t & 1) and (t >> 1) in T:
                    return False
        return True

    if all(0 <= x < N for x in bestS) and valid(bestS):
        return sorted(bestS)
    return sorted(start)
