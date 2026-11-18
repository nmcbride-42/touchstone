"""Parallel island swarm with ring migration, generic over proposers.

Each island owns a proposer (any object with `propose_raw(prompt, system,
fn_name)` and `last_meta`) and, usually, a strategy hint. Islands can mix model
families (e.g. one Anthropic island and one OpenAI island), which is the point:
different lineages get stuck in different places. Proposals for one
generation are requested from all islands in parallel threads; every
candidate is then verified sequentially in the main thread (the sandbox's
alarm-based time limit needs the main thread). After each generation, each
island's best is migrated to the next island in the ring.

What it provides:
  * parallel proposal calls, one thread per island;
  * a per-island namespace in the shared ledger (content-hash dedup and
    provenance per island, plus a combined problem-wide view);
  * an optional stop score that ends the run after the generation in which
    the best score reaches it;
  * a full history of every candidate and its verdict, for provenance.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ledger import Ledger
from ..core.verifier import VerificationHandle


@dataclass
class Island:
    idx: int
    proposer: Any
    migrants: list[dict] = field(default_factory=list)


def make_islands(specs: list[tuple[Any, str, str]]) -> list[Island]:
    """specs: [(LLMClient, role_name, strategy_hint), ...] -> CodeProposer islands."""
    from .llm import CodeProposer
    return [Island(idx=i, proposer=CodeProposer(client, role=role, role_hint=hint))
            for i, (client, role, hint) in enumerate(specs)]


def evolve_swarm(
    problem: str,
    handle: VerificationHandle,
    ledger: Ledger,
    seeds: list[str],
    build_prompt: Callable[[list[dict], int], str],
    system: str,
    fn_name: str,
    islands: list[Island],
    generations: int = 3,
    per_gen: int = 3,
    stop_score: float | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    n_islands = len(islands)
    best: dict = {"score": float("-inf"), "code": None, "detail": {}, "island": None,
                  "driver": None, "role": None, "generation": 0}
    history: list[dict] = []

    def eval_and_record(isl: Island, code: str, gen: int, source_tag: str) -> None:
        isl_prob = f"{problem}__isl{isl.idx}"
        if ledger.seen(isl_prob, code):
            return
        v = handle.verify(code)
        ledger.record(isl_prob, code, v, gen)
        if not ledger.seen(problem, code):
            ledger.record(problem, code, v, gen)
        role = getattr(isl.proposer, "role", f"island_{isl.idx}")
        history.append({"generation": gen, "island": isl.idx, "driver": source_tag,
                        "role": role, "valid": v.valid,
                        "score": float(v.score) if v.valid else None,
                        "detail": v.detail, "code": code})
        if v.valid and v.score > best["score"]:
            best.update(score=float(v.score), code=code, detail=v.detail, island=isl.idx,
                        driver=source_tag, role=role, generation=gen)
            log(f"  [NEW BEST] gen={gen} isl={isl.idx} ({role}) score={v.score:+.4f}")
        elif v.valid:
            log(f"    gen={gen} isl={isl.idx} ({role}) score={v.score:+.4f}")
        else:
            log(f"    gen={gen} isl={isl.idx} ({role}) invalid: "
                f"{str(v.detail.get('error', ''))[:60]}")

    for isl in islands:                                   # generation 0: seeds
        for code in seeds:
            eval_and_record(isl, code, 0, "seed")

    for gen in range(1, generations + 1):
        log(f"\n--- generation {gen}/{generations} ({n_islands} islands) ---")
        prompts = []
        for isl in islands:
            local_top = [{"code": e["genome"], "score": e["score"]}
                         for e in ledger.best(f"{problem}__isl{isl.idx}", 2)]
            seen = {e["code"] for e in local_top}
            examples = local_top + [m for m in isl.migrants if m["code"] not in seen]
            prompts.append((isl, build_prompt(examples, per_gen)))

        results: list[tuple[Island, list[str], dict]] = []
        with cf.ThreadPoolExecutor(max_workers=n_islands) as ex:
            futs = {ex.submit(isl.proposer.propose_raw, prm, system, fn_name): isl
                    for isl, prm in prompts}
            for fut in cf.as_completed(futs):
                isl = futs[fut]
                results.append((isl, fut.result(), dict(isl.proposer.last_meta)))

        for isl, codes, meta in sorted(results, key=lambda r: r[0].idx):
            log(f"  island {isl.idx} ({meta.get('driver')}/{meta.get('role')}): "
                f"{len(codes)} candidates in {meta.get('elapsed_s')}s"
                + ("" if meta.get("ok", True) else f"  error={meta.get('error')}"))
            for code in codes:
                eval_and_record(isl, code, gen, str(meta.get("driver", "llm")))

        if stop_score is not None and best["score"] >= stop_score:
            log(f"reached stop_score={stop_score} at generation {gen}")
            break

        tops = [ledger.best(f"{problem}__isl{isl.idx}", 1) for isl in islands]
        for i, top in enumerate(tops):                    # ring migration
            if top:
                dst = islands[(i + 1) % n_islands]
                dst.migrants = [{"code": top[0]["genome"], "score": top[0]["score"]}]

    best["history"] = history
    return best
