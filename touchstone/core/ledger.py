"""The memory of what's been tried: a keyed, append-mostly SQLite store.

Keys every candidate so nothing is re-proposed, records the verdict +
provenance so a result can be *proven* later, and tracks the running best
so the evolve loop has something to climb. Deterministic + append-mostly.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    cand_key   TEXT PRIMARY KEY,   -- content hash; dedup handle
    problem    TEXT NOT NULL,
    handle     TEXT NOT NULL,
    klass      INTEGER NOT NULL,
    genome     TEXT NOT NULL,      -- JSON: the candidate itself
    score      REAL,
    valid      INTEGER,
    detail     TEXT,               -- JSON: verdict provenance
    generation INTEGER,
    parent_key TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_problem_score ON candidates(problem, score DESC);
"""


def _key(problem: str, genome: Any) -> str:
    blob = json.dumps([problem, genome], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def seen(self, problem: str, genome: Any) -> bool:
        k = _key(problem, genome)
        cur = self.conn.execute("SELECT 1 FROM candidates WHERE cand_key=?", (k,))
        return cur.fetchone() is not None

    def record(self, problem: str, genome: Any, verdict, generation: int,
               parent_key: str | None = None) -> str:
        k = _key(problem, genome)
        self.conn.execute(
            "INSERT OR IGNORE INTO candidates "
            "(cand_key,problem,handle,klass,genome,score,valid,detail,"
            " generation,parent_key,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (k, problem, verdict.handle, verdict.klass, json.dumps(genome, default=str),
             verdict.score, int(verdict.valid), json.dumps(verdict.detail, default=str),
             generation, parent_key, time.time()),
        )
        self.conn.commit()
        return k

    def best(self, problem: str, n: int = 1) -> list[dict]:
        cur = self.conn.execute(
            "SELECT cand_key,genome,score,detail,generation FROM candidates "
            "WHERE problem=? AND valid=1 ORDER BY score DESC LIMIT ?", (problem, n))
        return [{"key": r[0], "genome": json.loads(r[1]), "score": r[2],
                 "detail": json.loads(r[3]), "generation": r[4]} for r in cur.fetchall()]

    def count(self, problem: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM candidates WHERE problem=?", (problem,))
        return cur.fetchone()[0]
