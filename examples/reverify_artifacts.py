"""Re-score every shipped artifact with its verifier. Offline, ~10 seconds.

Each artifact is verified in its own freshly spawned interpreter
(`touchstone.core.isolate.verify_isolated`), so no artifact can change the
verifier state another artifact is scored against.

Every number quoted in the README's results table comes from this script, not
from a log of the original run. If a result does not survive re-verification
it says so (exit code 1 if any artifact misses its expectation).

    python examples/reverify_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touchstone.core.isolate import verify_isolated
from touchstone.problems import apfree, binpack_trimodal, lightcurve, symbolic_regression

HANDLES = {
    "symreg": symbolic_regression.SymbolicRegressionHandle,
    "apfree": apfree.ApFreeHandle,
    "binpack_trimodal": binpack_trimodal.TrimodalBinPackingHandle,
    "lightcurve": lightcurve.LightCurveHandle,
}


def check(entry: dict) -> tuple[bool, object]:
    code = (ROOT / "artifacts" / entry["file"]).read_text()
    v = verify_isolated(HANDLES[entry["handle"]](), code)
    exp = entry["expect"]
    ok = v.valid == exp.get("valid", True)
    if "score_eq" in exp:
        ok &= v.score == exp["score_eq"]
    if "score_min" in exp:
        ok &= v.score >= exp["score_min"]
    if "score_max" in exp:
        ok &= v.score <= exp["score_max"]
    for key in exp.get("detail_true", []):
        ok &= bool(v.detail.get(key))
    return bool(ok), v


def main() -> int:
    manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text())
    failures = 0
    for entry in manifest["artifacts"]:
        ok, v = check(entry)
        failures += not ok
        print(f"[{'OK ' if ok else 'FAIL'}] {entry['file']:<34} score={v.score:+.4f}  "
              f"{entry['claim']}")
        shown = {k: v.detail[k] for k in v.detail if k not in ("fitted_params", "params")}
        print(f"        {shown}")
    print(f"\n{len(manifest['artifacts']) - failures}/{len(manifest['artifacts'])} "
          "artifacts reproduce their stated result")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
