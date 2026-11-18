import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from reverify_artifacts import check  # noqa: E402

ENTRIES = json.loads((ROOT / "artifacts" / "manifest.json").read_text())["artifacts"]


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["file"] for e in ENTRIES])
def test_artifact_reproduces_stated_result(entry):
    ok, verdict = check(entry)
    assert ok, (entry["claim"], verdict.score, verdict.detail)


def test_every_artifact_is_listed():
    files = {p.name for p in (ROOT / "artifacts").glob("*.py")}
    assert files == {e["file"] for e in ENTRIES}
