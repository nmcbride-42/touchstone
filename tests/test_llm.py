import sys

import pytest

from touchstone.proposers.llm import (CodeProposer, CommandClient, client_from_env,
                                      extract_functions, parse_code_blocks)

REPLY = """Here you go.
```python
def f(x, a):
    return a * x
```
```
def f(x, a, b):
    return a * x + b
```
```python
def g(x):
    return x
```
"""


def test_parse_and_extract():
    assert len(parse_code_blocks(REPLY)) == 3
    assert len(extract_functions(REPLY, "f")) == 2
    assert extract_functions("def f(x):\n    return x", "f") == ["def f(x):\n    return x"]
    assert extract_functions("no code", "f") == []


class FakeClient:
    name = "fake"

    def __init__(self, reply=REPLY, exc=None):
        self.reply, self.exc, self.seen = reply, exc, []

    def complete(self, prompt, system):
        self.seen.append(system)
        if self.exc:
            raise self.exc
        return self.reply


def test_code_proposer_adds_strategy_hint():
    c = FakeClient()
    p = CodeProposer(c, role="isl0", role_hint="be bold")
    assert len(p.propose_raw("prompt", "sys", fn_name="f")) == 2
    assert "ISLAND STRATEGY (isl0): be bold" in c.seen[0]
    assert p.last_meta["ok"] and p.last_meta["n_extracted"] == 2


def test_code_proposer_never_raises():
    p = CodeProposer(FakeClient(exc=TimeoutError("slow")))
    assert p.propose_raw("p", "s", "f") == []
    assert p.last_meta["ok"] is False and "slow" in p.last_meta["error"]


def test_command_client_round_trip():
    script = "import sys; sys.stdin.read(); print('```python\\ndef f(x):\\n    return x\\n```')"
    c = CommandClient(command=f'"{sys.executable}" -c "{script}"')
    assert extract_functions(c.complete("p", "s"), "f") == ["def f(x):\n    return x"]


def test_unknown_provider(monkeypatch):
    with pytest.raises(ValueError):
        client_from_env("nope")


def test_command_provider_requires_command(monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_LLM_COMMAND", raising=False)
    with pytest.raises(ValueError):
        client_from_env("command")


@pytest.mark.network
def test_live_provider_writes_a_verifiable_candidate():
    from touchstone.problems import apfree
    p = CodeProposer(client_from_env())
    codes = p.propose_raw(apfree.build_prompt([], 1), apfree.SYS, fn_name="construct")
    assert p.last_meta["ok"], p.last_meta
    assert codes and apfree.ApFreeHandle().verify(codes[0]).detail
