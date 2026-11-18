"""Run LLM-written code against ground truth without letting it run wild.

Candidates are small Python functions an LLM writes (`priority(item,
remaining)`, `construct(N)`, `f(x, *params)`, ...). Before anything runs, the
source passes two static screens:

  * a token screen (no `__`, `eval(`, `exec(`, `open(`, `getattr`, ...), and
  * an AST screen: no attribute whose name starts with `_`; no frame, code,
    generator, coroutine or traceback attributes (`gi_frame`, `f_back`,
    `f_globals`, `f_builtins`, `co_*`, `tb_*`, ...); no `str.format` /
    `format_map` (they resolve attribute paths from a runtime string); no
    attribute assignment or deletion (so `math.floor = ...` cannot patch a
    module the verifier also uses); no async code.

The code is then executed in a restricted namespace (whitelisted builtins with
no `open`, `type`, `object`, `getattr`, `vars`, `print` or `next`; only `math`
importable) and callers run it under a wall-clock alarm. Anything malformed,
slow, or unsafe resolves to an invalid candidate (score -inf); it never
crashes the loop.

Threat model, stated plainly: the screens close every escape route we know of
from this namespace (the frame-walk and verifier-patching attacks are pinned
by regression tests), but in-process restriction of CPython is not a proven
security boundary. It does not cap memory, and `time_limit` uses SIGALRM, so it
only works in the main thread on Unix. For a fresh interpreter per candidate,
so that one candidate cannot leave state behind for the next, use
`touchstone.core.isolate.verify_isolated` (the artifact re-verifier does). For
untrusted proposers, also run inside a container or VM with resource limits.
"""
from __future__ import annotations

import ast
import math
import signal
from contextlib import contextmanager
from typing import Callable

# Cheap first pass over the raw source. Note: we do NOT blanket-ban the
# substring "import": `import math` is legitimate and models reach for it.
# Imports are gated by a whitelisted __import__, so `import os` fails at exec
# time.
_FORBIDDEN = ("__", "open(", "eval(", "exec(", "compile(",
              "globals(", "locals(", "getattr", "setattr", "delattr",
              "input(", "exit(", "quit(")

# Attribute names that expose interpreter internals. Frames lead to the real
# builtins (f_builtins) and to the verifier's own module globals (f_globals,
# via f_back); code objects and tracebacks lead to frames.
_FORBIDDEN_ATTRS = frozenset({
    "gi_frame", "gi_code", "gi_yieldfrom",
    "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code", "ag_await",
    "tb_frame", "tb_next",
    "f_back", "f_globals", "f_builtins", "f_locals", "f_code", "f_trace",
    "format", "format_map", "mro",
})
_FORBIDDEN_ATTR_PREFIXES = ("_", "co_", "f_", "gi_", "cr_", "ag_", "tb_")

_ASYNC_NODES = (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)


def _screen(code: str) -> None:
    """Static checks. Raises UnsafeCode (or SyntaxError) before any exec."""
    for tok in _FORBIDDEN:
        if tok in code:
            raise UnsafeCode(f"forbidden token: {tok!r}")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _FORBIDDEN_ATTRS or attr.startswith(_FORBIDDEN_ATTR_PREFIXES):
                raise UnsafeCode(f"forbidden attribute: .{attr}")
            if not isinstance(node.ctx, ast.Load):
                raise UnsafeCode(f"attribute assignment/deletion not allowed: .{attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCode(f"forbidden name: {node.id}")
        elif isinstance(node, _ASYNC_NODES):
            raise UnsafeCode("async code not allowed")


_IMPORT_WHITELIST = {"math"}


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] in _IMPORT_WHITELIST:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import of {name!r} not allowed")


# Builtins a numeric heuristic legitimately needs — and nothing that touches
# the filesystem, network, or code execution.
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "pow": pow,
    "round": round, "len": len, "range": range, "float": float, "int": int,
    "bool": bool, "sorted": sorted, "map": map, "filter": filter, "zip": zip,
    "enumerate": enumerate, "all": all, "any": any, "list": list, "tuple": tuple,
    "set": set, "frozenset": frozenset, "dict": dict, "reversed": reversed,
    "divmod": divmod, "True": True, "False": False, "None": None,
    "__import__": _safe_import,   # gated: only `math`
}


class UnsafeCode(ValueError):
    pass


class CodeTimeout(TimeoutError):
    pass


@contextmanager
def time_limit(seconds: float):
    def _handler(signum, frame):
        raise CodeTimeout(f"exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def compile_function(code: str, name: str) -> Callable:
    """Compile an LLM-written `def {name}(...): ...` into a raw callable of
    whatever arity it declares. Raises UnsafeCode on anything dangerous.
    Static screens first, then exec in the restricted namespace."""
    _screen(code)
    ns: dict = {"__builtins__": _SAFE_BUILTINS, "math": math}
    exec(compile(code, f"<llm_{name}>", "exec"), ns)  # noqa: S102 — restricted ns
    fn = ns.get(name)
    if not callable(fn):
        raise UnsafeCode(f"no callable `{name}` defined")
    return fn


def compile_priority(code: str) -> Callable[[float, float], float]:
    """Compile a `def priority(item, remaining): ...`, coercing output to float."""
    fn = compile_function(code, "priority")

    def wrapped(item: float, remaining: float) -> float:
        return float(fn(item, remaining))  # NaN/inf handled by caller's comparison

    return wrapped
