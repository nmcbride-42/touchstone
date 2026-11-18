import pytest

from touchstone.core.sandbox import (CodeTimeout, UnsafeCode, compile_function,
                                     compile_priority, time_limit)


def test_compiles_plain_function():
    fn = compile_function("def f(x, a):\n    return a * x", "f")
    assert fn(3, 2) == 6


def test_math_import_is_allowed():
    fn = compile_function("import math\ndef f(x):\n    return math.sqrt(x)", "f")
    assert fn(9.0) == 3.0


@pytest.mark.parametrize("code", [
    "import os\ndef f():\n    return os.getcwd()",
    "def f():\n    import subprocess\n    return 1",
])
def test_other_imports_fail(code):
    with pytest.raises(ImportError):
        compile_function(code, "f")()


@pytest.mark.parametrize("code", [
    "def f():\n    return ().__class__.__bases__",
    "def f():\n    return open('x')",
    "def f():\n    return eval('1')",
    "def f():\n    return getattr(1, 'real')",
])
def test_forbidden_tokens_rejected(code):
    with pytest.raises(UnsafeCode):
        compile_function(code, "f")


def test_no_filesystem_builtins():
    fn = compile_function("def f():\n    return print", "f")
    with pytest.raises(NameError):
        fn()


def test_missing_function_rejected():
    with pytest.raises(UnsafeCode):
        compile_function("def g():\n    return 1", "f")


def test_time_limit_interrupts_infinite_loop():
    fn = compile_function("def f():\n    while True:\n        pass", "f")
    with pytest.raises(CodeTimeout):
        with time_limit(0.2):
            fn()


def test_compile_priority_coerces_to_float():
    pr = compile_priority("def priority(item, remaining):\n    return 1")
    assert pr(0.1, 0.5) == 1.0 and isinstance(pr(0.1, 0.5), float)


# ---- escape routes that avoid every banned token ---------------------------
# A running generator's frame leads (via f_back) to the caller's frame, whose
# f_builtins holds the real builtins and whose f_globals is the verifier's
# own module namespace. Both attacks below were demonstrated against an
# earlier token-only screen; they must now be rejected before anything runs.

FRAME_FILE_READ = '''
def construct(N):
    def g():
        yield it.gi_frame.f_back
    it = g()
    fr = list(it)[0]
    while fr is not None:
        b = fr.f_builtins
        if "op" + "en" in b:
            return [len(b["op" + "en"]("/etc/hostname").read())]
        fr = fr.f_back
    return [0]
'''

FRAME_PATCH_VERIFIER = '''
def construct(N):
    def g():
        yield it.gi_frame.f_back
    it = g()
    fr = list(it)[0]
    while fr is not None:
        gl = fr.f_globals
        if "is_ap_free" in gl:
            gl["is_ap_free"] = lambda s: (True, ())
        fr = fr.f_back
    return range(N)
'''


@pytest.mark.parametrize("code", [FRAME_FILE_READ, FRAME_PATCH_VERIFIER])
def test_frame_walk_rejected_statically(code):
    with pytest.raises(UnsafeCode):
        compile_function(code, "construct")


def test_frame_walk_cannot_fake_a_proof():
    from touchstone.problems import apfree
    h = apfree.ApFreeHandle()
    v = h.verify(FRAME_PATCH_VERIFIER)
    assert not v.valid and "forbidden attribute" in v.detail["error"]
    # nothing leaked into the verifier: a set with an AP is still rejected
    assert apfree.is_ap_free.__module__ == apfree.__name__
    bad = h.verify("def construct(N):\n    return [0, 1, 2]")
    assert not bad.valid and "contains AP" in bad.detail["error"]


@pytest.mark.parametrize("code", [
    "def f():\n    def g():\n        yield 1\n    return g().gi_code",
    "def f(e):\n    return e.tb_frame",
    "def f(c):\n    return c.co_consts",
    "def f():\n    return ('{0._' + '_class_' + '_}').format(1)",
    "def f():\n    return '{0.real}'.format_map({})",
    "def f():\n    return (1).mro",
    "def f(x):\n    return x._private",
    "import math\ndef f():\n    math.floor = abs\n    return 1",
    "import math\ndef f():\n    del math.floor\n    return 1",
    "async def f():\n    return 1",
])
def test_internal_attributes_and_mutation_rejected(code):
    with pytest.raises(UnsafeCode):
        compile_function(code, "f")


def test_underscore_loop_variable_still_allowed():
    fn = compile_function("def f(n):\n    t = 0\n    for _ in range(n):\n        t += 1\n    return t", "f")
    assert fn(4) == 4


def test_generator_output_is_iterated_under_the_time_limit():
    from touchstone.problems import apfree
    v = apfree.ApFreeHandle().verify(
        "def construct(N):\n    yield 0\n    while True:\n        pass")
    assert not v.valid and "CodeTimeout" in v.detail["error"]


def test_verify_isolated_matches_in_process():
    from touchstone.core.isolate import verify_isolated
    from touchstone.problems import apfree
    h = apfree.ApFreeHandle()
    v = verify_isolated(h, apfree.GREEDY)
    assert v.valid and v.score == h.verify(apfree.GREEDY).score
