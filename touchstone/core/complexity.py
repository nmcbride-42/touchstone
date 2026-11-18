"""Parsimony scoring for candidate models — the MDL half of "is it true?".

A model that fits equally well with fewer parameters and simpler structure is
more likely to be the real law (Occam, formalized as description length). We
approximate description length by counting fittable parameters plus AST
structure (operations, function calls, constants). Cheap, deterministic, and
enough to break ties between an overfit monster and a clean form.
"""
from __future__ import annotations

import ast
from typing import Any

# Per-node weights — operations and calls cost more than names/loads.
_WEIGHTS = {
    ast.BinOp: 1.0, ast.UnaryOp: 0.5, ast.Call: 1.5, ast.Compare: 1.0,
    ast.IfExp: 1.5, ast.Constant: 0.3, ast.Pow: 0.0,  # Pow is a BinOp.op, counted via BinOp
}


def complexity(code: str, n_params: int) -> dict[str, Any]:
    """Description-length proxy. Lower = simpler = preferred."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"n_params": n_params, "n_nodes": 9999, "ops": 9999, "dl": 9999.0}
    n_nodes = 0
    ops = 0.0
    for node in ast.walk(tree):
        n_nodes += 1
        ops += _WEIGHTS.get(type(node), 0.0)
    # description length: params dominate (each is a degree of freedom to
    # overfit with), structure is secondary.
    dl = 2.0 * n_params + ops
    return {"n_params": n_params, "n_nodes": n_nodes, "ops": round(ops, 1),
            "dl": round(dl, 2)}
