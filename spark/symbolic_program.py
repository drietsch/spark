"""SPARK M8: typed, serializable mathematical programs (not neural weights).

This is an early symbolic recurrent-cell prototype. Training synthesizes sparse
polynomial *structure* and rational constants using minimum-description-length
selection. Runtime executes a pure expression tree, optionally native-compiled.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

try:
    import numpy as np
except ImportError:  # runtime program evaluation needs only the standard library
    np = None

FORMAT = "spark-symbolic-program-v1"
ALLOWED = frozenset({"const", "state", "input", "add", "mul", "neg"})


def validate(node: dict, depth: int = 0) -> None:
    if not isinstance(node, dict) or depth > 40:
        raise ValueError("invalid or excessively deep expression")
    op = node.get("op")
    if op not in ALLOWED:
        raise ValueError(f"unsupported expression op: {op}")
    if op == "const":
        if set(node) != {"op", "num", "den"} or any(type(node[k]) is not int for k in ("num", "den")):
            raise ValueError("constant must be a rational integer numerator/denominator")
        if node["den"] <= 0 or node["den"] > 1024 or abs(node["num"]) > 10**9:
            raise ValueError("constant out of bounds")
    elif op in ("state", "input"):
        if set(node) != {"op"}:
            raise ValueError("variable has unexpected properties")
    elif op == "neg":
        if set(node) != {"op", "arg"}:
            raise ValueError("invalid negation")
        validate(node["arg"], depth+1)
    else:
        if set(node) != {"op", "left", "right"}:
            raise ValueError("invalid binary operation")
        validate(node["left"], depth+1)
        validate(node["right"], depth+1)


def evaluate(node: dict, state: float, token: float) -> float:
    op = node["op"]
    if op == "state":
        return float(state)
    if op == "input":
        return float(token)
    if op == "const":
        return node["num"] / node["den"]
    if op == "neg":
        return -evaluate(node["arg"], state, token)
    a = evaluate(node["left"], state, token)
    b = evaluate(node["right"], state, token)
    return a + b if op == "add" else a * b


def to_cpp(node: dict) -> str:
    """Safe C++ emission: only hardcoded whitelisted operators, no eval."""
    op = node["op"]
    if op == "state":
        return "s"
    if op == "input":
        return "x"
    if op == "const":
        return f"({node['num']}.0 / {node['den']}.0)"
    if op == "neg":
        return f"(-{to_cpp(node['arg'])})"
    symbol = "+" if op == "add" else "*"
    return f"({to_cpp(node['left'])} {symbol} {to_cpp(node['right'])})"


def cpp_source(node: dict) -> str:
    validate(node)
    return ("// SPARK symbolic expression; generated from validated AST.\n"
            "// No dynamic lookup tables or learned parameter matrices.\n"
            "#include <cmath>\n"
            "#include <cstddef>\n"
            "extern \"C\" double spark_expr_step_v1(double s, double x) {\n"
            f"  return {to_cpp(node)};\n"
            "}\n"
            "extern \"C\" double spark_expr_rollout_v1(double s, const double *xs, std::size_t n) {\n"
            "  for (std::size_t i = 0; i < n; ++i) s = spark_expr_step_v1(s, xs[i]);\n"
            "  return s;\n"
            "}\n")


def const(value: Fraction) -> dict:
    return {"op": "const", "num": value.numerator, "den": value.denominator}


def binop(op: str, left: dict, right: dict) -> dict:
    return {"op": op, "left": left, "right": right}


def monomial(i: int, j: int) -> dict:
    terms = [{"op": "state"} for _ in range(i)] + [{"op": "input"} for _ in range(j)]
    if not terms:
        return const(Fraction(1))
    expr = terms[0]
    for term in terms[1:]:
        expr = binop("mul", expr, term)
    return expr


def polynomial_program(terms: list[tuple[int, int, Fraction]]) -> dict:
    expr = None
    for i, j, coeff in sorted(terms):
        if coeff == 0:
            continue
        term = monomial(i, j)
        if coeff == -1:
            term = {"op": "neg", "arg": term}
        elif coeff != 1:
            term = binop("mul", const(coeff), term)
        expr = term if expr is None else binop("add", expr, term)
    return expr if expr is not None else const(Fraction(0))


def serialize(path: Path, expression: dict, metadata: dict | None = None) -> None:
    validate(expression)
    doc = {"format": FORMAT, "expression": expression, "metadata": metadata or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")


def deserialize(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if set(raw) != {"format", "expression", "metadata"} or raw["format"] != FORMAT:
        raise ValueError("unsupported SPARK symbolic program")
    validate(raw["expression"])
    return raw["expression"]


def candidate_monomials(degree: int, *, square_free: bool = False) -> list[tuple[int, int]]:
    return [(i, d - i) for d in range(degree + 1) for i in range(d + 1)
            if not square_free or (i <= 1 and d-i <= 1)]


def design_matrix(examples: list[tuple[float, float, float]], terms: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[s**i * x**j for i,j in terms] for s,x,_ in examples], dtype=np.float64)


@dataclass(frozen=True)
class SynthesizedProgram:
    expression: dict
    terms: tuple[tuple[int, int, Fraction], ...]
    degree: int
    train_mse: float
    description_cost: int

    def step(self, state: float, token: float) -> float:
        return evaluate(self.expression, state, token)


def synthesize(examples: list[tuple[float, float, float]], *, max_degree: int = 3,
               denominator_limit: int = 32, square_free: bool = False,
               tolerance: float = 1e-10) -> SynthesizedProgram:
    """Search equation structures, fit constants, then rationalize and prune.

    This is constrained polynomial program synthesis, NOT arbitrary program
    invention. The grammar and max degree are fixed by the experimenter.
    """
    if np is None:
        raise ImportError("numpy is required for offline symbolic training only")
    if len(examples) < 12 or not 0 <= max_degree <= 5 or not 1 <= denominator_limit <= 128:
        raise ValueError("invalid program-synthesis configuration")
    rows = np.asarray(examples, dtype=np.float64)
    if rows.shape != (len(examples), 3) or not np.all(np.isfinite(rows)):
        raise ValueError("examples must be finite (state, input, next_state) triples")
    y = rows[:, 2]
    best = None
    for degree in range(max_degree + 1):
        names = candidate_monomials(degree, square_free=square_free)
        X = design_matrix(examples, names)
        if np.linalg.matrix_rank(X) < len(names):
            continue
        fitted = np.linalg.lstsq(X, y, rcond=None)[0]
        # Coefficient discovery: prefer small rational constants that are
        # expressible directly in compiled code, not arbitrary float arrays.
        rational = [Fraction(float(v)).limit_denominator(denominator_limit) for v in fitted]
        coeff = np.asarray([float(v) for v in rational], dtype=np.float64)
        mse = float(np.mean((X @ coeff - y)**2))
        nz = [(i, j, value) for (i,j), value in zip(names, rational) if value]
        # MDL favors sparse, low-complexity programs once error is small.
        cost = sum(2+i+j + (1 if abs(c) != 1 else 0) + (1 if c.denominator > 1 else 0)
                   for i,j,c in nz)
        # Select the simplest expression that fits to tolerance. Otherwise
        # choose error before complexity without overclaiming exact recovery.
        key = (int(mse > tolerance), mse if mse > tolerance else 0.0, cost, degree)
        if best is None or key < best[0]:
            best = (key, SynthesizedProgram(polynomial_program(nz), tuple(nz), degree, mse, cost))
    if best is None:
        raise ValueError("not enough independent observations to identify a polynomial")
    validate(best[1].expression)
    return best[1]


def affine_baseline(examples: list[tuple[float,float,float]]) -> tuple[float, float, float]:
    X = design_matrix(examples, [(0,0), (1,0), (0,1)])
    return tuple(float(v) for v in np.linalg.lstsq(X, np.asarray([e[2] for e in examples]), rcond=None)[0])
