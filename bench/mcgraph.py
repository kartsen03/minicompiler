"""Reads minicompiler .mcg graph files and rebuilds them for PyTorch.

Mirrors src/parser.cpp (the subset the benchmark graphs use) and
src/random.cpp, so PyTorch sees the same ops, shapes, weights and inputs as
minicompiler. uniform_values() is checked against the same golden bits as
tests/test_parser.cpp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def uniform_values(count: int, seed: int, lo: float, hi: float) -> np.ndarray:
    """splitmix64-based uniform floats, bit-identical to src/random.cpp."""
    state = np.uint64(seed) + np.arange(1, count + 1, dtype=np.uint64) * _GOLDEN  # wraps mod 2^64
    z = (state ^ (state >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float32) * np.float32(2.0**-24)  # exact
    lo32, hi32 = np.float32(lo), np.float32(hi)
    return lo32 + (hi32 - lo32) * u  # float32 multiply, then float32 add


def _check_golden() -> None:
    bits = uniform_values(4, 42, -0.3, 0.7).view(np.uint32)
    expected = np.array([0x3EE214CA, 0xBE0F73A8, 0xBCAF4CC0, 0x3D350140], dtype=np.uint32)
    if not np.array_equal(bits, expected):
        raise RuntimeError("NumPy port of the minicompiler RNG disagrees with the C++ golden values")


@dataclass
class Op:
    name: str
    op: str
    operands: list[str]


@dataclass
class Graph:
    name: str = "graph"
    dims: dict[str, int] = field(default_factory=dict)
    inputs: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    constants: dict[str, np.ndarray] = field(default_factory=dict)
    ops: list[Op] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


_OPS = {"neg", "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "add", "sub", "mul", "div", "matmul"}


def parse(path: str, dims: dict[str, int] | None = None) -> Graph:
    overrides = dict(dims or {})
    g = Graph()

    def shape_of(text: str) -> tuple[int, ...]:
        m = re.fullmatch(r"f32\[(.*)\]", text.strip())
        if not m:
            raise ValueError(f"unsupported type {text!r}")
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
        return tuple(int(p) if p.lstrip("-").isdigit() else g.dims[p] for p in parts)

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            word = line.split()[0]
            rest = line[len(word):].strip()
            if word == "graph":
                g.name = rest
            elif word == "dim":
                name, value = (s.strip() for s in rest.split("="))
                g.dims[name] = overrides.pop(name, int(value))
            elif word == "input":
                name, type_text = (s.strip() for s in rest.split(":", 1))
                g.inputs.append((name, shape_of(type_text)))
            elif word == "const":
                name, rest2 = (s.strip() for s in rest.split(":", 1))
                type_text, init = (s.strip() for s in rest2.split("=", 1))
                shape = shape_of(type_text)
                count = int(np.prod(shape)) if shape else 1
                m = re.fullmatch(r"uniform\((.*)\)", init)
                if m:
                    kw = dict(kv.split("=") for kv in m.group(1).replace(" ", "").split(","))
                    values = uniform_values(count, int(kw["seed"]), float(kw["lo"]), float(kw["hi"]))
                elif init.startswith("["):
                    values = np.array([float(v) for v in init.strip("[]").split(",")], dtype=np.float32)
                else:
                    values = np.full(count, np.float32(float(init)), dtype=np.float32)
                g.constants[name] = values.reshape(shape)
            elif word == "output":
                g.outputs.extend(s.strip() for s in rest.split(","))
            else:
                name, expr = (s.strip() for s in line.split("=", 1))
                op, args = expr.split(None, 1)
                if op not in _OPS:
                    raise ValueError(f"unknown op {op!r}")
                g.ops.append(Op(name, op, [a.strip() for a in args.split(",")]))
    if overrides:
        raise ValueError(f"the graph declares no dim(s) {sorted(overrides)}")
    return g


def random_inputs(g: Graph, seed: int) -> list[np.ndarray]:
    """Input k is uniform_values(n, seed + k, -1, 1), as in make_random_inputs()."""
    out = []
    for k, (_, shape) in enumerate(g.inputs):
        count = int(np.prod(shape)) if shape else 1
        out.append(uniform_values(count, seed + k, -1.0, 1.0).reshape(shape))
    return out


_TORCH_EXPR = {
    "neg": "-{0}",
    "exp": "torch.exp({0})",
    "log": "torch.log({0})",
    "sqrt": "torch.sqrt({0})",
    "relu": "torch.relu({0})",
    "sigmoid": "torch.sigmoid({0})",
    "tanh": "torch.tanh({0})",
    "add": "{0} + {1}",
    "sub": "{0} - {1}",
    "mul": "{0} * {1}",
    "div": "{0} / {1}",
    "matmul": "{0} @ {1}",
}


def torch_source(g: Graph) -> str:
    """Straight-line Python for the graph, one PyTorch op per graph op, the way
    the graph is written (no op is merged or reordered)."""
    args = ", ".join(name for name, _ in g.inputs)
    lines = [f"def {g.name}({args}):"]
    for op in g.ops:
        lines.append(f"    {op.name} = " + _TORCH_EXPR[op.op].format(*op.operands))
    lines.append("    return (" + ", ".join(g.outputs) + ",)")
    return "\n".join(lines) + "\n"


def build_torch_fn(g: Graph, device: str):
    """Returns the graph as a Python function of its inputs, with the
    constants bound as tensors on `device`."""
    import torch

    scope = {"torch": torch}
    for name, value in g.constants.items():
        scope[name] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
    exec(compile(torch_source(g), f"<{g.name}.mcg>", "exec"), scope)
    return scope[g.name], scope


_check_golden()
