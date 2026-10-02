#!/usr/bin/env python3
"""Shortens `perf report --stdio --sort dso,symbol` lines for the docs: drops parameter
lists, abbreviates Eigen's array maps, and collapses deeply nested template
arguments, keeping the outer template arguments that tell kernels apart."""

import sys

REPLACE = {
    "Eigen::Map<Eigen::Array<float, -1, 1, 0, -1, 1> const, 0, Eigen::Stride<0, 0> >": "ConstArrayMap",
    "Eigen::Map<Eigen::Array<float, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >": "ArrayMap",
    "(anonymous namespace)::": "",
    "minicompiler::": "",
}


def shorten(s: str) -> str:
    for old, new in REPLACE.items():
        s = s.replace(old, new)
    # The parameter list is the last top-level parenthesized group.
    depth, start, end = 0, -1, -1
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
    if 0 <= start < end:
        s = s[:start] + s[end + 1:]
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
            if depth == 2:
                out.append("<...>")
            if depth >= 2:
                continue
        elif ch == ">":
            depth -= 1
            if depth >= 1:
                continue
        elif depth >= 2:
            continue
        out.append(ch)
    return " ".join("".join(out).split())


for line in sys.stdin:
    parts = line.split(None, 3)  # "40.64%", shared object, "[.]", symbol
    if len(parts) == 4 and parts[0].endswith("%"):
        print(f"{parts[0]:>7}  {parts[1]:<10} {shorten(parts[3])}")
