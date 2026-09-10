#!/usr/bin/env python3
"""Compare a first-token OCQ8 forward pass with the legacy float checkpoint.

This is host-side development verification only; the OpenComputers runtime has
no Python dependency. The comparison covers every matrix, norm, RoPE, softmax,
and classifier path at position zero.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def take(values, at, count): return values[at:at + count], at + count
def matvec(weights, x, rows, columns): return [sum(weights[r * columns + c] * x[c] for c in range(columns)) for r in range(rows)]
def norm(x, w):
    scale = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
    return [x[i] * scale * w[i] for i in range(len(x))]
def softmax(x):
    maximum = max(x); raw = [math.exp(v - maximum) for v in x]; total = sum(raw)
    return [v / total for v in raw]


def legacy_forward(path: Path, token: int) -> list[float]:
    raw = path.read_bytes()
    dim, hidden, layers, heads, kv_heads, raw_vocab, seq = struct.unpack_from("<7i", raw)
    assert raw_vocab > 0 and (dim, hidden, layers, heads, kv_heads, raw_vocab) == (64, 172, 5, 8, 4, 512)
    values = list(struct.unpack_from("<%df" % ((len(raw) - 28) // 4), raw, 28)); at = 0; vocab = raw_vocab; kv = 32
    emb, at = take(values, at, vocab * dim); ra, at = take(values, at, layers * dim); wq, at = take(values, at, layers * dim * dim); wk, at = take(values, at, layers * kv * dim); wv, at = take(values, at, layers * kv * dim); wo, at = take(values, at, layers * dim * dim); rf, at = take(values, at, layers * dim); w1, at = take(values, at, layers * hidden * dim); w2, at = take(values, at, layers * dim * hidden); w3, at = take(values, at, layers * hidden * dim); final, at = take(values, at, dim)
    x = emb[token * dim:(token + 1) * dim]
    for layer in range(layers):
        xb = norm(x, ra[layer * dim:(layer + 1) * dim])
        q = matvec(wq[layer * dim * dim:(layer + 1) * dim * dim], xb, dim, dim)
        _k = matvec(wk[layer * kv * dim:(layer + 1) * kv * dim], xb, kv, dim)
        v = matvec(wv[layer * kv * dim:(layer + 1) * kv * dim], xb, kv, dim)
        # At position zero, RoPE cannot change values and every head attends only to v.
        attention = []
        for head in range(heads): attention.extend(v[(head // 2) * 8:(head // 2 + 1) * 8])
        out = matvec(wo[layer * dim * dim:(layer + 1) * dim * dim], attention, dim, dim)
        x = [x[i] + out[i] for i in range(dim)]
        xb = norm(x, rf[layer * dim:(layer + 1) * dim])
        a = matvec(w1[layer * hidden * dim:(layer + 1) * hidden * dim], xb, hidden, dim)
        b = matvec(w3[layer * hidden * dim:(layer + 1) * hidden * dim], xb, hidden, dim)
        gate = [a[i] / (1.0 + math.exp(-a[i])) * b[i] for i in range(hidden)]
        out = matvec(w2[layer * dim * hidden:(layer + 1) * dim * hidden], gate, dim, hidden)
        x = [x[i] + out[i] for i in range(dim)]
    x = norm(x, final)
    return matvec(emb, x, vocab, dim)


def q8_forward(path: Path, token: int) -> list[float]:
    data = path.read_bytes(); assert data[:4] == b"OCQ8"; _, dim, hidden, layers, heads, kv_heads, vocab, _ = struct.unpack_from("<I7I", data, 4)
    kv = dim * kv_heads // heads; offset = 256
    def floats(count):
        nonlocal offset
        result = list(struct.unpack_from("<%df" % count, data, offset)); offset += count * 4; return result
    ra, rf, final = floats(layers * dim), floats(layers * dim), floats(dim)
    def section(rows, columns):
        nonlocal offset
        start = offset; offset += rows * (columns + 4); return start, rows, columns
    emb = section(vocab, dim); wq = section(layers * dim, dim); wk = section(layers * kv, dim); wv = section(layers * kv, dim); wo = section(layers * dim, dim); w1 = section(layers * hidden, dim); w2 = section(layers * dim, hidden); w3 = section(layers * hidden, dim)
    def qmat(section, x, row_base, rows):
        start, _, columns = section; output = []
        for row in range(rows):
            here = start + (row_base + row) * (columns + 4); scale = struct.unpack_from("<f", data, here)[0]; values = struct.unpack_from("<%db" % columns, data, here + 4)
            output.append(scale * sum(values[i] * x[i] for i in range(columns)))
        return output
    start, _, columns = emb; scale = struct.unpack_from("<f", data, start + token * (columns + 4))[0]; row = struct.unpack_from("<%db" % columns, data, start + token * (columns + 4) + 4); x = [scale * v for v in row]
    for layer in range(layers):
        xb = norm(x, ra[layer * dim:(layer + 1) * dim]); q = qmat(wq, xb, layer * dim, dim); _k = qmat(wk, xb, layer * kv, kv); v = qmat(wv, xb, layer * kv, kv)
        # Match the OC runtime's disk-backed Q12 key/value cache at position 0.
        cached_v = [max(-32768, min(32767, math.floor(value * 4096 + (0.5 if value >= 0 else -0.5)))) / 4096 for value in v]
        attention = []
        for head in range(heads): attention.extend(cached_v[(head // 2) * 8:(head // 2 + 1) * 8])
        out = qmat(wo, attention, layer * dim, dim); x = [x[i] + out[i] for i in range(dim)]
        xb = norm(x, rf[layer * dim:(layer + 1) * dim]); a = qmat(w1, xb, layer * hidden, hidden); b = qmat(w3, xb, layer * hidden, hidden); gate = [a[i] / (1 + math.exp(-a[i])) * b[i] for i in range(hidden)]
        out = qmat(w2, gate, layer * dim, dim); x = [x[i] + out[i] for i in range(dim)]
    return qmat(emb, norm(x, final), 0, vocab)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("legacy", type=Path); parser.add_argument("--model", type=Path, default=Path("model/model.bin")); parser.add_argument("--token", type=int, default=1)
    args = parser.parse_args(); float_logits = legacy_forward(args.legacy, args.token); q8_logits = q8_forward(args.model, args.token)
    errors = [abs(a - b) for a, b in zip(float_logits, q8_logits)]
    print("first-token logits: max absolute error %.8f, mean absolute error %.8f" % (max(errors), sum(errors) / len(errors)))
    print("float argmax", max(range(len(float_logits)), key=float_logits.__getitem__), "OCQ8 argmax", max(range(len(q8_logits)), key=q8_logits.__getitem__))


if __name__ == "__main__": main()
