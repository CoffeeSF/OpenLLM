#!/usr/bin/env python3
"""Fast, dependency-free integrity and geometry check for the shipped assets."""
from __future__ import annotations

import argparse
import hashlib
import math
import struct
from pathlib import Path

HEADER = 256


def verify_model(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < HEADER or data[:4] != b"OCQ8":
        raise ValueError("missing OCQ8 header")
    version, dim, hidden, layers, heads, kv_heads, vocab, max_seq = struct.unpack_from("<I7I", data, 4)
    shared = data[36] != 0
    if (version, dim, hidden, layers, heads, kv_heads, vocab, max_seq, shared) != (1, 64, 172, 5, 8, 4, 512, 512, True):
        raise ValueError("unexpected stories260K architecture: %r" % ((version, dim, hidden, layers, heads, kv_heads, vocab, max_seq, shared),))
    kv_dim = dim * kv_heads // heads
    offset = HEADER + (layers * dim + layers * dim + dim) * 4
    matrices = [(vocab, dim), (layers * dim, dim), (layers * kv_dim, dim), (layers * kv_dim, dim), (layers * dim, dim), (layers * hidden, dim), (layers * dim, hidden), (layers * hidden, dim)]
    largest_abs = 0.0
    for rows, width in matrices:
        for _ in range(rows):
            scale = struct.unpack_from("<f", data, offset)[0]
            if not math.isfinite(scale) or scale < 0:
                raise ValueError("invalid row scale")
            values = struct.unpack_from("<%db" % width, data, offset + 4)
            largest_abs = max(largest_abs, max((abs(value * scale) for value in values), default=0.0))
            offset += width + 4
    if offset != len(data): raise ValueError("unexpected trailing or missing model bytes")
    print("model OK: %d bytes, max reconstructed absolute weight %.8f" % (len(data), largest_abs))
    print("sha256", hashlib.sha256(data).hexdigest())


def verify_tokenizer(path: Path) -> None:
    data = path.read_bytes()
    offset = 4
    maximum = struct.unpack_from("<I", data, 0)[0]
    seen = set()
    for _ in range(512):
        score, length = struct.unpack_from("<fI", data, offset); offset += 8
        piece = data[offset:offset + length]; offset += length
        if len(piece) != length or not math.isfinite(score): raise ValueError("invalid tokenizer record")
        seen.add(piece)
    if offset != len(data): raise ValueError("unexpected tokenizer bytes")
    if maximum != max(map(len, seen)): raise ValueError("tokenizer maximum length mismatch")
    print("tokenizer OK: %d bytes, 512 pieces, maximum piece length %d" % (len(data), maximum))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model/model.bin"))
    parser.add_argument("--tokenizer", type=Path, default=Path("model/tokenizer.bin"))
    args = parser.parse_args()
    verify_model(args.model)
    verify_tokenizer(args.tokenizer)


if __name__ == "__main__":
    main()
