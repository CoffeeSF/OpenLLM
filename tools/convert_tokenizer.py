#!/usr/bin/env python3
"""Turn a SentencePiece .model into llama2.c's small tokenizer.bin format.

This intentionally parses only the protobuf fields SentencePiece uses for
ModelProto.pieces. It avoids a Python SentencePiece dependency during setup.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("invalid protobuf varint")


def fields(data: bytes):
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 1:
            value, offset = data[offset:offset + 8], offset + 8
        elif wire == 2:
            size, offset = read_varint(data, offset)
            value, offset = data[offset:offset + size], offset + size
        elif wire == 5:
            value, offset = data[offset:offset + 4], offset + 4
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)
        if offset > len(data):
            raise ValueError("truncated protobuf field")
        yield number, wire, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-vocab", type=int, default=512)
    args = parser.parse_args()
    pieces: list[tuple[float, bytes]] = []
    for number, wire, message in fields(args.input.read_bytes()):
        if number != 1 or wire != 2:
            continue
        text = None
        score = 0.0
        for sub_number, sub_wire, value in fields(message):
            if sub_number == 1 and sub_wire == 2:
                text = value
            elif sub_number == 2 and sub_wire == 5:
                score = struct.unpack("<f", value)[0]
        if text is not None:
            # llama2.c's tokenizer exporter writes display-friendly forms rather
            # than raw SentencePiece pieces. Match its tok512.bin exactly.
            if text == b"<s>":
                text = b"\n<s>\n"
            elif text == b"</s>":
                text = b"\n</s>\n"
            else:
                text = text.replace(b"\xe2\x96\x81", b" ")
            pieces.append((score, text))
    if len(pieces) != args.expect_vocab:
        raise ValueError("expected %d pieces, got %d" % (args.expect_vocab, len(pieces)))
    maximum = max(len(piece) for _, piece in pieces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        out.write(struct.pack("<I", maximum))
        for score, piece in pieces:
            out.write(struct.pack("<fI", score, len(piece)))
            out.write(piece)
    print("wrote", args.output, args.output.stat().st_size, "bytes; max token length", maximum)


if __name__ == "__main__":
    main()
