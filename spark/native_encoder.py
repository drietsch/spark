"""Portable SPARKENC v1 writer for the *learned*, matrix-free semantic encoder.

The runtime consumes the binary artifact without JSON parsing, Python or PyTorch.
All serialized feature names are ASCII and all vote weights are signed int8.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from spark.semantic import BITS, SemanticEncoder

MAGIC = b'SPARKENC'
HEADER = struct.Struct('<8sHHIQQQQIIQ')  # exactly 64 bytes
LENGTH = struct.Struct('<H')
VERSION = (1, 0)
MAX_FEATURES = 3_000_000
MAX_BINARY_SIZE = 1 << 30


def export_encoder(encoder: SemanticEncoder, path: Path) -> dict:
    if not isinstance(encoder, SemanticEncoder):
        raise TypeError('expected SemanticEncoder')
    if len(encoder.weights) > MAX_FEATURES or not 0 <= encoder.fallback_bits < (1 << BITS):
        raise ValueError('encoder exceeds native limits')
    payload = bytearray(encoder.fallback_bits.to_bytes(32, 'little'))
    for key, values in sorted(encoder.weights.items()):
        if not isinstance(key, str) or not isinstance(values, list) or len(values) != 256:
            raise ValueError('invalid native encoder feature')
        name = key.encode('ascii')
        if not name or len(name) > 127 or b'\x00' in name:
            raise ValueError('invalid native feature key length')
        if any(type(v) is not int or not -7 <= v <= 7 for v in values):
            raise ValueError('invalid native feature weight')
        payload += LENGTH.pack(len(name)) + name + struct.pack('<256b', *values)
    total = HEADER.size + len(payload)
    if total > MAX_BINARY_SIZE:
        raise ValueError('encoder exceeds maximum native file size')
    header = HEADER.pack(MAGIC, *VERSION, 0, len(encoder.weights), encoder.fingerprint64(),
                         total, HEADER.size, zlib.crc32(payload), 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)
    return {'features': len(encoder.weights), 'bytes': total,
            'fingerprint64': f'{encoder.fingerprint64():016x}', 'format': 'SPARKENC/1.0'}
