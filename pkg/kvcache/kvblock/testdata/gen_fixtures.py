"""Generate ground-truth block hashes using vLLM's exact sha256_cbor logic.

Mirrors:
- vllm/utils/hashing.py::sha256_cbor
- vllm/v1/core/kv_cache_utils.py::init_none_hash, hash_block_tokens, get_request_block_hasher
- vllm/v1/core/kv_cache_utils.py::maybe_convert_block_hash (with VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1)

Outputs JSON fixtures the Go side will load and compare against.
"""

from __future__ import annotations

import cbor2
import hashlib
import json
import os
from pathlib import Path


def sha256_cbor(obj) -> bytes:
    return hashlib.sha256(cbor2.dumps(obj, canonical=True)).digest()


def make_none_hash(seed: str | None) -> bytes:
    """vLLM's init_none_hash. With seed='' or any string -> sha256_cbor(seed)."""
    if seed is None:
        # Without PYTHONHASHSEED vLLM uses os.urandom(32). Not reproducible; skip.
        raise ValueError("seed must be set for reproducibility")
    return sha256_cbor(seed)


def hash_block_tokens(parent: bytes, tokens: tuple[int, ...], extra) -> bytes:
    """vLLM's hash_block_tokens with sha256_cbor."""
    return sha256_cbor((parent, tokens, extra))


def truncate_to_uint64(digest: bytes) -> int:
    """vLLM's maybe_convert_block_hash with VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1."""
    return int.from_bytes(digest, byteorder="big") & ((1 << 64) - 1)


def chunk(tokens: list[int], block_size: int) -> list[list[int]]:
    out = []
    for i in range(0, len(tokens), block_size):
        end = i + block_size
        if end > len(tokens):
            break
        out.append(tokens[i:end])
    return out


def hash_request(
    tokens: list[int],
    block_size: int,
    seed: str,
    per_block_extras: list | None = None,
) -> tuple[list[bytes], list[int]]:
    """Returns (full_digests, truncated_uint64s)."""
    none_hash = make_none_hash(seed)
    chunks = chunk(tokens, block_size)

    if per_block_extras is None:
        per_block_extras = [None] * len(chunks)
    assert len(per_block_extras) == len(chunks), (
        f"extras length {len(per_block_extras)} != chunks {len(chunks)}"
    )

    digests: list[bytes] = []
    truncated: list[int] = []
    parent = none_hash
    for i, ck in enumerate(chunks):
        extras = per_block_extras[i]
        # vLLM hashes a tuple, not a list. cbor2 encodes both as arrays
        # identically, but mirror the tuple to be safe.
        d = hash_block_tokens(parent, tuple(ck), extras)
        digests.append(d)
        truncated.append(truncate_to_uint64(d))
        parent = d
    return digests, truncated


def main() -> None:
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)

    fixtures = []

    # 1) Plain text, no extras, single full block
    fixtures.append({
        "name": "plain_one_block",
        "seed": "0",
        "block_size": 16,
        "tokens": list(range(1, 17)),
        "extras": None,
    })

    # 2) Plain text, no extras, several blocks; tokens vary in width
    fixtures.append({
        "name": "plain_three_blocks_mixed_ids",
        "seed": "0",
        "block_size": 16,
        "tokens": (
            list(range(0, 16))
            + [200, 256, 257, 65535, 65536, 100000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            + [1234567, 999999, 50000, 60000, 70000, 80000, 90000, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ),
        "extras": None,
    })

    # 3) Empty seed (NONE_HASH = sha256_cbor(""))
    fixtures.append({
        "name": "empty_seed",
        "seed": "",
        "block_size": 16,
        "tokens": list(range(100, 116)),
        "extras": None,
    })

    # 4) Numeric-string seed
    fixtures.append({
        "name": "seed_42",
        "seed": "42",
        "block_size": 16,
        "tokens": list(range(1, 17)),
        "extras": None,
    })

    # 5) Lora extras: extras = ("my-lora",) — tuple of one string
    fixtures.append({
        "name": "lora_only",
        "seed": "0",
        "block_size": 16,
        "tokens": list(range(1, 33)),
        "extras": [("my-lora",)] * 2,
    })

    # 6) Cache-salt only on first block
    fixtures.append({
        "name": "cache_salt_first_block_only",
        "seed": "0",
        "block_size": 16,
        "tokens": list(range(1, 33)),
        "extras": [("salt-xyz",), None],
    })

    # 7) MM-style: lora + multiple mm identifiers + cache_salt on block 0,
    # then only mm identifier on block 1.
    fixtures.append({
        "name": "lora_mm_salt",
        "seed": "0",
        "block_size": 16,
        "tokens": list(range(1, 33)),
        "extras": [
            ("my-lora", "mm_hash_a", "mm_hash_b", "salt-xyz"),
            ("my-lora", "mm_hash_b"),
        ],
    })

    # 8) Bigger block size
    fixtures.append({
        "name": "block_size_64",
        "seed": "0",
        "block_size": 64,
        "tokens": list(range(1, 129)),
        "extras": None,
    })

    # 9) Non-empty seed with longer chain
    fixtures.append({
        "name": "seed_long_chain",
        "seed": "vllm-prefix-cache-seed",
        "block_size": 16,
        "tokens": list(range(1000, 1000 + 16 * 5)),
        "extras": None,
    })

    results = []
    for fx in fixtures:
        digests, truncated = hash_request(
            fx["tokens"], fx["block_size"], fx["seed"], fx["extras"]
        )
        results.append({
            "name": fx["name"],
            "seed": fx["seed"],
            "block_size": fx["block_size"],
            "tokens": fx["tokens"],
            "extras": fx["extras"],
            "digests_hex": [d.hex() for d in digests],
            "truncated_uint64": truncated,
            "none_hash_hex": make_none_hash(fx["seed"]).hex(),
        })

    out_path = out_dir / "vllm_block_hashes.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {len(results)} fixtures to {out_path}")
    for r in results:
        print(f"  {r['name']:<30} seed={r['seed']!r:<26} block_size={r['block_size']:<3} "
              f"blocks={len(r['truncated_uint64'])} first_uint64={r['truncated_uint64'][0]}")


if __name__ == "__main__":
    main()
