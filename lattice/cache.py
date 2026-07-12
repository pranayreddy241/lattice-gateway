"""Semantic response cache.

Near-duplicate prompts ("what is a mutex" / "What is a mutex?") should
not pay for a second inference. Prompts are embedded with a hashed
character-trigram vectorizer (stdlib-only, deterministic, no model
download); entries whose cosine similarity clears `threshold` count as
hits. LRU + TTL bound memory and staleness.

Lookup path (found via load testing — a naive full scan of 2k entries
in pure Python blocked the event loop and pushed gateway p95 past 1s):
  1. O(1) exact-match on the prompt hash.
  2. Inverted-index candidate pruning: each entry is posted under its
     top-K strongest embedding dimensions; a query only scores entries
     sharing at least one of its own top-K dimensions. Near-duplicates
     share dominant trigram buckets, so recall stays high while the
     scored set shrinks by orders of magnitude.
Swap in FAISS/HNSW behind the same interface for bigger caches.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

DIM = 256
TOP_K_POSTINGS = 8


def embed(text: str, dim: int = DIM) -> dict[int, float]:
    """Hashed char-trigram embedding as a sparse {dim: weight} map, L2-normalized."""
    counts: dict[int, float] = {}
    t = f"  {text.lower().strip()}  "
    for i in range(len(t) - 2):
        tri = t[i : i + 3]
        h = int.from_bytes(hashlib.blake2b(tri.encode(), digest_size=4).digest(), "big")
        counts[h % dim] = counts.get(h % dim, 0.0) + 1.0
    norm = math.sqrt(sum(x * x for x in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(w * b.get(d, 0.0) for d, w in a.items())


def _top_dims(vector: dict[int, float], k: int = TOP_K_POSTINGS) -> list[int]:
    return sorted(vector, key=lambda d: vector[d], reverse=True)[:k]


@dataclass
class _Entry:
    vector: dict[int, float]
    value: object
    inserted_at: float
    posted_dims: list[int] = field(default_factory=list)


class SemanticCache:
    def __init__(
        self,
        max_entries: int = 2048,
        ttl_seconds: float = 300.0,
        threshold: float = 0.95,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._postings: dict[int, set[str]] = {}
        self.hits = 0
        self.misses = 0

    # -- internals ----------------------------------------------------

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        for dim in entry.posted_dims:
            bucket = self._postings.get(dim)
            if bucket is not None:
                bucket.discard(key)
                if not bucket:
                    del self._postings[dim]

    def _evict_expired(self) -> None:
        now = self._clock()
        stale = [
            k
            for k, e in self._entries.items()
            if now - e.inserted_at > self.ttl_seconds
        ]
        for k in stale:
            self._remove(k)

    # -- public API ---------------------------------------------------

    def get(self, prompt: str) -> object | None:
        self._evict_expired()
        exact_key = hashlib.sha256(prompt.encode()).hexdigest()
        entry = self._entries.get(exact_key)
        if entry is not None:  # O(1) fast path
            self._entries.move_to_end(exact_key)
            self.hits += 1
            return entry.value

        query = embed(prompt)
        candidates: set[str] = set()
        for dim in _top_dims(query):
            candidates |= self._postings.get(dim, set())
        best_key, best_sim = None, 0.0
        for key in candidates:
            sim = cosine(query, self._entries[key].vector)
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None and best_sim >= self.threshold:
            self._entries.move_to_end(best_key)  # LRU touch
            self.hits += 1
            return self._entries[best_key].value
        self.misses += 1
        return None

    def put(self, prompt: str, value: object) -> None:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in self._entries:
            self._remove(key)
        vector = embed(prompt)
        dims = _top_dims(vector)
        self._entries[key] = _Entry(vector, value, self._clock(), dims)
        self._entries.move_to_end(key)
        for dim in dims:
            self._postings.setdefault(dim, set()).add(key)
        while len(self._entries) > self.max_entries:
            oldest = next(iter(self._entries))
            self._remove(oldest)  # evict LRU

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._entries)
