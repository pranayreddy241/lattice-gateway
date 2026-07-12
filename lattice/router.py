"""Latency-aware routing with power-of-two-choices (P2C).

Each backend keeps an EWMA of observed latency and a count of
outstanding requests. P2C samples two healthy backends and picks the
one with the lower score `ewma_latency * (outstanding + 1)` — a classic
trick that gets most of the benefit of full least-loaded routing
without a global scan, and avoids herd behavior on stale data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .backends import Backend
from .breaker import CircuitBreaker


class NoHealthyBackend(RuntimeError):
    """All backends have open circuit breakers."""


@dataclass
class BackendState:
    backend: Backend
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    ewma_latency_ms: float = 50.0  # optimistic prior
    outstanding: int = 0
    total_requests: int = 0
    total_failures: int = 0

    EWMA_ALPHA = 0.2

    @property
    def score(self) -> float:
        return self.ewma_latency_ms * (self.outstanding + 1)

    def on_dispatch(self) -> None:
        self.outstanding += 1
        self.total_requests += 1

    def on_complete(self, latency_ms: float, ok: bool) -> None:
        self.outstanding = max(0, self.outstanding - 1)
        if ok:
            self.breaker.record_success()
            self.ewma_latency_ms = (
                self.EWMA_ALPHA * latency_ms
                + (1 - self.EWMA_ALPHA) * self.ewma_latency_ms
            )
        else:
            self.total_failures += 1
            self.breaker.record_failure()


class Router:
    def __init__(
        self,
        backends: list[Backend],
        strategy: str = "p2c_ewma",
        breaker_factory=CircuitBreaker,
        rng: random.Random | None = None,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend required")
        self.states = {
            b.name: BackendState(backend=b, breaker=breaker_factory())
            for b in backends
        }
        self.strategy = strategy
        self._rng = rng or random.Random()
        self._rr_index = 0

    def healthy(self, exclude: set[str] = frozenset()) -> list[BackendState]:
        return [
            s
            for name, s in self.states.items()
            if name not in exclude and s.breaker.allow()
        ]

    def pick(self, exclude: set[str] = frozenset()) -> BackendState:
        """Choose a backend, skipping excluded names and open breakers.

        Note: `breaker.allow()` admits half-open probes, so a recovering
        backend naturally receives a single trial request.
        """
        candidates = self.healthy(exclude)
        if not candidates:
            raise NoHealthyBackend("no healthy backend available")
        if self.strategy == "round_robin":
            self._rr_index = (self._rr_index + 1) % len(candidates)
            return candidates[self._rr_index]
        if self.strategy == "least_outstanding":
            return min(candidates, key=lambda s: s.outstanding)
        # p2c_ewma (default)
        if len(candidates) == 1:
            return candidates[0]
        a, b = self._rng.sample(candidates, 2)
        return a if a.score <= b.score else b

    def snapshot(self) -> list[dict]:
        return [
            {
                "name": name,
                "state": s.breaker.state.value,
                "ewma_latency_ms": round(s.ewma_latency_ms, 2),
                "outstanding": s.outstanding,
                "total_requests": s.total_requests,
                "total_failures": s.total_failures,
            }
            for name, s in self.states.items()
        ]
