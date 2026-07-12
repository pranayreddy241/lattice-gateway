import random

import pytest

from lattice.backends import MockBackend
from lattice.breaker import CircuitBreaker
from lattice.router import NoHealthyBackend, Router


def make_router(n=3, strategy="p2c_ewma"):
    backends = [MockBackend(name=f"b{i}") for i in range(n)]
    return Router(backends, strategy=strategy, rng=random.Random(42))


def test_p2c_prefers_lower_latency_backend():
    router = make_router(2)
    router.states["b0"].ewma_latency_ms = 10.0
    router.states["b1"].ewma_latency_ms = 200.0
    picks = [router.pick().backend.name for _ in range(50)]
    assert picks.count("b0") == 50  # with 2 candidates P2C compares both


def test_outstanding_requests_shift_traffic():
    router = make_router(2)
    router.states["b0"].ewma_latency_ms = 50.0
    router.states["b1"].ewma_latency_ms = 50.0
    router.states["b0"].outstanding = 20
    assert router.pick().backend.name == "b1"


def test_open_breaker_removes_backend_from_rotation():
    router = make_router(2)
    for _ in range(CircuitBreaker().failure_threshold):
        router.states["b0"].on_complete(latency_ms=5, ok=False)
    picks = {router.pick().backend.name for _ in range(20)}
    assert picks == {"b1"}


def test_all_breakers_open_raises():
    router = make_router(1)
    for _ in range(CircuitBreaker().failure_threshold):
        router.states["b0"].on_complete(latency_ms=5, ok=False)
    with pytest.raises(NoHealthyBackend):
        router.pick()


def test_exclude_prevents_retry_on_same_backend():
    router = make_router(2)
    first = router.pick().backend.name
    second = router.pick(exclude={first}).backend.name
    assert second != first


def test_ewma_updates_on_success():
    router = make_router(1)
    state = router.states["b0"]
    before = state.ewma_latency_ms
    state.on_dispatch()
    state.on_complete(latency_ms=500.0, ok=True)
    assert state.ewma_latency_ms > before
