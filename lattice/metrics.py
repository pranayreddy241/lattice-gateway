"""Prometheus metrics with a no-op fallback.

If prometheus_client is missing, the gateway still runs — metrics
become no-ops instead of import errors.
"""
from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover
    HAVE_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain"

    class _Noop:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            pass

        def observe(self, *a, **k):
            pass

        def set(self, *a, **k):
            pass

    def Counter(*a, **k):  # type: ignore
        return _Noop()

    def Gauge(*a, **k):  # type: ignore
        return _Noop()

    def Histogram(*a, **k):  # type: ignore
        return _Noop()

    def generate_latest():  # type: ignore
        return b"# prometheus_client not installed\n"


REQUESTS = Counter(
    "lattice_requests_total", "Requests received", ["outcome"]
)
BACKEND_REQUESTS = Counter(
    "lattice_backend_requests_total", "Requests per backend", ["backend", "outcome"]
)
LATENCY = Histogram(
    "lattice_request_latency_seconds",
    "End-to-end request latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
CACHE_HITS = Counter("lattice_cache_hits_total", "Semantic cache hits")
CACHE_MISSES = Counter("lattice_cache_misses_total", "Semantic cache misses")
RATE_LIMITED = Counter("lattice_rate_limited_total", "Requests rejected by rate limiter")
BREAKER_STATE = Gauge(
    "lattice_breaker_state", "0=closed 1=half_open 2=open", ["backend"]
)
