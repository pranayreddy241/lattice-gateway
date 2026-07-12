"""FastAPI gateway: auth -> rate limit -> cache -> route -> batch -> failover.

Request lifecycle for POST /v1/completions:
  1. rate-limit by API key (429 on rejection)
  2. semantic cache lookup (hit returns immediately, cached=true)
  3. router picks a healthy backend (P2C over EWMA latency)
  4. request rides a micro-batch to the backend
  5. on failure, retry on a *different* backend (up to max_retries),
     while the circuit breaker records the failure
  6. successful responses populate the cache
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import metrics
from .backends import Backend, BackendError, CompletionRequest, MockBackend
from .batcher import MicroBatcher
from .breaker import State
from .cache import SemanticCache
from .ratelimit import RateLimiter
from .router import NoHealthyBackend, Router

_BREAKER_GAUGE_VALUE = {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}


@dataclasses.dataclass
class GatewayConfig:
    strategy: str = "p2c_ewma"
    max_retries: int = 2
    rate_per_second: float = 50.0
    burst: int = 100
    cache_ttl_seconds: float = 300.0
    cache_threshold: float = 0.95
    max_batch_size: int = 8
    max_batch_wait_ms: float = 8.0


class CompletionIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_768)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class Gateway:
    def __init__(self, backends: list[Backend], config: GatewayConfig) -> None:
        self.config = config
        self.router = Router(backends, strategy=config.strategy)
        self.cache = SemanticCache(
            ttl_seconds=config.cache_ttl_seconds, threshold=config.cache_threshold
        )
        self.limiter = RateLimiter(config.rate_per_second, config.burst)
        self.batchers = {
            b.name: MicroBatcher(
                b.generate_batch, config.max_batch_size, config.max_batch_wait_ms
            )
            for b in backends
        }

    async def complete(self, request: CompletionRequest):
        start = time.monotonic()

        if not self.limiter.allow(request.api_key or "anonymous"):
            metrics.RATE_LIMITED.inc()
            metrics.REQUESTS.labels(outcome="rate_limited").inc()
            raise HTTPException(status_code=429, detail="rate limit exceeded")

        if request.temperature == 0.0:  # only deterministic requests are cacheable
            cached = self.cache.get(request.prompt)
            if cached is not None:
                metrics.CACHE_HITS.inc()
                metrics.REQUESTS.labels(outcome="cache_hit").inc()
                response = dataclasses.replace(cached, cached=True)
                response.latency_ms = (time.monotonic() - start) * 1000
                return response
            metrics.CACHE_MISSES.inc()

        tried: set[str] = set()
        last_error: Exception | None = None
        for _ in range(1 + self.config.max_retries):
            try:
                state = self.router.pick(exclude=tried)
            except NoHealthyBackend:
                break
            name = state.backend.name
            tried.add(name)
            state.on_dispatch()
            attempt_start = time.monotonic()
            try:
                response = await self.batchers[name].submit(request)
            except BackendError as exc:
                state.on_complete((time.monotonic() - attempt_start) * 1000, ok=False)
                metrics.BACKEND_REQUESTS.labels(backend=name, outcome="error").inc()
                self._export_breaker_state(name)
                last_error = exc
                continue
            latency_ms = (time.monotonic() - attempt_start) * 1000
            state.on_complete(latency_ms, ok=True)
            metrics.BACKEND_REQUESTS.labels(backend=name, outcome="ok").inc()
            self._export_breaker_state(name)
            if request.temperature == 0.0:
                self.cache.put(request.prompt, response)
            response.latency_ms = (time.monotonic() - start) * 1000
            metrics.REQUESTS.labels(outcome="ok").inc()
            metrics.LATENCY.observe(time.monotonic() - start)
            return response

        metrics.REQUESTS.labels(outcome="unavailable").inc()
        detail = f"all backends unavailable ({last_error})" if last_error else \
            "all backends unavailable"
        raise HTTPException(status_code=503, detail=detail)

    def _export_breaker_state(self, name: str) -> None:
        state = self.router.states[name].breaker.state
        metrics.BREAKER_STATE.labels(backend=name).set(_BREAKER_GAUGE_VALUE[state])

    async def close(self) -> None:
        for batcher in self.batchers.values():
            await batcher.close()


def create_app(
    backends: list[Backend] | None = None, config: GatewayConfig | None = None
) -> FastAPI:
    config = config or GatewayConfig()
    backends = backends or [
        MockBackend(name="mock-a", base_latency_ms=15),
        MockBackend(name="mock-b", base_latency_ms=25),
    ]
    gateway = Gateway(backends, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await gateway.close()

    app = FastAPI(title="Lattice", version="0.3.0", lifespan=lifespan)
    app.state.gateway = gateway

    @app.post("/v1/completions")
    async def completions(body: CompletionIn, http_request: Request):
        api_key = http_request.headers.get("x-api-key", "anonymous")
        request = CompletionRequest(
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            api_key=api_key,
            request_id=str(uuid.uuid4()),
        )
        response = await gateway.complete(request)
        return {
            "text": response.text,
            "backend": response.backend,
            "cached": response.cached,
            "latency_ms": round(response.latency_ms, 2),
        }

    @app.get("/healthz")
    async def healthz():
        healthy = gateway.router.healthy()
        status = "ok" if healthy else "degraded"
        return {"status": status, "healthy_backends": [s.backend.name for s in healthy]}

    @app.get("/v1/backends")
    async def backend_stats():
        return {
            "backends": gateway.router.snapshot(),
            "cache": {
                "entries": len(gateway.cache),
                "hit_rate": round(gateway.cache.hit_rate, 4),
            },
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(
            content=metrics.generate_latest(), media_type=metrics.CONTENT_TYPE_LATEST
        )

    return app
