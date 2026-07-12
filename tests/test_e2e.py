"""End-to-end tests through the ASGI app: failover, caching, rate limits."""
import httpx
import pytest

from lattice.backends import MockBackend
from lattice.server import GatewayConfig, create_app


def client_for(backends, config=None):
    app = create_app(backends, config or GatewayConfig())
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://gw"), app


@pytest.mark.asyncio
async def test_completion_roundtrip():
    client, _ = client_for([MockBackend(name="a", base_latency_ms=1)])
    resp = await client.post("/v1/completions", json={"prompt": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "a"
    assert not body["cached"]
    await client.aclose()


@pytest.mark.asyncio
async def test_failover_to_healthy_backend():
    flaky = MockBackend(name="flaky", base_latency_ms=1, fail_rate=1.0)
    stable = MockBackend(name="stable", base_latency_ms=1)
    client, _ = client_for([flaky, stable])
    for _ in range(10):
        resp = await client.post("/v1/completions", json={"prompt": "q"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "stable"
    await client.aclose()


@pytest.mark.asyncio
async def test_all_backends_down_returns_503():
    dead = MockBackend(name="dead", base_latency_ms=1, fail_rate=1.0)
    client, _ = client_for([dead])
    saw_503 = False
    for _ in range(8):
        resp = await client.post("/v1/completions", json={"prompt": "q"})
        if resp.status_code == 503:
            saw_503 = True
    assert saw_503
    await client.aclose()


@pytest.mark.asyncio
async def test_semantic_cache_serves_repeat_prompt():
    backend = MockBackend(name="a", base_latency_ms=1)
    client, _ = client_for([backend])
    p = {"prompt": "explain two-phase commit", "temperature": 0.0}
    first = await client.post("/v1/completions", json=p)
    second = await client.post("/v1/completions", json=p)
    assert not first.json()["cached"]
    assert second.json()["cached"]
    assert backend.calls == 1  # second request never reached the backend
    await client.aclose()


@pytest.mark.asyncio
async def test_nonzero_temperature_bypasses_cache():
    backend = MockBackend(name="a", base_latency_ms=1)
    client, _ = client_for([backend])
    p = {"prompt": "creative story", "temperature": 0.9}
    await client.post("/v1/completions", json=p)
    resp = await client.post("/v1/completions", json=p)
    assert not resp.json()["cached"]
    assert backend.calls == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_returns_429():
    config = GatewayConfig(rate_per_second=1, burst=2)
    client, _ = client_for([MockBackend(name="a", base_latency_ms=1)], config)
    codes = []
    for i in range(5):
        r = await client.post(
            "/v1/completions",
            json={"prompt": f"unique prompt {i}", "temperature": 0.5},
            headers={"x-api-key": "user-1"},
        )
        codes.append(r.status_code)
    assert 429 in codes
    await client.aclose()


@pytest.mark.asyncio
async def test_health_and_stats_endpoints():
    client, _ = client_for([MockBackend(name="a", base_latency_ms=1)])
    health = await client.get("/healthz")
    assert health.json()["status"] == "ok"
    await client.post("/v1/completions", json={"prompt": "q"})
    stats = await client.get("/v1/backends")
    assert stats.json()["backends"][0]["total_requests"] == 1
    await client.aclose()
