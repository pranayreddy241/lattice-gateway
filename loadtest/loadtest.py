"""Closed-loop load test against the in-process ASGI app.

Runs N concurrent workers issuing completions against mock backends and
reports throughput plus latency percentiles. A fraction of prompts are
repeats so the semantic cache gets exercised realistically.

Usage: python loadtest/loadtest.py [--workers 64] [--duration 15]
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx

from lattice.backends import MockBackend
from lattice.server import GatewayConfig, create_app

def make_pool(size: int) -> list[str]:
    return [f"question number {i} about distributed systems" for i in range(size)]


async def worker(
    client: httpx.AsyncClient,
    stop_at: float,
    latencies: list[float],
    errors: list[int],
    cache_hits: list[int],
    repeat_fraction: float,
    pool: list[str],
    temperature: float,
    rng: random.Random,
) -> None:
    while time.monotonic() < stop_at:
        if rng.random() < repeat_fraction:
            prompt = rng.choice(pool[:20])  # hot set -> cache hits
        else:
            prompt = rng.choice(pool)
        start = time.monotonic()
        try:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": prompt, "temperature": temperature},
                headers={"x-api-key": f"lt-{rng.randint(0, 7)}"},
            )
            elapsed = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed)
                if resp.json()["cached"]:
                    cache_hits.append(1)
            else:
                errors.append(resp.status_code)
        except Exception:
            errors.append(-1)


def pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * p))
    return sorted_values[idx]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--repeat-fraction", type=float, default=0.3)
    parser.add_argument("--pool-size", type=int, default=100_000,
                        help="unique prompts; large pool = cold cache")
    parser.add_argument("--no-cache", action="store_true",
                        help="send temperature=0.7 so nothing is cacheable")
    args = parser.parse_args()
    pool = make_pool(args.pool_size)
    temperature = 0.7 if args.no_cache else 0.0

    backends = [
        MockBackend(name="gpu-a", base_latency_ms=30, jitter_ms=10),
        MockBackend(name="gpu-b", base_latency_ms=45, jitter_ms=15),
        MockBackend(name="gpu-c", base_latency_ms=60, jitter_ms=20, fail_rate=0.02),
    ]
    config = GatewayConfig(rate_per_second=10_000, burst=20_000)
    app = create_app(backends, config)
    transport = httpx.ASGITransport(app=app)

    latencies: list[float] = []
    errors: list[int] = []
    cache_hits: list[int] = []
    rng = random.Random(7)

    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        stop_at = time.monotonic() + args.duration
        start = time.monotonic()
        await asyncio.gather(
            *(
                worker(client, stop_at, latencies, errors, cache_hits,
                       args.repeat_fraction, pool, temperature, rng)
                for _ in range(args.workers)
            )
        )
        wall = time.monotonic() - start
        stats = (await client.get("/v1/backends")).json()

    latencies.sort()
    total = len(latencies) + len(errors)
    print(f"requests      : {total} ({len(errors)} errors)")
    print(f"throughput    : {len(latencies) / wall:.0f} req/s sustained")
    print(f"latency p50   : {pct(latencies, 0.50):.1f} ms")
    print(f"latency p95   : {pct(latencies, 0.95):.1f} ms")
    print(f"latency p99   : {pct(latencies, 0.99):.1f} ms")
    print(f"latency mean  : {statistics.fmean(latencies):.1f} ms")
    print(f"cache hits    : {len(cache_hits)} "
          f"({len(cache_hits) / max(1, len(latencies)):.1%} of successes)")
    for b in stats["backends"]:
        print(f"backend {b['name']:6s}: {b['total_requests']} reqs, "
              f"{b['total_failures']} failures, "
              f"ewma {b['ewma_latency_ms']} ms, state={b['state']}")


if __name__ == "__main__":
    asyncio.run(main())
