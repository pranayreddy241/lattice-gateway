"""Backend abstractions.

A Backend turns a batch of prompts into a batch of completions.
`MockBackend` simulates latency and failures for tests and load tests;
`HTTPBackend` speaks the OpenAI-compatible /v1/completions protocol.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    api_key: str = ""
    request_id: str = ""


@dataclass
class CompletionResponse:
    text: str
    backend: str
    tokens: int = 0
    cached: bool = False
    latency_ms: float = 0.0


class BackendError(RuntimeError):
    """Raised when a backend fails to serve a batch."""


class Backend(Protocol):
    name: str

    async def generate_batch(
        self, requests: list[CompletionRequest]
    ) -> list[CompletionResponse]: ...


@dataclass
class MockBackend:
    """Deterministic-ish fake model server for tests and load tests."""

    name: str = "mock"
    base_latency_ms: float = 20.0
    jitter_ms: float = 5.0
    fail_rate: float = 0.0
    rng: random.Random = field(default_factory=random.Random)
    calls: int = 0

    async def generate_batch(
        self, requests: list[CompletionRequest]
    ) -> list[CompletionResponse]:
        self.calls += 1
        # Batched inference amortizes cost: latency grows sub-linearly
        # with batch size rather than multiplying by it.
        latency = self.base_latency_ms + self.rng.uniform(0, self.jitter_ms)
        latency *= 1 + 0.1 * (len(requests) - 1)
        await asyncio.sleep(latency / 1000)
        if self.rng.random() < self.fail_rate:
            raise BackendError(f"{self.name}: simulated failure")
        return [
            CompletionResponse(
                text=f"[{self.name}] completion for: {r.prompt[:40]}",
                backend=self.name,
                tokens=min(r.max_tokens, len(r.prompt.split()) + 8),
            )
            for r in requests
        ]


class HTTPBackend:
    """OpenAI-compatible HTTP backend (vLLM, TGI, llama.cpp server...)."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def generate_batch(
        self, requests: list[CompletionRequest]
    ) -> list[CompletionResponse]:
        # OpenAI-compatible servers accept a list of prompts in one call.
        payload = {
            "model": self.model,
            "prompt": [r.prompt for r in requests],
            "max_tokens": max(r.max_tokens for r in requests),
            "temperature": requests[0].temperature,
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/completions", json=payload
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name}: {exc}") from exc
        body = resp.json()
        choices = sorted(body.get("choices", []), key=lambda c: c.get("index", 0))
        if len(choices) != len(requests):
            raise BackendError(
                f"{self.name}: expected {len(requests)} choices, got {len(choices)}"
            )
        return [
            CompletionResponse(
                text=c.get("text", ""),
                backend=self.name,
                tokens=body.get("usage", {}).get("completion_tokens", 0),
            )
            for c in choices
        ]

    async def aclose(self) -> None:
        await self._client.aclose()
