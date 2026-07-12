"""Token-bucket rate limiting, one bucket per API key."""
from __future__ import annotations

import time
from typing import Callable


class TokenBucket:
    def __init__(
        self,
        rate_per_second: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = rate_per_second
        self.capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._last_refill = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(
            self.capacity, self._tokens + (now - self._last_refill) * self.rate
        )
        self._last_refill = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


class RateLimiter:
    def __init__(
        self,
        rate_per_second: float = 20.0,
        burst: int = 40,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = rate_per_second
        self.burst = burst
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, api_key: str) -> bool:
        bucket = self._buckets.get(api_key)
        if bucket is None:
            bucket = self._buckets[api_key] = TokenBucket(
                self.rate, self.burst, self._clock
            )
        return bucket.try_acquire()
