"""Per-backend circuit breaker.

State machine: CLOSED -> OPEN (after `failure_threshold` consecutive
failures) -> HALF_OPEN (after `recovery_seconds`) -> CLOSED on a
successful probe, or back to OPEN on a failed one.

The clock is injectable so tests never have to sleep.
"""
from __future__ import annotations

import enum
import time
from typing import Callable


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> State:
        self._maybe_transition_to_half_open()
        return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if self._state is State.OPEN:
            if self._clock() - self._opened_at >= self.recovery_seconds:
                self._state = State.HALF_OPEN
                self._probe_in_flight = False

    def allow(self) -> bool:
        """May a request be sent to this backend right now?"""
        self._maybe_transition_to_half_open()
        if self._state is State.CLOSED:
            return True
        if self._state is State.HALF_OPEN and not self._probe_in_flight:
            # Admit exactly one probe request while half-open.
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._probe_in_flight = False
        self._state = State.CLOSED

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._state is State.HALF_OPEN or (
            self._consecutive_failures >= self.failure_threshold
        ):
            self._trip()

    def _trip(self) -> None:
        self._state = State.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False
