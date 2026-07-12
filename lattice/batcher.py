"""Adaptive micro-batching.

Individual requests are queued per backend and flushed as a single
batched inference call when either (a) the batch is full or (b) the
oldest request has waited `max_wait_ms`. This trades a small, bounded
queueing delay for much higher backend throughput.

(This is gateway-side micro-batching; token-level continuous batching
lives inside engines like vLLM, behind this gateway.)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .backends import CompletionRequest, CompletionResponse

DispatchFn = Callable[[list[CompletionRequest]], Awaitable[list[CompletionResponse]]]


@dataclass
class _Pending:
    request: CompletionRequest
    future: asyncio.Future = field(default_factory=asyncio.Future)


class MicroBatcher:
    def __init__(
        self,
        dispatch: DispatchFn,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
        max_inflight_batches: int = 4,
    ) -> None:
        self._dispatch = dispatch
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000
        self._queue: list[_Pending] = []
        self._flush_event = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._closed = False
        self.batches_dispatched = 0
        self.requests_dispatched = 0
        # Overlap batch dispatch: awaiting each flush inline serializes
        # batches per backend (head-of-line blocking, found via load
        # testing). Flushes run as tasks, bounded by this semaphore.
        self._inflight = asyncio.Semaphore(max_inflight_batches)
        self._flush_tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.ensure_future(self._run())

    async def submit(self, request: CompletionRequest) -> CompletionResponse:
        if self._closed:
            raise RuntimeError("batcher is closed")
        self.start()
        pending = _Pending(request)
        self._queue.append(pending)
        if len(self._queue) >= self.max_batch_size:
            self._flush_event.set()
        return await pending.future

    async def _run(self) -> None:
        while not self._closed:
            while not self._queue and not self._closed:
                self._flush_event.clear()
                try:
                    await asyncio.wait_for(self._flush_event.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
            if self._closed:
                break
            # A request has arrived; give the batch max_wait to fill up.
            if len(self._queue) < self.max_batch_size:
                try:
                    self._flush_event.clear()
                    await asyncio.wait_for(
                        self._flush_event.wait(), timeout=self.max_wait_s
                    )
                except asyncio.TimeoutError:
                    pass
            batch, self._queue = (
                self._queue[: self.max_batch_size],
                self._queue[self.max_batch_size :],
            )
            if batch:
                await self._inflight.acquire()
                task = asyncio.ensure_future(self._flush(batch))
                self._flush_tasks.add(task)
                task.add_done_callback(self._flush_tasks.discard)

    async def _flush(self, batch: list[_Pending]) -> None:
        self.batches_dispatched += 1
        self.requests_dispatched += len(batch)
        try:
            responses = await self._dispatch([p.request for p in batch])
            for pending, response in zip(batch, responses):
                if not pending.future.done():
                    pending.future.set_result(response)
        except Exception as exc:  # propagate to every waiter in the batch
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)
        finally:
            self._inflight.release()

    async def close(self) -> None:
        self._closed = True
        self._flush_event.set()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        for task in list(self._flush_tasks):
            task.cancel()
