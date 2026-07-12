import asyncio

import pytest

from lattice.backends import BackendError, CompletionRequest, CompletionResponse
from lattice.batcher import MicroBatcher


def req(i: int) -> CompletionRequest:
    return CompletionRequest(prompt=f"prompt {i}")


@pytest.mark.asyncio
async def test_full_batch_dispatches_together():
    seen_batches = []

    async def dispatch(batch):
        seen_batches.append(len(batch))
        return [CompletionResponse(text=r.prompt, backend="t") for r in batch]

    batcher = MicroBatcher(dispatch, max_batch_size=4, max_wait_ms=1000)
    results = await asyncio.gather(*(batcher.submit(req(i)) for i in range(4)))
    assert [r.text for r in results] == [f"prompt {i}" for i in range(4)]
    assert seen_batches == [4]
    await batcher.close()


@pytest.mark.asyncio
async def test_partial_batch_flushes_after_max_wait():
    seen_batches = []

    async def dispatch(batch):
        seen_batches.append(len(batch))
        return [CompletionResponse(text="ok", backend="t") for _ in batch]

    batcher = MicroBatcher(dispatch, max_batch_size=100, max_wait_ms=20)
    result = await asyncio.wait_for(batcher.submit(req(0)), timeout=2.0)
    assert result.text == "ok"
    assert seen_batches == [1]
    await batcher.close()


@pytest.mark.asyncio
async def test_oversized_load_splits_into_multiple_batches():
    async def dispatch(batch):
        return [CompletionResponse(text="ok", backend="t") for _ in batch]

    batcher = MicroBatcher(dispatch, max_batch_size=3, max_wait_ms=5)
    await asyncio.gather(*(batcher.submit(req(i)) for i in range(10)))
    assert batcher.requests_dispatched == 10
    assert batcher.batches_dispatched >= 4  # ceil(10/3)
    await batcher.close()


@pytest.mark.asyncio
async def test_dispatch_failure_propagates_to_all_waiters():
    async def dispatch(batch):
        raise BackendError("boom")

    batcher = MicroBatcher(dispatch, max_batch_size=2, max_wait_ms=5)
    results = await asyncio.gather(
        batcher.submit(req(0)), batcher.submit(req(1)), return_exceptions=True
    )
    assert all(isinstance(r, BackendError) for r in results)
    await batcher.close()
