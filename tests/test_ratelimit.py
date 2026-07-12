from lattice.ratelimit import RateLimiter, TokenBucket


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_burst_then_reject():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_second=1, burst=3, clock=clock)
    assert all(bucket.try_acquire() for _ in range(3))
    assert not bucket.try_acquire()


def test_refill_over_time():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_second=2, burst=2, clock=clock)
    bucket.try_acquire()
    bucket.try_acquire()
    assert not bucket.try_acquire()
    clock.now = 1.0  # 2 tokens refilled
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()


def test_refill_capped_at_capacity():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_second=100, burst=5, clock=clock)
    clock.now = 100.0
    assert bucket.available == 5.0


def test_keys_are_isolated():
    clock = FakeClock()
    limiter = RateLimiter(rate_per_second=1, burst=1, clock=clock)
    assert limiter.allow("user-a")
    assert not limiter.allow("user-a")
    assert limiter.allow("user-b")
