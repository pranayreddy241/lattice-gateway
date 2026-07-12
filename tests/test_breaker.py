from lattice.breaker import CircuitBreaker, State


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


def make(threshold=3, recovery=10.0):
    clock = FakeClock()
    return CircuitBreaker(threshold, recovery, clock), clock


def test_opens_after_consecutive_failures():
    br, _ = make(threshold=3)
    for _ in range(2):
        br.record_failure()
    assert br.state is State.CLOSED
    br.record_failure()
    assert br.state is State.OPEN
    assert not br.allow()


def test_success_resets_failure_count():
    br, _ = make(threshold=3)
    br.record_failure()
    br.record_failure()
    br.record_success()
    br.record_failure()
    br.record_failure()
    assert br.state is State.CLOSED


def test_half_open_after_recovery_admits_single_probe():
    br, clock = make(threshold=1, recovery=10.0)
    br.record_failure()
    assert not br.allow()
    clock.advance(10.1)
    assert br.state is State.HALF_OPEN
    assert br.allow()  # the one probe
    assert not br.allow()  # second request while probing is rejected


def test_probe_success_closes_probe_failure_reopens():
    br, clock = make(threshold=1, recovery=10.0)
    br.record_failure()
    clock.advance(10.1)
    assert br.allow()
    br.record_success()
    assert br.state is State.CLOSED

    br.record_failure()
    clock.advance(10.1)
    assert br.allow()
    br.record_failure()
    assert br.state is State.OPEN
