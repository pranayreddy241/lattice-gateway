from lattice.cache import SemanticCache, cosine, embed


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_exact_hit():
    cache = SemanticCache()
    cache.put("what is a mutex", "answer")
    assert cache.get("what is a mutex") == "answer"


def test_near_duplicate_hits():
    cache = SemanticCache(threshold=0.90)
    cache.put("What is a mutex in operating systems?", "answer")
    assert cache.get("what is a mutex in operating systems") == "answer"


def test_unrelated_prompt_misses():
    cache = SemanticCache()
    cache.put("what is a mutex", "answer")
    assert cache.get("recipe for banana bread") is None


def test_ttl_expiry():
    clock = FakeClock()
    cache = SemanticCache(ttl_seconds=60, clock=clock)
    cache.put("prompt", "answer")
    clock.now = 61.0
    assert cache.get("prompt") is None


def test_lru_eviction_keeps_recently_used():
    cache = SemanticCache(max_entries=2, threshold=0.99)
    cache.put("alpha alpha alpha", 1)
    cache.put("beta beta beta", 2)
    assert cache.get("alpha alpha alpha") == 1  # touch alpha
    cache.put("gamma gamma gamma", 3)  # evicts beta (LRU)
    assert cache.get("beta beta beta") is None
    assert cache.get("alpha alpha alpha") == 1


def test_embedding_is_normalized_and_stable():
    v1, v2 = embed("hello world"), embed("hello world")
    assert v1 == v2
    assert abs(cosine(v1, v2) - 1.0) < 1e-9


def test_hit_rate_accounting():
    cache = SemanticCache()
    cache.put("p", "a")
    cache.get("p")
    cache.get("something else entirely")
    assert 0.0 < cache.hit_rate < 1.0
