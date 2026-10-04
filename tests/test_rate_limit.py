from financial_rag.ingestion.rate_limit import RateLimiter


def test_rate_limiter_waits_for_minimum_interval() -> None:
    now = [10.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2.0, clock=clock, sleep=sleep)
    limiter.wait()
    limiter.wait()

    assert sleeps == [0.5]
