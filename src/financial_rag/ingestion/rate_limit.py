"""Small deterministic request rate limiter for SEC traffic."""

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Enforce a minimum interval between requests across threads."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            message = "requests_per_second must be positive"
            raise ValueError(message)
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_request_at: float | None = None

    def wait(self) -> None:
        """Block until the next request is permitted."""

        with self._lock:
            now = self._clock()
            if self._last_request_at is not None:
                remaining = self._interval - (now - self._last_request_at)
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last_request_at = now
