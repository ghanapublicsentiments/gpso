"""Tests for utility functions."""

from pipeline.utils import RateLimiter, process_in_parallel


class TestRateLimiter:
    """Test rate limiter."""

    def test_rate_limiter(self):
        """Test rate limiting."""
        limiter = RateLimiter(max_requests_per_minute=120)
        limiter.wait_if_needed()
        assert len(limiter.requests) <= 120


class TestProcessInParallel:
    """Test parallel processing."""

    def test_process_in_parallel(self):
        """Test parallel processing."""
        func = lambda x: x * 2
        items = [1, 2, 3]
        results = process_in_parallel(items, func, max_workers=2)
        assert results == [2, 4, 6]
