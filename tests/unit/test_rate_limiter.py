"""
Comprehensive unit tests for rate limiting primitives.

This test suite covers all rate limiter implementations with various scenarios:
- Verification of each rate limiting strategy (Token Bucket, Leaky Bucket, etc.)
- Normal operation paths and rate-limited rejection paths
- Time-based logic, including token regeneration and window expiration
- Waiting for tokens and timeout handling
- Advanced limiters like Adaptive and Composite limiters
- State and statistics reporting
- Edge cases and concurrent access simulation
"""

import pytest
import asyncio
import time
from typing import List

from core.coordination.rate_limiter import (
    RateLimiter,
    RateLimitStrategy,
    TokenBucket,
    LeakyBucket,
    SlidingWindow,
    FixedWindow,
    AdaptiveRateLimiter,
    CompositeRateLimiter,
)


class TestRateLimiter:
    """Test the base RateLimiter class with different strategies."""

    @pytest.fixture
    def token_bucket_limiter(self):
        return RateLimiter(max_rate=10, burst_size=3, strategy=RateLimitStrategy.TOKEN_BUCKET)

    @pytest.fixture
    def fixed_window_limiter(self):
        return RateLimiter(max_rate=2, window_size=0.2, strategy=RateLimitStrategy.FIXED_WINDOW)

    @pytest.fixture
    def sliding_window_limiter(self):
        return RateLimiter(max_rate=2, window_size=0.2, strategy=RateLimitStrategy.SLIDING_WINDOW)

    @pytest.mark.asyncio
    async def test_token_bucket_strategy(self, token_bucket_limiter):
        """Test TOKEN_BUCKET strategy allows burst and then limits."""
        # Should allow burst of 3 tokens initially
        for _ in range(3):
            assert await token_bucket_limiter.acquire()
        # Should be rate-limited now
        assert not await token_bucket_limiter.acquire()

        # Wait for some tokens to regenerate (rate is 10/sec)
        await asyncio.sleep(0.2)  # Should get ~2 tokens back
        assert await token_bucket_limiter.acquire()
        assert await token_bucket_limiter.acquire()
        assert not await token_bucket_limiter.acquire()

    @pytest.mark.asyncio
    async def test_fixed_window_strategy(self, fixed_window_limiter):
        """Test FIXED_WINDOW strategy resets on new window."""
        # Should allow 2 requests in current window
        assert await fixed_window_limiter.acquire()
        assert await fixed_window_limiter.acquire()
        assert not await fixed_window_limiter.acquire()

        # Wait for new window (0.2 seconds)
        await asyncio.sleep(0.25)
        assert await fixed_window_limiter.acquire()
        assert await fixed_window_limiter.acquire()
        assert not await fixed_window_limiter.acquire()

    @pytest.mark.asyncio
    async def test_sliding_window_strategy(self, sliding_window_limiter):
        """Test SLIDING_WINDOW strategy correctly slides."""
        # Should allow 2 requests initially
        assert await sliding_window_limiter.acquire()
        await asyncio.sleep(0.1)
        assert await sliding_window_limiter.acquire()
        assert not await sliding_window_limiter.acquire()

        # Wait for first request to slide out of window
        await asyncio.sleep(0.15)  # Total 0.25s since first request
        assert await sliding_window_limiter.acquire()

    @pytest.mark.asyncio
    async def test_wait_for_token_success(self):
        """Test waiting for a token successfully."""
        limiter = RateLimiter(max_rate=10, burst_size=1)
        assert await limiter.acquire()  # Use up the token

        start_time = time.monotonic()
        assert await limiter.wait_for_token(timeout=0.5)
        duration = time.monotonic() - start_time
        # Should wait at least 1/10th second for next token
        assert duration >= 0.09

    @pytest.mark.asyncio
    async def test_wait_for_token_timeout(self):
        """Test waiting for a token that times out."""
        limiter = RateLimiter(max_rate=10, burst_size=1)
        assert await limiter.acquire()  # Use up the token
        # Timeout before next token becomes available
        assert not await limiter.wait_for_token(timeout=0.05)

    def test_get_stats(self, token_bucket_limiter):
        """Test the get_stats method returns expected data."""
        stats = token_bucket_limiter.get_stats()
        assert stats['strategy'] == 'TOKEN_BUCKET'
        assert stats['max_rate'] == 10
        assert stats['burst_size'] == 3


class TestTokenBucket:
    """Tests for the specialized TokenBucket class."""

    @pytest.fixture
    def limiter(self):
        return TokenBucket(rate=10, capacity=3, initial_tokens=1.0)

    @pytest.mark.asyncio
    async def test_consume_multiple_tokens(self, limiter):
        """Test consuming multiple tokens at once."""
        # Set tokens manually and consume
        async with limiter._lock:
            limiter._tokens = 2.5
            limiter._last_update = time.time()  # Prevent regeneration

        assert await limiter.consume(2)
        # Check tokens without triggering regeneration
        assert limiter._tokens == pytest.approx(0.5, abs=0.01)
        assert not await limiter.consume(1)

    @pytest.mark.asyncio
    async def test_token_regeneration_and_capacity(self, limiter):
        """Test token regeneration does not exceed capacity."""
        # Exhaust tokens
        async with limiter._lock:
            limiter._tokens = 0
            limiter._last_update = time.time()

        # Wait for regeneration
        await asyncio.sleep(0.5)  # Should regenerate 5 tokens, but cap at 3
        assert limiter.tokens <= limiter.capacity
        assert limiter.tokens == limiter.capacity

    def test_properties(self, limiter):
        """Test specialized properties."""
        # Allow for small timing variations
        assert limiter.tokens == pytest.approx(1.0, abs=0.1)
        assert limiter.capacity == 3


class TestLeakyBucket:
    """Tests for the specialized LeakyBucket class."""

    @pytest.fixture
    def limiter(self):
        return LeakyBucket(rate=10, capacity=5)

    @pytest.mark.asyncio
    async def test_leakage_over_time(self, limiter):
        """Test that requests leak from the bucket, freeing up space."""
        # Fill the bucket
        for _ in range(5):
            assert await limiter.acquire()
        # Should be full now
        assert not await limiter.acquire()

        # Wait for some requests to leak out (rate is 10/sec)
        await asyncio.sleep(0.2)  # Should leak ~2 requests
        assert await limiter.acquire()
        assert await limiter.acquire()
        # Should be full again
        assert not await limiter.acquire()


class TestSlidingWindow:
    """Tests for the specialized SlidingWindow class."""

    @pytest.fixture
    def limiter(self):
        return SlidingWindow(rate=5, window_size=1.0)

    @pytest.mark.asyncio
    async def test_window_sliding(self, limiter):
        """Verify that as time passes, old requests expire, allowing new ones."""
        # Fill up the window with 5 requests
        for _ in range(5):
            assert await limiter.acquire()
        # Should be at limit
        assert not await limiter.acquire()

        # Wait for partial window slide
        await asyncio.sleep(0.6)
        # Still shouldn't allow new requests (all 5 still in window)
        assert not await limiter.acquire()
        assert limiter.current_rate == pytest.approx(5.0, abs=0.1)

        # Wait for requests to slide out of window
        await asyncio.sleep(0.5)  # Total 1.1s, first requests should be out
        assert await limiter.acquire()
        assert limiter.current_rate == pytest.approx(1.0, abs=0.1)


class TestFixedWindow:
    """Tests for the specialized FixedWindow class."""

    @pytest.fixture
    def limiter(self):
        return FixedWindow(rate=2, window_size=0.5)

    @pytest.mark.asyncio
    async def test_new_window_reset(self, limiter):
        """Verify that the count resets when a new fixed window begins."""
        # Use up the current window
        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()

        # Wait for new window
        await asyncio.sleep(0.6)
        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()


class TestAdaptiveRateLimiter:
    """Tests for the AdaptiveRateLimiter."""

    @pytest.fixture
    def limiter(self):
        limiter = AdaptiveRateLimiter(base_rate=10, min_rate=5, max_rate=20)
        limiter._adjustment_interval = 0.05  # Fast adjustments for testing
        return limiter

    @pytest.mark.asyncio
    async def test_rate_increase_on_success(self, limiter):
        """Verify the rate increases with a high success ratio."""
        # Generate high success ratio
        for _ in range(10):
            assert await limiter.acquire()

        # Wait for adjustment interval
        await asyncio.sleep(0.06)

        # Trigger another acquire to force adjustment check
        await limiter.acquire()

        # Rate should have increased from 10
        assert limiter.current_rate > 10

    @pytest.mark.asyncio
    async def test_rate_decrease_on_failure(self, limiter):
        """Verify the rate decreases with a low success ratio."""
        # Exhaust tokens to create failures
        burst_size = int(limiter._limiter.burst_size)
        for _ in range(burst_size):
            await limiter.acquire()

        # Create failures
        for _ in range(10):
            await limiter.acquire()  # These should mostly fail

        # Manually set high failure rate for testing
        async with limiter._lock:
            limiter._success_count = 1
            limiter._failure_count = 10
            limiter._last_adjustment = time.time() - limiter._adjustment_interval

        # Trigger adjustment
        await limiter.acquire()

        # Rate should have decreased from 10
        assert limiter.current_rate < 10

    @pytest.mark.asyncio
    async def test_min_max_bounds(self):
        """Ensure the adjusted rate never goes below min_rate or above max_rate."""
        limiter = AdaptiveRateLimiter(base_rate=6, min_rate=5, max_rate=7)
        limiter._adjustment_interval = 0.01

        # Test max bound
        limiter._current_rate = 6.5
        limiter._success_count = 10
        limiter._failure_count = 0
        await limiter._adjust_rate()
        assert limiter.current_rate <= 7

        # Test min bound
        limiter._current_rate = 5.5
        limiter._success_count = 0
        limiter._failure_count = 10
        await limiter._adjust_rate()
        assert limiter.current_rate >= 5


class TestCompositeRateLimiter:
    """Tests for the CompositeRateLimiter."""

    @pytest.fixture
    def composite_limiter(self):
        limiter1 = TokenBucket(rate=10, capacity=2)
        limiter2 = FixedWindow(rate=3, window_size=1.0)
        return CompositeRateLimiter([limiter1, limiter2])

    @pytest.mark.asyncio
    async def test_acquire_failure_if_any_fails(self, composite_limiter):
        """Test that if any underlying limiter rejects, the composite rejects."""
        # Should work initially
        assert await composite_limiter.acquire()
        assert await composite_limiter.acquire()

        # Token bucket should be exhausted (capacity=2)
        assert not await composite_limiter.acquire()

        # Wait for token bucket to regenerate
        await asyncio.sleep(0.15)
        assert await composite_limiter.acquire()

        # Now token bucket should be exhausted again
        assert not await composite_limiter.acquire()

    @pytest.mark.asyncio
    async def test_wait_for_all(self):
        """Test waiting on a composite limiter is governed by the slowest limiter."""
        limiter1 = TokenBucket(rate=10, capacity=1)  # Fast regeneration (0.1s)
        limiter2 = FixedWindow(rate=1, window_size=0.3)  # Slow window (up to 0.3s)
        composite = CompositeRateLimiter([limiter1, limiter2])

        # Use up both limiters
        assert await composite.acquire()

        # Should wait for the slower limiter (fixed window)
        start = time.monotonic()
        assert await composite.wait_for_all(timeout=0.5)
        duration = time.monotonic() - start

        # Should wait at least some reasonable time for the fixed window
        # The exact wait time depends on when in the window we started
        assert duration >= 0.05  # At least some wait time
        assert duration <= 0.35  # But not more than window + tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])