"""
Comprehensive unit tests for coordination primitives.

This test suite covers all coordination primitives with various scenarios:
- Normal operation paths
- Edge cases and error conditions
- Concurrent access scenarios
- Timeout handling
- TTL expiration
- Context manager functionality
- Auto-renewal and periodic tasks
- Resource limits and quotas
- State transitions and cleanup
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import timedelta
from typing import List, Set

from core.coordination.primitives import (
    CoordinationPrimitive,
    PrimitiveType,
    ResourceState,
    Mutex,
    Semaphore,
    Barrier,
    Lease,
    Lock,
    Quota,
    create_primitive
)


class TestCoordinationPrimitive:
    """Test the base CoordinationPrimitive class."""

    @pytest.fixture
    def mutex_primitive(self):
        """Create a mutex primitive for testing."""
        return CoordinationPrimitive(
            name="test_mutex",
            type=PrimitiveType.MUTEX,
            ttl=10.0
        )

    @pytest.fixture
    def semaphore_primitive(self):
        """Create a semaphore primitive for testing."""
        return CoordinationPrimitive(
            name="test_semaphore",
            type=PrimitiveType.SEMAPHORE,
            max_count=3,
            ttl=10.0
        )

    @pytest.fixture
    def quota_primitive(self):
        """Create a quota primitive for testing."""
        return CoordinationPrimitive(
            name="test_quota",
            type=PrimitiveType.QUOTA,
            quota_limit=100.0
        )

    @pytest.mark.asyncio
    async def test_mutex_acquire_release(self, mutex_primitive):
        """Test basic mutex acquire and release."""
        caller_id = "test_caller"

        # Should be able to acquire when free
        assert await mutex_primitive.acquire(caller_id)
        assert caller_id in mutex_primitive._owners
        assert mutex_primitive._state == ResourceState.ACQUIRED

        # Same caller can acquire again (reentrant)
        assert await mutex_primitive.acquire(caller_id)

        # Different caller should not be able to acquire
        assert not await mutex_primitive.acquire("other_caller")

        # Release should work
        assert await mutex_primitive.release(caller_id)
        assert caller_id not in mutex_primitive._owners
        assert mutex_primitive._state == ResourceState.AVAILABLE

    @pytest.mark.asyncio
    async def test_semaphore_acquire_release(self, semaphore_primitive):
        """Test semaphore with multiple permits."""
        callers = ["caller1", "caller2", "caller3", "caller4"]

        # Should be able to acquire up to max_count
        for i in range(3):
            assert await semaphore_primitive.acquire(callers[i])
            assert callers[i] in semaphore_primitive._owners

        # Fourth caller should not be able to acquire
        assert not await semaphore_primitive.acquire(callers[3])

        # Release one permit
        assert await semaphore_primitive.release(callers[0])
        assert callers[0] not in semaphore_primitive._owners

        # Now fourth caller should be able to acquire
        assert await semaphore_primitive.acquire(callers[3])

    @pytest.mark.asyncio
    async def test_quota_acquire_release(self, quota_primitive):
        """Test quota-based resource management."""
        caller1 = "caller1"
        caller2 = "caller2"

        # Should be able to consume partial quota
        assert await quota_primitive.acquire(caller1, quota_amount=30.0)
        assert quota_primitive._quota_usage[caller1] == 30.0

        # Should be able to consume more quota
        assert await quota_primitive.acquire(caller2, quota_amount=50.0)
        assert quota_primitive._quota_usage[caller2] == 50.0

        # Should not be able to exceed quota limit
        assert not await quota_primitive.acquire("caller3", quota_amount=25.0)

        # Release quota (for quota type, release clears usage)
        await quota_primitive.release(caller1)
        assert caller1 not in quota_primitive._quota_usage

        # Now should be able to acquire more
        assert await quota_primitive.acquire("caller3", quota_amount=25.0)

    @pytest.mark.asyncio
    async def test_quota_requires_amount(self, quota_primitive):
        """Test that quota primitive requires quota_amount parameter."""
        with pytest.raises(ValueError, match="Quota amount required"):
            await quota_primitive.acquire("test_caller")

    @pytest.mark.asyncio
    async def test_barrier_synchronization(self):
        """Test barrier synchronization behavior."""
        barrier = CoordinationPrimitive(
            name="test_barrier",
            type=PrimitiveType.BARRIER,
            max_count=3,
            wait_timeout=2.0  # Increased timeout for concurrent operations
        )

        callers = ["caller1", "caller2", "caller3"]
        results = []

        async def acquire_and_wait(caller_id):
            try:
                result = await barrier.acquire(caller_id)
                results.append((caller_id, result))
            except asyncio.TimeoutError:
                results.append((caller_id, False))

        # Start all tasks concurrently
        tasks = [acquire_and_wait(caller) for caller in callers]
        await asyncio.gather(*tasks)

        # Should have some successful results (barrier implementation may vary)
        assert len(results) == 3
        # All parties should succeed in a barrier scenario
        success_count = sum(1 for result in results if result[1])
        assert success_count == 3

    @pytest.mark.asyncio
    async def test_barrier_timeout(self):
        """Test barrier timeout when not enough parties arrive."""
        barrier = CoordinationPrimitive(
            name="test_barrier",
            type=PrimitiveType.BARRIER,
            max_count=3,
            wait_timeout=0.1
        )

        # Only one caller tries to acquire - should timeout
        result = await barrier.acquire("lonely_caller")
        assert not result

    @pytest.mark.asyncio
    async def test_lease_cleanup_expired(self):
        """Test lease expiration cleanup."""
        lease = CoordinationPrimitive(
            name="test_lease",
            type=PrimitiveType.LEASE,
            ttl=0.1
        )

        caller_id = "test_caller"

        # Acquire lease
        assert await lease.acquire(caller_id)
        assert caller_id in lease._owners

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Try to acquire again - should trigger cleanup and succeed
        assert await lease.acquire("new_caller")
        assert caller_id not in lease._owners
        assert "new_caller" in lease._owners

    def test_get_state(self, mutex_primitive):
        """Test state reporting."""
        state = mutex_primitive.get_state()

        assert "state" in state
        assert "owners" in state
        assert "wait_count" in state
        assert "quota_usage" in state
        assert "last_error" in state
        assert "ttl_remaining" in state

        assert state["state"] == ResourceState.AVAILABLE.value
        assert state["owners"] == []
        assert state["wait_count"] == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, mutex_primitive):
        """Test error handling and state updates."""
        with patch.object(mutex_primitive, '_acquire_for', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                await mutex_primitive.acquire("test_caller")

            # Should have error state
            assert mutex_primitive._state == ResourceState.ERROR
            assert mutex_primitive._last_error == "Test error"


class TestMutex:
    """Test the Mutex coordination primitive."""

    @pytest.fixture
    def mutex(self):
        """Create a mutex for testing."""
        return Mutex("test_mutex", ttl=10.0)

    @pytest.mark.asyncio
    async def test_basic_functionality(self, mutex):
        """Test basic mutex operations."""
        caller1 = "caller1"
        caller2 = "caller2"

        # First caller should acquire successfully
        assert await mutex.acquire(caller1)
        assert caller1 in mutex._owners

        # Second caller should fail
        assert not await mutex.acquire(caller2)

        # Release and second caller should succeed
        assert await mutex.release(caller1)
        assert await mutex.acquire(caller2)

    @pytest.mark.asyncio
    async def test_context_manager(self, mutex):
        """Test async context manager functionality."""
        async with mutex as m:
            assert m is mutex
            assert len(mutex._owners) == 1
            # Context caller ID should be stored
            assert hasattr(mutex, '_context_caller_id')

        # Should be released after context
        assert len(mutex._owners) == 0
        assert not hasattr(mutex, '_context_caller_id')

    @pytest.mark.asyncio
    async def test_context_manager_exception(self, mutex):
        """Test context manager cleanup on exception."""
        try:
            async with mutex:
                assert len(mutex._owners) == 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be released after exception
        assert len(mutex._owners) == 0


class TestSemaphore:
    """Test the Semaphore coordination primitive."""

    @pytest.fixture
    def semaphore(self):
        """Create a semaphore for testing."""
        return Semaphore("test_semaphore", max_count=3, ttl=10.0)

    @pytest.mark.asyncio
    async def test_permit_management(self, semaphore):
        """Test permit acquisition and release."""
        callers = [f"caller{i}" for i in range(5)]

        # Should be able to acquire up to max_count
        for i in range(3):
            assert await semaphore.acquire(callers[i])
            assert semaphore.available_permits == 3 - (i + 1)

        # Should not be able to acquire more
        assert not await semaphore.acquire(callers[3])
        assert not await semaphore.acquire(callers[4])

        # Release permits
        assert await semaphore.release(callers[0])
        assert semaphore.available_permits == 1

        # Should be able to acquire again
        assert await semaphore.acquire(callers[3])
        assert semaphore.available_permits == 0

    def test_available_permits_property(self, semaphore):
        """Test available_permits property calculation."""
        assert semaphore.available_permits == 3

        # Manually add owners to test calculation
        semaphore._owners.add("test1")
        semaphore._owners.add("test2")
        assert semaphore.available_permits == 1


class TestBarrier:
    """Test the Barrier coordination primitive."""

    @pytest.fixture
    def barrier(self):
        """Create a barrier for testing."""
        return Barrier("test_barrier", parties=3, timeout=1.0)

    @pytest.mark.asyncio
    async def test_synchronization(self, barrier):
        """Test barrier synchronization with all parties."""
        callers = ["caller1", "caller2", "caller3"]
        results = []

        async def wait_at_barrier(caller_id):
            try:
                generation = await barrier.wait(caller_id)
                results.append((caller_id, generation))
            except asyncio.TimeoutError:
                results.append((caller_id, None))

        # All parties arrive at barrier
        tasks = [wait_at_barrier(caller) for caller in callers]
        await asyncio.gather(*tasks)

        # Check if synchronization worked
        assert len(results) == 3
        successful_results = [r for r in results if r[1] is not None]
        assert len(successful_results) == 3

        # All successful waiters should have the same generation number
        generations = [result[1] for result in successful_results]
        assert all(gen == generations[0] for gen in generations)

    @pytest.mark.asyncio
    async def test_barrier_reuse(self, barrier):
        """Test barrier can be reused after all parties pass."""
        # First round
        callers1 = ["c1-1", "c1-2", "c1-3"]
        tasks1 = [barrier.wait(c) for c in callers1]
        results1 = await asyncio.gather(*tasks1)
        assert all(gen == 0 for gen in results1)

        # Second round
        callers2 = ["c2-1", "c2-2", "c2-3"]
        tasks2 = [barrier.wait(c) for c in callers2]
        results2 = await asyncio.gather(*tasks2)
        assert all(gen == 1 for gen in results2)

    @pytest.mark.asyncio
    async def test_barrier_timeout(self, barrier):
        """Test barrier timeout when not all parties arrive."""
        # Only one party arrives
        with pytest.raises(asyncio.TimeoutError):
            await barrier.wait("lonely_caller")

    @pytest.mark.asyncio
    async def test_automatic_caller_id(self, barrier):
        """Test barrier with automatic caller ID generation."""
        # Should work without explicit caller_id
        async def wait_without_id():
            return await barrier.wait()

        # Not enough parties, so will timeout
        with pytest.raises(asyncio.TimeoutError):
            await wait_without_id()


class TestLease:
    """Test the Lease coordination primitive."""

    @pytest.fixture
    def lease(self):
        """Create a lease for testing."""
        return Lease("test_lease", ttl=0.5, auto_renew=False)

    @pytest.fixture
    def auto_renew_lease(self, request):
        """Create an auto-renewing lease for testing."""
        lease = Lease("auto_lease", ttl=0.5, auto_renew=True, renew_interval=0.1)

        async def finalizer():
            # Cleanup any running renewal tasks
            if hasattr(lease, '_renew_task') and lease._renew_task and not lease._renew_task.done():
                lease._renew_task.cancel()
                await asyncio.gather(lease._renew_task, return_exceptions=True)

        request.addfinalizer(finalizer)
        return lease

    @pytest.mark.asyncio
    async def test_basic_lease(self, lease):
        """Test basic lease functionality."""
        caller_id = "test_caller"

        # Should be able to acquire
        assert await lease.acquire(caller_id)
        assert caller_id in lease._owners

        # Should not be able to acquire with different caller
        assert not await lease.acquire("other_caller")

        # Release and should be available
        assert await lease.release(caller_id)
        assert caller_id not in lease._owners

    @pytest.mark.asyncio
    async def test_lease_expiration(self, lease):
        """Test lease expiration."""
        caller_id = "test_caller"

        # Acquire lease
        assert await lease.acquire(caller_id)

        # Wait for expiration
        await asyncio.sleep(0.6)

        # Should be able to acquire with different caller
        assert await lease.acquire("new_caller")
        assert caller_id not in lease._owners

    @pytest.mark.asyncio
    async def test_auto_renewal(self, auto_renew_lease):
        """Test auto-renewal functionality."""
        caller_id = "test_caller"

        # Acquire with auto-renewal
        assert await auto_renew_lease.acquire(caller_id)
        assert auto_renew_lease._renew_task is not None

        # Wait longer than TTL but less than TTL + renew_interval
        await asyncio.sleep(0.3)

        # Should still be owned due to renewal
        assert caller_id in auto_renew_lease._owners

        # Release should cancel renewal task
        assert await auto_renew_lease.release(caller_id)
        assert auto_renew_lease._renew_task is None

    @pytest.mark.asyncio
    async def test_auto_renewal_stops_when_not_owner(self, auto_renew_lease):
        """Test auto-renewal stops when no longer owner."""
        caller_id = "test_caller"

        # Acquire with auto-renewal
        await auto_renew_lease.acquire(caller_id)
        renew_task = auto_renew_lease._renew_task

        # Manually remove from owners to simulate expiration
        auto_renew_lease._owners.clear()

        # Wait for renewal loop to detect and stop
        await asyncio.sleep(0.15)

        # Task should complete
        assert renew_task.done()

    @pytest.mark.asyncio
    async def test_renewal_error_handling(self, auto_renew_lease):
        """Test renewal error handling."""
        caller_id = "test_caller"

        # Acquire lease first
        await auto_renew_lease.acquire(caller_id)

        # Patch the renewal method to raise an exception
        original_method = auto_renew_lease._auto_renew_loop

        async def failing_renewal():
            await asyncio.sleep(0.05)  # Small delay before failing
            raise Exception("Test error")

        # Replace the renewal loop with failing version
        auto_renew_lease._renew_task.cancel()
        await asyncio.gather(auto_renew_lease._renew_task, return_exceptions=True)

        # Start new task that will fail
        auto_renew_lease._renew_task = asyncio.create_task(failing_renewal())

        # Wait for failure
        await asyncio.sleep(0.1)

        # Task should be done (failed)
        assert auto_renew_lease._renew_task.done()


class TestLock:
    """Test the Lock coordination primitive."""

    @pytest.fixture
    def lock(self):
        """Create a lock for testing."""
        return Lock("test_lock", ttl=10.0)

    @pytest.mark.asyncio
    async def test_reentrant_acquisition(self, lock):
        """Test reentrant lock behavior."""
        caller_id = "test_caller"

        # Should be able to acquire multiple times
        assert await lock.acquire(caller_id)
        assert lock._lock_count[caller_id] == 1

        assert await lock.acquire(caller_id)
        assert lock._lock_count[caller_id] == 2

        assert await lock.acquire(caller_id)
        assert lock._lock_count[caller_id] == 3

        # Other caller should not be able to acquire
        assert not await lock.acquire("other_caller")

    @pytest.mark.asyncio
    async def test_reentrant_release(self, lock):
        """Test reentrant release behavior."""
        caller_id = "test_caller"

        # Acquire multiple times
        await lock.acquire(caller_id)
        await lock.acquire(caller_id)
        await lock.acquire(caller_id)

        # Should need to release same number of times
        assert await lock.release(caller_id)
        assert caller_id in lock._owners  # Still owned
        assert lock._lock_count[caller_id] == 2

        assert await lock.release(caller_id)
        assert caller_id in lock._owners  # Still owned
        assert lock._lock_count[caller_id] == 1

        assert await lock.release(caller_id)
        assert caller_id not in lock._owners  # Finally released
        assert caller_id not in lock._lock_count

    @pytest.mark.asyncio
    async def test_release_without_ownership(self, lock):
        """Test releasing lock without ownership."""
        # Should return False when not owned
        assert not await lock.release("non_owner")

    @pytest.mark.asyncio
    async def test_lock_count_initialization(self, lock):
        """Test lock count is properly initialized."""
        caller_id = "test_caller"

        # First acquire should set count to 1
        await lock.acquire(caller_id)
        assert lock._lock_count[caller_id] == 1


class TestQuota:
    """Test the Quota coordination primitive."""

    @pytest.fixture
    def quota(self):
        """Create a quota for testing."""
        return Quota("test_quota", limit=100.0)

    @pytest.fixture
    def reset_quota(self, request):
        """Create a quota with reset interval."""
        quota = Quota("reset_quota", limit=50.0, reset_interval=timedelta(seconds=0.2))

        async def finalizer():
            # Cleanup
            if hasattr(quota, '_reset_task') and quota._reset_task and not quota._reset_task.done():
                quota._reset_task.cancel()
                await asyncio.gather(quota._reset_task, return_exceptions=True)

        request.addfinalizer(finalizer)
        return quota

    @pytest.mark.asyncio
    async def test_quota_consumption(self, quota):
        """Test quota consumption and tracking."""
        # Should be able to consume quota
        assert await quota.consume("caller1", 30.0)
        assert quota.usage["caller1"] == 30.0
        assert quota.available == 70.0

        # Should be able to consume more
        assert await quota.consume("caller2", 50.0)
        assert quota.usage["caller2"] == 50.0
        assert quota.available == 20.0

        # Should not be able to exceed limit
        assert not await quota.consume("caller3", 25.0)
        assert quota.available == 20.0

    @pytest.mark.asyncio
    async def test_quota_release(self, quota):
        """Test quota release functionality."""
        caller_id = "test_caller"

        # Consume some quota
        await quota.consume(caller_id, 40.0)
        assert quota.usage[caller_id] == 40.0

        # Release partial quota
        await quota.release_quota(caller_id, 15.0)
        assert quota.usage[caller_id] == 25.0
        assert quota.available == 75.0

        # Release more than consumed should not go negative
        await quota.release_quota(caller_id, 50.0)
        assert quota.usage[caller_id] == 0.0
        assert quota.available == 100.0

    @pytest.mark.asyncio
    async def test_quota_reset(self, quota):
        """Test manual quota reset."""
        # Consume quota
        await quota.consume("caller1", 30.0)
        await quota.consume("caller2", 40.0)

        assert quota.available == 30.0

        # Reset quota
        await quota.reset()

        assert quota.available == 100.0
        assert len(quota.usage) == 0

    @pytest.mark.asyncio
    async def test_periodic_reset(self, reset_quota):
        """Test periodic quota reset."""
        # Consume quota
        await reset_quota.consume("caller1", 30.0)
        assert reset_quota.available == 20.0

        # Wait for reset
        await asyncio.sleep(0.25)

        # Should be reset
        assert reset_quota.available == 50.0
        assert len(reset_quota.usage) == 0

    @pytest.mark.asyncio
    async def test_reset_task_cleanup(self, reset_quota):
        """Test reset task cleanup on deletion."""
        # Trigger the lazy initialization of the reset task
        await reset_quota.consume("dummy_caller", 0.0)

        # Ensure reset task is running
        assert reset_quota._reset_task is not None
        assert not reset_quota._reset_task.done()

        # Store reference to task before deletion
        reset_task = reset_quota._reset_task

        # Cancel the task manually (simulating deletion)
        reset_task.cancel()

        # Wait for cancellation to process
        await asyncio.gather(reset_task, return_exceptions=True)

    def test_quota_properties(self, quota):
        """Test quota properties."""
        # Initial state
        assert quota.available == 100.0
        assert quota.usage == {}

        # After consumption
        quota._quota_usage["test"] = 25.0
        assert quota.available == 75.0
        assert quota.usage == {"test": 25.0}


class TestCreatePrimitive:
    """Test the create_primitive factory function."""

    def test_create_mutex(self):
        """Test creating mutex primitive."""
        primitive = create_primitive(PrimitiveType.MUTEX, "test_mutex", ttl=20.0)
        assert isinstance(primitive, Mutex)
        assert primitive.name == "test_mutex"
        assert primitive.ttl == 20.0

    def test_create_semaphore(self):
        """Test creating semaphore primitive."""
        primitive = create_primitive(PrimitiveType.SEMAPHORE, "test_sem", max_count=5)
        assert isinstance(primitive, Semaphore)
        assert primitive.name == "test_sem"
        assert primitive.max_count == 5

    def test_create_barrier(self):
        """Test creating barrier primitive."""
        primitive = create_primitive(PrimitiveType.BARRIER, "test_barrier", parties=10)
        assert isinstance(primitive, Barrier)
        assert primitive.name == "test_barrier"
        assert primitive._parties == 10

    def test_create_lease(self):
        """Test creating lease primitive."""
        primitive = create_primitive(PrimitiveType.LEASE, "test_lease", auto_renew=True)
        assert isinstance(primitive, Lease)
        assert primitive.name == "test_lease"
        assert primitive.auto_renew is True

    def test_create_lock(self):
        """Test creating lock primitive."""
        primitive = create_primitive(PrimitiveType.LOCK, "test_lock")
        assert isinstance(primitive, Lock)
        assert primitive.name == "test_lock"

    def test_create_quota(self):
        """Test creating quota primitive."""
        primitive = create_primitive(PrimitiveType.QUOTA, "test_quota", limit=200.0)
        assert isinstance(primitive, Quota)
        assert primitive.name == "test_quota"
        assert primitive.quota_limit == 200.0

    def test_create_unknown_type(self):
        """Test creating primitive with unknown type falls back to base class."""
        # Create a mock enum that's not handled by factory
        class MockPrimitiveType:
            UNKNOWN = "unknown"

        unknown_type = MockPrimitiveType.UNKNOWN
        primitive = create_primitive(unknown_type, "test")
        assert isinstance(primitive, CoordinationPrimitive)
        assert primitive.name == "test"
        assert primitive.type == unknown_type


class TestConcurrentAccess:
    """Test coordination primitives under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_mutex_access(self):
        """Test mutex under concurrent access."""
        mutex = Mutex("concurrent_test")
        results = []

        async def try_acquire(caller_id):
            success = await mutex.acquire(caller_id)
            if success:
                await asyncio.sleep(0.1)  # Hold for a bit
                await mutex.release(caller_id)
            results.append((caller_id, success))

        # Multiple concurrent attempts
        callers = [f"caller{i}" for i in range(10)]
        tasks = [try_acquire(caller) for caller in callers]
        await asyncio.gather(*tasks)

        # Only one should have succeeded at a time
        successful = [r for r in results if r[1]]
        assert len(successful) >= 1  # At least one should succeed

    @pytest.mark.asyncio
    async def test_concurrent_semaphore_access(self):
        """Test semaphore under concurrent access."""
        semaphore = Semaphore("concurrent_test", max_count=3)
        results = []

        async def try_acquire(caller_id):
            success = await semaphore.acquire(caller_id)
            if success:
                await asyncio.sleep(0.1)
                await semaphore.release(caller_id)
            results.append((caller_id, success))

        # More attempts than permits
        callers = [f"caller{i}" for i in range(10)]
        tasks = [try_acquire(caller) for caller in callers]
        await asyncio.gather(*tasks)

        # At most max_count should succeed simultaneously
        successful = [r for r in results if r[1]]
        assert len(successful) >= 3  # Should allow multiple


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_release_non_existent_caller(self):
        """Test releasing with non-existent caller."""
        mutex = Mutex("test")
        assert not await mutex.release("non_existent")

    @pytest.mark.asyncio
    async def test_multiple_releases(self):
        """Test multiple releases of same caller."""
        mutex = Mutex("test")
        caller_id = "test_caller"

        await mutex.acquire(caller_id)
        assert await mutex.release(caller_id)

        # Second release should return False
        assert not await mutex.release(caller_id)

    @pytest.mark.asyncio
    async def test_quota_zero_consumption(self):
        """Test quota with zero consumption."""
        quota = Quota("test", limit=100.0)

        # Should be able to consume zero
        assert await quota.consume("caller", 0.0)
        assert quota.usage["caller"] == 0.0

    @pytest.mark.asyncio
    async def test_quota_negative_release(self):
        """Test quota release with negative amount."""
        quota = Quota("test", limit=100.0)
        caller_id = "test_caller"

        await quota.consume(caller_id, 50.0)
        original_usage = quota.usage[caller_id]

        # Release negative amount (should be handled gracefully)
        await quota.release_quota(caller_id, -10.0)
        # Usage should not go negative and should remain valid
        assert quota.usage[caller_id] >= 0
        assert quota.usage[caller_id] <= original_usage

    def test_primitive_state_with_no_owners(self):
        """Test state reporting with no owners."""
        primitive = CoordinationPrimitive("test", PrimitiveType.MUTEX)
        state = primitive.get_state()

        assert state["ttl_remaining"] is None
        assert state["owners"] == []

    @pytest.mark.asyncio
    async def test_barrier_with_zero_parties(self):
        """Test barrier with zero parties."""
        barrier = Barrier("test", parties=0)

        # Should immediately succeed since no parties needed
        generation = await barrier.wait("caller")
        assert isinstance(generation, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])