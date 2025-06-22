"""
Comprehensive test coverage for core.agent.context module.

Tests cover:
- Context initialization and configuration
- Per-state scratch data (typed and untyped)
- Shared state variables (free, typed, validated)
- Constants and secrets management
- TTL cache functionality
- Output helpers
- Metadata persistence and restoration
- Pydantic v1/v2 compatibility
- Human-in-the-loop functionality
- Error handling and edge cases
"""

import pytest
import asyncio
import time
import importlib
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import the module to test
from core.agent.context import Context, StateType, TypedContextData

# Test Pydantic models for validation testing
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
    
    class TestUser(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
    
    class TestProduct(BaseModel):
        id: int
        name: str
        price: float
        
except ImportError:
    PYDANTIC_AVAILABLE = False
    TestUser = None
    TestProduct = None


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def shared_state():
    """Basic shared state dictionary for testing."""
    return {
        "existing_key": "existing_value",
        "const_test_constant": "constant_value",
        "secret_test_secret": "secret_value"
    }


@pytest.fixture
def context(shared_state):
    """Basic context instance for testing."""
    return Context(shared_state=shared_state, cache_ttl=300)


@pytest.fixture
def context_with_metadata(shared_state):
    """Context with pre-existing metadata for testing restoration."""
    shared_state.update({
        "_meta_typed_test_var": "builtins.str",
        "test_var": "test_value",
        "_meta_validated_user_data": "tests.test_context.TestUser" if PYDANTIC_AVAILABLE else "dict",
    })
    
    if PYDANTIC_AVAILABLE:
        shared_state["user_data"] = TestUser(name="John", age=30)
    
    return Context(shared_state=shared_state)


@pytest.fixture
def mock_pydantic_unavailable():
    """Mock Pydantic being unavailable for testing error handling."""
    with patch.object(sys.modules['core.agent.context'], '_PYD_VER', 0):
        with patch.object(sys.modules['core.agent.context'], '_PBM', None):
            with patch.object(sys.modules['core.agent.context'], '_PYD_ERR', ImportError("No module named 'pydantic'"), create=True):
                yield


# ============================================================================
# CONTEXT INITIALIZATION TESTS
# ============================================================================

class TestContextInitialization:
    """Test cases for Context initialization."""

    def test_context_basic_initialization(self):
        """Test basic Context initialization."""
        shared_state = {"key": "value"}
        context = Context(shared_state=shared_state)

        assert context.shared_state is shared_state
        assert context.cache_ttl == 300  # Default value
        assert isinstance(context._state_data, dict)
        assert isinstance(context._typed_data, dict)
        assert isinstance(context._protected_keys, set)
        assert isinstance(context._typed_var_types, dict)
        assert isinstance(context._validated_types, dict)
        assert isinstance(context._cache, dict)

    def test_context_custom_cache_ttl(self):
        """Test Context initialization with custom cache TTL."""
        context = Context(shared_state={}, cache_ttl=600)
        assert context.cache_ttl == 600

    def test_context_restore_metadata_empty(self):
        """Test metadata restoration with empty shared state."""
        context = Context(shared_state={})
        assert len(context._typed_var_types) == 0
        assert len(context._validated_types) == 0

    def test_context_restore_metadata_with_data(self, context_with_metadata):
        """Test metadata restoration with existing metadata."""
        context = context_with_metadata

        assert "test_var" in context._typed_var_types
        assert context._typed_var_types["test_var"] == str

        if PYDANTIC_AVAILABLE:
            assert "user_data" in context._validated_types

    def test_context_restore_metadata_orphaned_keys(self):
        """Test metadata restoration with orphaned metadata keys."""
        shared_state = {
            "_meta_typed_missing_var": "builtins.str",  # No corresponding var
            "_meta_validated_missing_data": "dict"      # No corresponding data
        }
        context = Context(shared_state=shared_state)

        # Should not restore metadata for missing variables
        assert len(context._typed_var_types) == 0
        assert len(context._validated_types) == 0


# ============================================================================
# PER-STATE SCRATCH DATA TESTS
# ============================================================================

class TestPerStateScratchData:
    """Test cases for per-state scratch data management."""

    def test_set_get_state_basic(self, context):
        """Test basic set/get state operations."""
        context.set_state("test_key", "test_value")
        assert context.get_state("test_key") == "test_value"

    def test_get_state_with_default(self, context):
        """Test getting state with default value."""
        assert context.get_state("nonexistent", "default") == "default"
        assert context.get_state("nonexistent") is None

    def test_set_state_overwrite_protected_key_error(self, context):
        """Test that setting state on protected keys raises error."""
        context._protected_keys.add("protected_key")

        with pytest.raises(KeyError, match="Cannot overwrite typed slot 'protected_key'"):
            context.set_state("protected_key", "value")

    def test_get_state_protected_key_error(self, context):
        """Test that getting state from protected keys raises error."""
        context._protected_keys.add("protected_key")

        with pytest.raises(KeyError, match="Cannot access typed slot 'protected_key'"):
            context.get_state("protected_key")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_set_typed_valid_model(self, context):
        """Test setting typed data with valid Pydantic model."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)

        assert "user" in context._protected_keys
        assert "user" in context._typed_data
        assert context._typed_data["user"] is user
        assert "user" not in context._state_data

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_set_typed_invalid_type(self, context):
        """Test setting typed data with non-Pydantic object."""
        with pytest.raises(TypeError, match="set_typed expects a Pydantic BaseModel"):
            context.set_typed("invalid", "not a model")

    def test_set_typed_pydantic_unavailable(self, context, mock_pydantic_unavailable):
        """Test set_typed when Pydantic is unavailable."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            context.set_typed("test", "value")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_get_typed_valid(self, context):
        """Test getting typed data with correct type."""
        user = TestUser(name="Bob", age=30)
        context.set_typed("user", user)

        retrieved = context.get_typed("user", TestUser)
        assert retrieved is user
        assert isinstance(retrieved, TestUser)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_get_typed_wrong_type(self, context):
        """Test getting typed data with wrong expected type."""
        user = TestUser(name="Charlie", age=35)
        context.set_typed("user", user)

        # Try to get as wrong type
        result = context.get_typed("user", TestProduct)
        assert result is None

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_get_typed_nonexistent(self, context):
        """Test getting non-existent typed data."""
        result = context.get_typed("nonexistent", TestUser)
        assert result is None

    def test_get_typed_pydantic_unavailable(self, context, mock_pydantic_unavailable):
        """Test get_typed when Pydantic is unavailable."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            context.get_typed("test", str)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_update_typed_valid(self, context):
        """Test updating typed data."""
        user = TestUser(name="David", age=40)
        context.set_typed("user", user)

        context.update_typed("user", name="David Updated", age=41)

        updated_user = context.get_typed("user", TestUser)
        assert updated_user.name == "David Updated"
        assert updated_user.age == 41

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_update_typed_nonexistent(self, context):
        """Test updating non-existent typed data."""
        with pytest.raises(KeyError, match="No typed data under 'nonexistent'"):
            context.update_typed("nonexistent", name="test")

    def test_update_typed_pydantic_unavailable(self, context, mock_pydantic_unavailable):
        """Test update_typed when Pydantic is unavailable."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            context.update_typed("test", field="value")


# ============================================================================
# FREE VARIABLES TESTS
# ============================================================================

class TestFreeVariables:
    """Test cases for free variable management."""

    def test_set_get_variable_basic(self, context):
        """Test basic set/get variable operations."""
        context.set_variable("test_var", "test_value")
        assert context.get_variable("test_var") == "test_value"

    def test_get_variable_with_default(self, context):
        """Test getting variable with default value."""
        assert context.get_variable("nonexistent", "default") == "default"
        assert context.get_variable("nonexistent") is None

    def test_set_variable_overwrite_existing(self, context):
        """Test overwriting existing variable."""
        context.set_variable("var", "value1")
        context.set_variable("var", "value2")
        assert context.get_variable("var") == "value2"

    def test_set_variable_reserved_prefix_const(self, context):
        """Test setting variable with reserved const_ prefix."""
        with pytest.raises(KeyError, match="Keys beginning with 'const_' or 'secret_' are reserved"):
            context.set_variable("const_test", "value")

    def test_set_variable_reserved_prefix_secret(self, context):
        """Test setting variable with reserved secret_ prefix."""
        with pytest.raises(KeyError, match="Keys beginning with 'const_' or 'secret_' are reserved"):
            context.set_variable("secret_test", "value")

    def test_get_variable_keys_filters_reserved(self, context):
        """Test that get_variable_keys filters out reserved prefixes."""
        context.set_variable("normal_var", "value")
        # Reserved keys already exist in shared_state fixture

        keys = context.get_variable_keys()

        assert "normal_var" in keys
        assert "existing_key" in keys
        assert "const_test_constant" not in keys
        assert "secret_test_secret" not in keys
        # Should also filter metadata keys
        for key in keys:
            assert not key.startswith("_meta_")


# ============================================================================
# TYPED VARIABLES TESTS
# ============================================================================

class TestTypedVariables:
    """Test cases for typed variable management."""

    def test_set_typed_variable_first_time(self, context):
        """Test setting typed variable for the first time."""
        context.set_typed_variable("test_var", "string_value")

        assert context.get_variable("test_var") == "string_value"
        assert context._typed_var_types["test_var"] == str
        assert "_meta_typed_test_var" in context.shared_state

    def test_set_typed_variable_same_type(self, context):
        """Test setting typed variable with same type."""
        context.set_typed_variable("test_var", "value1")
        context.set_typed_variable("test_var", "value2")  # Same type (str)

        assert context.get_variable("test_var") == "value2"

    def test_set_typed_variable_different_type_error(self, context):
        """Test setting typed variable with different type raises error."""
        context.set_typed_variable("test_var", "string_value")

        with pytest.raises(TypeError, match="Typed variable 'test_var' already holds str"):
            context.set_typed_variable("test_var", 123)

    def test_set_typed_variable_reserved_prefix(self, context):
        """Test setting typed variable with reserved prefix."""
        with pytest.raises(KeyError, match="Keys beginning with 'const_' or 'secret_' are reserved"):
            context.set_typed_variable("const_test", "value")

    def test_get_typed_variable_correct_type(self, context):
        """Test getting typed variable with correct expected type."""
        context.set_typed_variable("test_var", 42)
        result = context.get_typed_variable("test_var", int)
        assert result == 42

    def test_get_typed_variable_wrong_type(self, context):
        """Test getting typed variable with wrong expected type."""
        context.set_typed_variable("test_var", "string")
        result = context.get_typed_variable("test_var", int)
        assert result is None

    def test_get_typed_variable_nonexistent(self, context):
        """Test getting non-existent typed variable."""
        result = context.get_typed_variable("nonexistent", str)
        assert result is None

    def test_typed_variable_metadata_persistence(self, context_with_metadata):
        """Test that typed variable metadata persists correctly."""
        context = context_with_metadata

        # Should be restored from metadata
        assert "test_var" in context._typed_var_types
        assert context._typed_var_types["test_var"] == str


# ============================================================================
# VALIDATED DATA TESTS
# ============================================================================

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestValidatedData:
    """Test cases for validated data management."""

    def test_set_validated_data_first_time(self, context):
        """Test setting validated data for the first time."""
        user = TestUser(name="Alice", age=25)
        context.set_validated_data("user", user)

        assert context.shared_state["user"] is user
        assert context._validated_types["user"] == TestUser
        assert "_meta_validated_user" in context.shared_state

    def test_set_validated_data_same_type(self, context):
        """Test setting validated data with same type."""
        user1 = TestUser(name="Alice", age=25)
        user2 = TestUser(name="Bob", age=30)

        context.set_validated_data("user", user1)
        context.set_validated_data("user", user2)  # Same type

        assert context.shared_state["user"] is user2

    def test_set_validated_data_different_type_error(self, context):
        """Test setting validated data with different type raises error."""
        user = TestUser(name="Alice", age=25)
        product = TestProduct(id=1, name="Widget", price=10.0)

        context.set_validated_data("item", user)

        with pytest.raises(TypeError, match="Validated key 'item' already stores TestUser"):
            context.set_validated_data("item", product)

    def test_set_validated_data_non_pydantic_error(self, context):
        """Test setting validated data with non-Pydantic object."""
        with pytest.raises(TypeError, match="set_validated_data expects a Pydantic BaseModel"):
            context.set_validated_data("invalid", "not a model")

    def test_set_validated_data_reserved_prefix(self, context):
        """Test setting validated data with reserved prefix."""
        user = TestUser(name="Alice", age=25)

        with pytest.raises(KeyError, match="Keys beginning with 'const_' or 'secret_' are reserved"):
            context.set_validated_data("const_user", user)

    def test_get_validated_data_correct_type(self, context):
        """Test getting validated data with correct expected type."""
        user = TestUser(name="Alice", age=25)
        context.set_validated_data("user", user)

        result = context.get_validated_data("user", TestUser)
        assert result is user

    def test_get_validated_data_wrong_type(self, context):
        """Test getting validated data with wrong expected type."""
        user = TestUser(name="Alice", age=25)
        context.set_validated_data("user", user)

        result = context.get_validated_data("user", TestProduct)
        assert result is None

    def test_get_validated_data_nonexistent(self, context):
        """Test getting non-existent validated data."""
        result = context.get_validated_data("nonexistent", TestUser)
        assert result is None

    def test_validated_data_metadata_persistence(self, context_with_metadata):
        """Test that validated data metadata persists correctly."""
        context = context_with_metadata

        # Should be restored from metadata
        if PYDANTIC_AVAILABLE:
            assert "user_data" in context._validated_types


class TestValidatedDataWithoutPydantic:
    """Test validated data methods when Pydantic is unavailable."""

    def test_set_validated_data_pydantic_unavailable(self, context, mock_pydantic_unavailable):
        """Test set_validated_data when Pydantic is unavailable."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            context.set_validated_data("test", "value")

    def test_get_validated_data_pydantic_unavailable(self, context, mock_pydantic_unavailable):
        """Test get_validated_data when Pydantic is unavailable."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            context.get_validated_data("test", str)


# ============================================================================
# CONSTANTS AND SECRETS TESTS
# ============================================================================

class TestConstantsAndSecrets:
    """Test cases for constants and secrets management."""

    def test_set_constant_new(self, context):
        """Test setting a new constant."""
        context.set_constant("api_version", "v1.0")
        assert context.get_constant("api_version") == "v1.0"
        assert context.shared_state["const_api_version"] == "v1.0"

    def test_set_constant_already_exists_error(self, context):
        """Test setting constant that already exists raises error."""
        context.set_constant("api_version", "v1.0")

        with pytest.raises(ValueError, match="Immutable key 'api_version' already set"):
            context.set_constant("api_version", "v2.0")

    def test_get_constant_with_default(self, context):
        """Test getting constant with default value."""
        assert context.get_constant("nonexistent", "default") == "default"
        assert context.get_constant("nonexistent") is None

    def test_get_constant_existing(self, context):
        """Test getting existing constant from fixture."""
        assert context.get_constant("test_constant") == "constant_value"

    def test_set_secret_new(self, context):
        """Test setting a new secret."""
        context.set_secret("api_key", "secret123")
        assert context.get_secret("api_key") == "secret123"
        assert context.shared_state["secret_api_key"] == "secret123"

    def test_set_secret_already_exists_error(self, context):
        """Test setting secret that already exists raises error."""
        context.set_secret("api_key", "secret123")

        with pytest.raises(ValueError, match="Immutable key 'api_key' already set"):
            context.set_secret("api_key", "secret456")

    def test_get_secret_nonexistent(self, context):
        """Test getting non-existent secret returns None."""
        assert context.get_secret("nonexistent") is None

    def test_get_secret_existing(self, context):
        """Test getting existing secret from fixture."""
        assert context.get_secret("test_secret") == "secret_value"


# ============================================================================
# OUTPUT HELPERS TESTS
# ============================================================================

class TestOutputHelpers:
    """Test cases for output helper methods."""

    def test_set_get_output(self, context):
        """Test setting and getting output values."""
        context.set_output("result", "success")
        assert context.get_output("result") == "success"

    def test_get_output_with_default(self, context):
        """Test getting output with default value."""
        assert context.get_output("nonexistent", "default") == "default"
        assert context.get_output("nonexistent") is None

    def test_get_output_keys(self, context):
        """Test getting all output keys."""
        context.set_output("result1", "value1")
        context.set_output("result2", "value2")
        context.set_state("non_output", "value")  # Should not appear

        output_keys = context.get_output_keys()
        assert output_keys == {"result1", "result2"}

    def test_output_isolation_from_regular_state(self, context):
        """Test that output state is isolated from regular state."""
        context.set_output("key", "output_value")
        context.set_state("key", "state_value")

        assert context.get_output("key") == "output_value"
        assert context.get_state("key") == "state_value"


# ============================================================================
# TTL CACHE TESTS
# ============================================================================

class TestTTLCache:
    """Test cases for TTL cache functionality."""

    def test_set_get_cached_basic(self, context):
        """Test basic cache set/get operations."""
        context.set_cached("key", "value")
        assert context.get_cached("key") == "value"

    def test_set_cached_custom_ttl(self, context):
        """Test setting cached value with custom TTL."""
        context.set_cached("key", "value", ttl=100)
        assert context.get_cached("key") == "value"

    def test_get_cached_with_default(self, context):
        """Test getting cached value with default."""
        assert context.get_cached("nonexistent", "default") == "default"
        assert context.get_cached("nonexistent") is None

    def test_cache_expiration(self, context):
        """Test that cached values expire after TTL."""
        # Set with very short TTL
        context.set_cached("key", "value", ttl=0.05)
        assert context.get_cached("key") == "value"

        # Wait for expiration
        time.sleep(0.1)

        # Should return default after expiration
        assert context.get_cached("key", "default") == "default"
        assert "key" not in context._cache  # Should be cleaned up

    def test_cache_not_expired(self, context):
        """Test that cached values don't expire before TTL."""
        context.set_cached("key", "value", ttl=1.0)

        # Should still be valid immediately
        assert context.get_cached("key") == "value"

        # Should still be valid after short delay
        time.sleep(0.1)
        assert context.get_cached("key") == "value"

    def test_cache_default_ttl(self, context):
        """Test that default TTL from context is used."""
        context.cache_ttl = 0.05
        context.set_cached("key", "value")  # No explicit TTL

        assert context.get_cached("key") == "value"

        time.sleep(0.1)
        assert context.get_cached("key", "default") == "default"

    def test_cache_overwrite(self, context):
        """Test overwriting cached values."""
        context.set_cached("key", "value1")
        context.set_cached("key", "value2")
        assert context.get_cached("key") == "value2"


# ============================================================================
# HOUSEKEEPING TESTS
# ============================================================================

class TestHousekeeping:
    """Test cases for state management and housekeeping."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_remove_state_typed_only(self, context):
        """Test removing only typed state."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)
        context.set_state("regular", "value")

        removed = context.remove_state("user", StateType.TYPED)
        assert removed is True
        assert "user" not in context._typed_data
        assert "user" not in context._protected_keys
        assert "regular" in context._state_data

    def test_remove_state_untyped_only(self, context):
        """Test removing only untyped state."""
        context.set_state("regular", "value")
        context.set_state("another", "value2")

        removed = context.remove_state("regular", StateType.UNTYPED)
        assert removed is True
        assert "regular" not in context._state_data
        assert "another" in context._state_data

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_remove_state_any_type(self, context):
        """Test removing state of any type."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)
        context.set_state("regular", "value")

        # Remove typed data
        removed = context.remove_state("user", StateType.ANY)
        assert removed is True
        assert "user" not in context._typed_data

        # Remove untyped data
        removed = context.remove_state("regular", StateType.ANY)
        assert removed is True
        assert "regular" not in context._state_data

    def test_remove_state_nonexistent(self, context):
        """Test removing non-existent state."""
        removed = context.remove_state("nonexistent", StateType.ANY)
        assert removed is False

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_clear_state_typed_only(self, context):
        """Test clearing only typed state."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)
        context.set_state("regular", "value")

        context.clear_state(StateType.TYPED)

        assert len(context._typed_data) == 0
        assert len(context._protected_keys) == 0
        assert "regular" in context._state_data

    def test_clear_state_untyped_only(self, context):
        """Test clearing only untyped state."""
        context.set_state("regular1", "value1")
        context.set_state("regular2", "value2")

        context.clear_state(StateType.UNTYPED)

        assert len(context._state_data) == 0

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_clear_state_any_type(self, context):
        """Test clearing state of any type."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)
        context.set_state("regular", "value")

        context.clear_state(StateType.ANY)

        assert len(context._typed_data) == 0
        assert len(context._protected_keys) == 0
        assert len(context._state_data) == 0

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_get_keys_all_types(self, context):
        """Test getting keys of all types."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)
        context.set_state("regular", "value")

        keys = context.get_keys(StateType.ANY)
        assert keys == {"user", "regular"}

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_get_keys_typed_only(self, context):
        """Test getting only typed keys."""
        user = TestUser(name="Alice", age=25)
        context.set_typed("user", user)
        context.set_state("regular", "value")

        keys = context.get_keys(StateType.TYPED)
        assert keys == {"user"}

    def test_get_keys_untyped_only(self, context):
        """Test getting only untyped keys."""
        context.set_state("regular1", "value1")
        context.set_state("regular2", "value2")
        context._protected_keys.add("protected")  # Should be excluded

        keys = context.get_keys(StateType.UNTYPED)
        assert keys == {"regular1", "regular2"}


# ============================================================================
# HUMAN-IN-THE-LOOP TESTS
# ============================================================================

class TestHumanInTheLoop:
    """Test cases for human-in-the-loop functionality."""

    @pytest.mark.asyncio
    async def test_human_in_the_loop_basic(self, context):
        """Test basic human-in-the-loop functionality."""
        mock_input = Mock(return_value="user_response")

        with patch('builtins.input', mock_input):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value="user_response")

                result = await context.human_in_the_loop("Enter value: ")
                assert result == "user_response"

    @pytest.mark.asyncio
    async def test_human_in_the_loop_with_timeout_success(self, context):
        """Test human-in-the-loop with timeout that succeeds."""
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="quick_response")

            result = await context.human_in_the_loop("Enter value: ", timeout=1.0)
            assert result == "quick_response"

    @pytest.mark.asyncio
    async def test_human_in_the_loop_with_timeout_expires(self, context):
        """Test human-in-the-loop with timeout that expires."""
        async def slow_response(*args):
            await asyncio.sleep(2.0)
            return "slow_response"

        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
            result = await context.human_in_the_loop(
                "Enter value: ",
                timeout=0.1,
                default="timeout_default"
            )
            assert result == "timeout_default"

    @pytest.mark.asyncio
    async def test_human_in_the_loop_timeout_no_default(self, context):
        """Test human-in-the-loop timeout with no default returns empty string."""
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
            result = await context.human_in_the_loop("Enter value: ", timeout=0.1)
            assert result == ""

    @pytest.mark.asyncio
    async def test_human_in_the_loop_with_validator_valid(self, context):
        """Test human-in-the-loop with validator that passes."""
        def is_number(value):
            try:
                int(value)
                return True
            except ValueError:
                return False

        mock_input_responses = ["123"]  # Valid number

        with patch('builtins.input', side_effect=mock_input_responses):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=mock_input_responses)

                result = await context.human_in_the_loop("Enter number: ", validator=is_number)
                assert result == "123"

    @pytest.mark.asyncio
    async def test_human_in_the_loop_with_validator_retry(self, context):
        """Test human-in-the-loop with validator that requires retry."""
        def is_number(value):
            try:
                int(value)
                return True
            except ValueError:
                return False

        mock_input_responses = ["abc", "def", "123"]  # Two invalid, then valid

        with patch('builtins.input', side_effect=mock_input_responses):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=mock_input_responses)

                result = await context.human_in_the_loop("Enter number: ", validator=is_number)
                assert result == "123"

    @pytest.mark.asyncio
    async def test_human_in_the_loop_no_timeout_no_validator(self, context):
        """Test human-in-the-loop without timeout or validator."""
        with patch('builtins.input', return_value="simple_response"):
            result = await context.human_in_the_loop("Enter value: ")
            assert result == "simple_response"


# ============================================================================
# UTILITY AND HELPER TESTS
# ============================================================================

class TestUtilityMethods:
    """Test cases for utility and helper methods."""

    def test_now_method(self, context):
        """Test the _now method returns monotonic time."""
        time1 = context._now()
        time.sleep(0.01)
        time2 = context._now()

        assert isinstance(time1, float)
        assert isinstance(time2, float)
        assert time2 > time1

    def test_ensure_pydantic_available(self, context):
        """Test _ensure_pydantic with Pydantic available."""
        if PYDANTIC_AVAILABLE:
            # Should not raise
            context._ensure_pydantic()
        else:
            with pytest.raises(ImportError):
                context._ensure_pydantic()

    def test_ensure_pydantic_unavailable(self, context, mock_pydantic_unavailable):
        """Test _ensure_pydantic when Pydantic is unavailable."""
        with pytest.raises(ImportError, match="Pydantic is required"):
            context._ensure_pydantic()

    def test_guard_reserved_const(self, context):
        """Test _guard_reserved with const_ prefix."""
        with pytest.raises(KeyError, match="Keys beginning with 'const_' or 'secret_' are reserved"):
            context._guard_reserved("const_test")

    def test_guard_reserved_secret(self, context):
        """Test _guard_reserved with secret_ prefix."""
        with pytest.raises(KeyError, match="Keys beginning with 'const_' or 'secret_' are reserved"):
            context._guard_reserved("secret_test")

    def test_guard_reserved_normal_key(self, context):
        """Test _guard_reserved with normal key (should not raise)."""
        context._guard_reserved("normal_key")  # Should not raise

    def test_persist_meta(self, context):
        """Test _persist_meta method."""
        context._persist_meta("_test_prefix_", "test_key", str)

        expected_key = "_test_prefix_test_key"
        assert expected_key in context.shared_state
        assert context.shared_state[expected_key] == "builtins.str"


# ============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# ============================================================================

class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_context_with_none_shared_state(self):
        """Test context behavior with None shared_state."""
        # This should raise AttributeError since dict methods will be called on None
        with pytest.raises(AttributeError):
            Context(shared_state=None)

    def test_context_with_empty_shared_state(self):
        """Test context with empty shared state."""
        context = Context(shared_state={})

        # Should work normally
        context.set_variable("test", "value")
        assert context.get_variable("test") == "value"

    def test_cache_with_zero_ttl(self, context):
        """Test cache behavior with zero TTL."""
        # Note: ttl=0 currently uses default cache_ttl due to "ttl or self.cache_ttl" logic
        # This test verifies current behavior - in practice you'd use a very small positive value
        context.set_cached("key", "value", ttl=0)

        # Currently behaves like default TTL due to implementation
        assert context.get_cached("key") == "value"

    def test_cache_with_negative_ttl(self, context):
        """Test cache behavior with negative TTL."""
        context.set_cached("key", "value", ttl=-1)
        
        # Should expire immediately
        assert context.get_cached("key", "default") == "default"

    def test_large_cache_ttl(self, context):
        """Test cache with very large TTL."""
        context.set_cached("key", "value", ttl=86400)  # 24 hours
        assert context.get_cached("key") == "value"

    def test_metadata_with_invalid_class_path(self):
        """Test metadata restoration with invalid class path."""
        shared_state = {
            "_meta_validated_invalid": "nonexistent.module.Class",
            "invalid": "some_value"
        }
        
        # Should not crash, just not restore the metadata
        context = Context(shared_state=shared_state)
        assert len(context._validated_types) == 0

    def test_complex_nested_data_structures(self, context):
        """Test context with complex nested data structures."""
        complex_data = {
            "nested": {
                "deeply": {
                    "nested": ["list", "with", {"mixed": "types"}]
                }
            },
            "numbers": [1, 2, 3, 4, 5],
            "mixed_list": [1, "two", 3.0, True, None]
        }
        
        context.set_variable("complex", complex_data)
        retrieved = context.get_variable("complex")
        
        assert retrieved == complex_data
        assert retrieved["nested"]["deeply"]["nested"][2]["mixed"] == "types"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_model_with_complex_validation(self, context):
        """Test context with complex Pydantic models."""
        class ComplexModel(BaseModel):
            data: Dict[str, Any]
            users: list[TestUser]
            metadata: Optional[Dict[str, str]] = None
        
        users = [
            TestUser(name="Alice", age=25),
            TestUser(name="Bob", age=30, email="bob@example.com")
        ]
        
        complex_model = ComplexModel(
            data={"key": "value", "nested": {"inner": 42}},
            users=users,
            metadata={"version": "1.0", "type": "test"}
        )
        
        context.set_validated_data("complex", complex_model)
        retrieved = context.get_validated_data("complex", ComplexModel)
        
        assert retrieved is complex_model
        assert len(retrieved.users) == 2
        assert retrieved.users[0].name == "Alice"
        assert retrieved.metadata["version"] == "1.0"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple context features."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_workflow_state_management(self, context):
        """Test a complete workflow using various state management features."""
        # Set up initial state
        context.set_constant("workflow_version", "1.0")
        context.set_secret("api_key", "secret123")
        context.set_variable("step", 1)
        
        # Store workflow data
        user = TestUser(name="Workflow User", age=35, email="user@workflow.com")
        context.set_validated_data("current_user", user)
        
        # Set typed variable for step tracking
        context.set_typed_variable("current_step", "initialization")
        
        # Cache some expensive computation result
        context.set_cached("expensive_result", {"computed": "value"}, ttl=300)
        
        # Set some output
        context.set_output("status", "in_progress")
        
        # Verify everything is accessible
        assert context.get_constant("workflow_version") == "1.0"
        assert context.get_secret("api_key") == "secret123"
        assert context.get_variable("step") == 1
        
        retrieved_user = context.get_validated_data("current_user", TestUser)
        assert retrieved_user.name == "Workflow User"
        
        assert context.get_typed_variable("current_step", str) == "initialization"
        assert context.get_cached("expensive_result")["computed"] == "value"
        assert context.get_output("status") == "in_progress"

    def test_context_persistence_and_restoration(self):
        """Test that context state persists and can be restored."""
        # Create initial context
        shared_state = {}
        context1 = Context(shared_state=shared_state)
        
        # Set up various types of data
        context1.set_variable("var1", "value1")
        context1.set_typed_variable("typed_var", 42)
        context1.set_constant("const1", "constant_value")
        context1.set_secret("secret1", "secret_value")
        
        if PYDANTIC_AVAILABLE:
            user = TestUser(name="Persistent User", age=40)
            context1.set_validated_data("user", user)
        
        # Create new context with same shared_state
        context2 = Context(shared_state=shared_state)
        
        # Verify data is restored
        assert context2.get_variable("var1") == "value1"
        assert context2.get_typed_variable("typed_var", int) == 42
        assert context2.get_constant("const1") == "constant_value"
        assert context2.get_secret("secret1") == "secret_value"
        
        if PYDANTIC_AVAILABLE:
            restored_user = context2.get_validated_data("user", TestUser)
            assert restored_user.name == "Persistent User"
            assert restored_user.age == 40
        
        # Verify metadata was restored
        assert "typed_var" in context2._typed_var_types
        assert context2._typed_var_types["typed_var"] == int
        
        if PYDANTIC_AVAILABLE:
            assert "user" in context2._validated_types

    def test_error_handling_across_features(self, context):
        """Test error handling across different context features."""
        # Test reserved key errors across different methods
        reserved_keys = ["const_test", "secret_test"]
        
        for key in reserved_keys:
            with pytest.raises(KeyError, match="reserved"):
                context.set_variable(key, "value")
            
            with pytest.raises(KeyError, match="reserved"):
                context.set_typed_variable(key, "value")
            
            if PYDANTIC_AVAILABLE:
                user = TestUser(name="Test", age=25)
                with pytest.raises(KeyError, match="reserved"):
                    context.set_validated_data(key, user)

    @pytest.mark.asyncio
    async def test_concurrent_access_patterns(self, context):
        """Test context behavior under concurrent access patterns."""
        async def worker1():
            for i in range(10):
                context.set_variable(f"worker1_var_{i}", f"value_{i}")
                context.set_cached(f"worker1_cache_{i}", f"cached_{i}")
                await asyncio.sleep(0.001)
        
        async def worker2():
            for i in range(10):
                context.set_state(f"worker2_state_{i}", f"state_{i}")
                context.set_output(f"worker2_output_{i}", f"output_{i}")
                await asyncio.sleep(0.001)
        
        # Run workers concurrently
        await asyncio.gather(worker1(), worker2())
        
        # Verify all data was set correctly
        for i in range(10):
            assert context.get_variable(f"worker1_var_{i}") == f"value_{i}"
            assert context.get_cached(f"worker1_cache_{i}") == f"cached_{i}"
            assert context.get_state(f"worker2_state_{i}") == f"state_{i}"
            assert context.get_output(f"worker2_output_{i}") == f"output_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])