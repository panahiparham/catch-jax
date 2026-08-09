"""Shared test configuration.

Unlike pinball-jax, we do NOT enable JAX float64 here. Catch's dynamics are
integer-exact: the paddle and ball positions are int32, ball descent is just a
shift by one row per step, and rewards are exact multiples of ±1.0. There is no
floating-point physics that could lose precision, so float32 is exact and
parity testing does not require float64 to compare against the numpy reference
at machine precision.

(pinball-jax enables x64 because its physics uses floating-point forces,
velocities, and collisions, where float32 accumulates error; catch has no such
issue.)
"""

from __future__ import annotations

import jax
import pytest

from catch_jax.catch import Catch, CatchParams
from catch_jax.dancing_catch import DancingCatch, DancingCatchParams


@pytest.fixture
def env() -> Catch:
    """Default Catch environment (10 rows, 5 columns)."""
    return Catch()


@pytest.fixture
def key() -> jax.Array:
    """PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def deterministic_key() -> jax.Array:
    """A fixed PRNG key for statistical tests (guaranteed reproducibility)."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def dancing_env() -> DancingCatch:
    """Default DancingCatch environment (10 rows, 5 columns)."""
    return DancingCatch()
