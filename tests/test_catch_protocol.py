"""Tests ensuring the Catch environment adheres to the GymEnv protocol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from catch_jax.catch import Catch, CatchParams, CatchState, NUM_ACTIONS
from catch_jax.gym_env import GymEnv


@pytest.fixture
def key() -> jax.Array:
    """PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


class TestProtocolConformance:
    """Tests verifying Catch conforms to the GymEnv protocol."""

    def test_catch_conforms_to_gym_env_protocol(self) -> None:
        """Catch() must be recognized as a GymEnv by isinstance check."""
        env = Catch()
        assert isinstance(env, GymEnv)

    def test_observation_space_shape_default(self) -> None:
        """Observation space shape must match (rows, columns) for default size."""
        env = Catch()
        obs_space = env.observation_space()
        assert obs_space.shape == (10, 5)

    def test_observation_space_shape_custom(self) -> None:
        """Observation space shape must match custom board sizes."""
        env = Catch(rows=6, columns=3)
        obs_space = env.observation_space()
        assert obs_space.shape == (6, 3)

    def test_observation_space_dtype(self) -> None:
        """Observation dtype must be float32."""
        env = Catch()
        obs_space = env.observation_space()
        assert obs_space.dtype == jnp.float32

    def test_action_space_n(self) -> None:
        """Action space size must be NUM_ACTIONS (3)."""
        env = Catch()
        action_space = env.action_space()
        assert action_space.n == NUM_ACTIONS
        assert action_space.n == 3


class TestConstructor:
    """Tests for environment initialization."""

    def test_catch_no_args(self) -> None:
        """Catch() must work with zero arguments, unlike Pinball which requires a config."""
        env = Catch()
        assert env.rows == 10
        assert env.columns == 5

    def test_catch_custom_sizes(self) -> None:
        """Catch must accept custom rows and columns."""
        env = Catch(rows=8, columns=4)
        assert env.rows == 8
        assert env.columns == 4

    def test_catch_rows_too_small_raises(self) -> None:
        """Catch(rows=1) must raise ValueError.

        With one row, the spawn row and paddle row coincide, making the state
        undefined. Minimum is 2: one for the paddle, one above it for balls.
        """
        with pytest.raises(ValueError):
            Catch(rows=1)

    def test_catch_columns_zero_raises(self) -> None:
        """Catch(columns=0) must raise ValueError.

        The board must have at least one column to place the paddle and balls.
        """
        with pytest.raises(ValueError):
            Catch(columns=0)


class TestReset:
    """Tests for the reset() method."""

    def test_reset_returns_obs_and_state(self, key: jax.Array) -> None:
        """reset() must return (observation, state)."""
        env = Catch()
        obs, state = env.reset(key)
        assert obs is not None
        assert isinstance(state, CatchState)

    def test_reset_observation_shape_dtype(self, key: jax.Array) -> None:
        """reset() observation must have shape (rows, columns) and dtype float32."""
        env = Catch()
        obs, _ = env.reset(key)
        assert obs.shape == (10, 5)
        assert obs.dtype == jnp.float32

    def test_reset_observation_shape_custom(self, key: jax.Array) -> None:
        """reset() observation must match custom board dimensions."""
        env = Catch(rows=6, columns=3)
        obs, _ = env.reset(key)
        assert obs.shape == (6, 3)
        assert obs.dtype == jnp.float32

    def test_reset_state_is_catch_state(self, key: jax.Array) -> None:
        """reset() must return a CatchState NamedTuple."""
        env = Catch()
        _, state = env.reset(key)
        assert isinstance(state, CatchState)

    def test_reset_timestep_zero(self, key: jax.Array) -> None:
        """reset() state must have timestep == 0."""
        env = Catch()
        _, state = env.reset(key)
        assert state.timestep == 0

    def test_reset_paddle_centred(self, key: jax.Array) -> None:
        """reset() paddle must be centred at columns // 2."""
        env = Catch(rows=5, columns=7)
        _, state = env.reset(key)
        assert state.paddle_x == 3  # 7 // 2

    def test_reset_exactly_one_ball(self, key: jax.Array) -> None:
        """reset() must place exactly one ball.

        The ball_mask must have exactly one True entry across all rows.
        """
        env = Catch()
        _, state = env.reset(key)
        assert jnp.sum(state.ball_mask) == 1

    def test_reset_ball_at_row_zero(self, key: jax.Array) -> None:
        """reset() ball must be placed at row 0.

        The single ball appears at the top of the board (row 0) in a uniform
        random column, ready to fall down on the first step.
        """
        env = Catch()
        _, state = env.reset(key)
        assert bool(state.ball_mask[0])
        # All other rows must be empty
        assert not jnp.any(state.ball_mask[1:])


class TestStep:
    """Tests for the step() method."""

    def test_step_returns_six_tuple(self, key: jax.Array) -> None:
        """step() must return a 6-tuple: (obs, state, reward, terminated, truncated, info)."""
        env = Catch()
        obs, state = env.reset(key)
        result = env.step(key, state, 1)
        assert len(result) == 6

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_step_returns_correct_types(self, key: jax.Array, action: int) -> None:
        """step() must return the correct types for all actions."""
        env = Catch()
        obs, state = env.reset(key)
        obs_out, next_state, reward, terminated, truncated, info = env.step(
            key, state, action
        )

        assert obs_out.shape == (10, 5)
        assert obs_out.dtype == jnp.float32
        assert isinstance(next_state, CatchState)
        assert isinstance(reward, jax.Array)
        assert terminated.shape == () and terminated.dtype == jnp.bool_
        assert truncated.shape == () and truncated.dtype == jnp.bool_
        assert isinstance(info, dict)

    def test_step_increments_timestep(self, key: jax.Array) -> None:
        """step() must increment timestep by 1."""
        env = Catch()
        obs, state = env.reset(key)
        assert state.timestep == 0

        _, state, _, _, _, _ = env.step(key, state, 1)
        assert state.timestep == 1

        _, state, _, _, _, _ = env.step(key, state, 1)
        assert state.timestep == 2

    def test_step_info_is_empty(self, key: jax.Array) -> None:
        """step() info dict must always be empty."""
        env = Catch()
        obs, state = env.reset(key)
        _, _, _, _, _, info = env.step(key, state, 1)
        assert info == {}

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_step_terminated_truncated_dtypes(self, key: jax.Array, action: int) -> None:
        """step() terminated and truncated must be scalar bool arrays."""
        env = Catch()
        obs, state = env.reset(key)
        _, _, _, terminated, truncated, _ = env.step(key, state, action)

        assert terminated.shape == ()
        assert terminated.dtype == jnp.bool_
        assert truncated.shape == ()
        assert truncated.dtype == jnp.bool_


class TestTermination:
    """Tests for episode termination signals."""

    def test_never_terminates_long_rollout(self, key: jax.Array) -> None:
        """Catch is a continuing environment: terminated must always be False.

        Even after many steps (200), a single rollout must never signal
        termination. Only truncation (due to max_steps_in_episode) can stop
        an episode.
        """
        env = Catch()
        obs, state = env.reset(key)

        for step_i in range(200):
            _, state, _, terminated, _, _ = env.step(key, state, 1)
            assert not bool(terminated), (
                f"terminated must be False (continuing env), but got True at step {step_i}"
            )

    def test_truncates_at_max_steps_in_episode(self, key: jax.Array) -> None:
        """step() must truncate (only) when timestep >= max_steps_in_episode.

        This mirrors the test_pinball_protocol.py style: step N-1 times and
        verify truncated is False each time, then step once more and verify
        truncated is True at exactly timestep == N.
        """
        env = Catch()
        max_steps = 3
        obs, state = env.reset(key)
        params = CatchParams(max_steps_in_episode=max_steps)

        # Step 1 through N-1: truncated must be False
        for expected_timestep in range(1, max_steps):
            _, state, _, _, truncated, _ = env.step(key, state, 1, params)
            assert state.timestep == expected_timestep
            assert not bool(truncated), (
                f"truncated must be False before max_steps, at timestep {expected_timestep}"
            )

        # Step N: truncated must be True
        _, state, _, _, truncated, _ = env.step(key, state, 1, params)
        assert state.timestep == max_steps
        assert bool(truncated), (
            f"truncated must be True when timestep >= max_steps_in_episode"
        )
