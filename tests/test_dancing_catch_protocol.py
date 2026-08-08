"""Tests ensuring the DancingCatch environment adheres to the GymEnv protocol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from catch_jax.dancing_catch import DancingCatch, DancingCatchParams, DancingCatchState, NUM_ACTIONS
from catch_jax.gym_env import GymEnv


@pytest.fixture
def key() -> jax.Array:
    """PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


class TestProtocolConformance:
    """Tests verifying DancingCatch conforms to the GymEnv protocol."""

    def test_dancing_catch_conforms_to_gym_env_protocol(self) -> None:
        """DancingCatch() must be recognized as a GymEnv by isinstance check."""
        env = DancingCatch()
        assert isinstance(env, GymEnv)

    @pytest.mark.parametrize("rows,columns", [(10, 5), (6, 3), (2, 2), (20, 7)])
    def test_observation_space_shape(self, rows: int, columns: int) -> None:
        """Observation space shape must be (rows*columns,) - 1-D, not 2-D like Catch."""
        env = DancingCatch(rows=rows, columns=columns)
        obs_space = env.observation_space()
        expected_shape = (rows * columns,)
        assert obs_space.shape == expected_shape
        assert len(obs_space.shape) == 1, "Observation must be 1-D for DancingCatch"

    def test_observation_space_dtype(self) -> None:
        """Observation dtype must be float32."""
        env = DancingCatch()
        obs_space = env.observation_space()
        assert obs_space.dtype == jnp.float32

    def test_action_space_n(self) -> None:
        """Action space size must be NUM_ACTIONS (3)."""
        env = DancingCatch()
        action_space = env.action_space()
        assert action_space.n == NUM_ACTIONS
        assert action_space.n == 3


class TestConstructor:
    """Tests for environment initialization."""

    def test_dancing_catch_no_args(self) -> None:
        """DancingCatch() must work with zero arguments."""
        env = DancingCatch()
        assert env.rows == 10
        assert env.columns == 5

    def test_dancing_catch_custom_sizes(self) -> None:
        """DancingCatch must accept custom rows and columns."""
        env = DancingCatch(rows=8, columns=4)
        assert env.rows == 8
        assert env.columns == 4

    def test_dancing_catch_rows_too_small_raises(self) -> None:
        """DancingCatch(rows=1) raises ValueError: the spawn and paddle rows would coincide."""
        with pytest.raises(ValueError):
            DancingCatch(rows=1)

    def test_dancing_catch_columns_zero_raises(self) -> None:
        """DancingCatch(columns=0) raises ValueError."""
        with pytest.raises(ValueError):
            DancingCatch(columns=0)


class TestReset:
    """Tests for the reset() method."""

    def test_reset_returns_obs_and_state(self, key: jax.Array) -> None:
        """reset() must return (observation, state)."""
        env = DancingCatch()
        obs, state = env.reset(key)
        assert obs is not None
        assert isinstance(state, DancingCatchState)

    @pytest.mark.parametrize("rows,columns", [(10, 5), (6, 3), (2, 2), (20, 7)])
    def test_reset_observation_shape_dtype(self, key: jax.Array, rows: int, columns: int) -> None:
        """reset() observation must have shape (rows*columns,) and dtype float32."""
        env = DancingCatch(rows=rows, columns=columns)
        obs, _ = env.reset(key)
        expected_shape = (rows * columns,)
        assert obs.shape == expected_shape
        assert obs.dtype == jnp.float32

    def test_reset_state_is_dancing_catch_state(self, key: jax.Array) -> None:
        """reset() must return a DancingCatchState NamedTuple."""
        env = DancingCatch()
        _, state = env.reset(key)
        assert isinstance(state, DancingCatchState)

    def test_reset_timestep_zero(self, key: jax.Array) -> None:
        """reset() state must have timestep == 0."""
        env = DancingCatch()
        _, state = env.reset(key)
        assert state.timestep == 0

    def test_reset_time_since_swap_zero(self, key: jax.Array) -> None:
        """reset() state must have time_since_swap == 0."""
        env = DancingCatch()
        _, state = env.reset(key)
        assert state.time_since_swap == 0

    def test_reset_shuffle_idx_is_identity(self, key: jax.Array) -> None:
        """reset() shuffle_idx must equal arange(rows*columns) with integer dtype."""
        env = DancingCatch(rows=6, columns=3)
        _, state = env.reset(key)
        expected_shuffle_idx = jnp.arange(6 * 3, dtype=jnp.int32)
        assert jnp.array_equal(state.shuffle_idx, expected_shuffle_idx)
        assert state.shuffle_idx.dtype == jnp.int32

    def test_reset_paddle_centred(self, key: jax.Array) -> None:
        """reset() paddle must be centred at columns // 2."""
        env = DancingCatch(rows=5, columns=7)
        _, state = env.reset(key)
        assert state.paddle_x == 3  # 7 // 2

    def test_reset_exactly_one_ball(self, key: jax.Array) -> None:
        """reset() places exactly one ball (one True entry in ball_mask)."""
        env = DancingCatch()
        _, state = env.reset(key)
        assert jnp.sum(state.ball_mask) == 1

    def test_reset_ball_at_row_zero(self, key: jax.Array) -> None:
        """reset() places the ball at row 0; all other rows are empty."""
        env = DancingCatch()
        _, state = env.reset(key)
        assert bool(state.ball_mask[0])
        assert not jnp.any(state.ball_mask[1:])


class TestStep:
    """Tests for the step() method."""

    def test_step_returns_six_tuple(self, key: jax.Array) -> None:
        """step() must return a 6-tuple: (obs, state, reward, terminated, truncated, info)."""
        env = DancingCatch()
        obs, state = env.reset(key)
        result = env.step(key, state, 1)
        assert len(result) == 6

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_step_returns_correct_types(self, key: jax.Array, action: int) -> None:
        """step() must return the correct types for all actions."""
        env = DancingCatch()
        obs, state = env.reset(key)
        obs_out, next_state, reward, terminated, truncated, info = env.step(
            key, state, action
        )

        expected_obs_shape = (10 * 5,)
        assert obs_out.shape == expected_obs_shape
        assert obs_out.dtype == jnp.float32
        assert isinstance(next_state, DancingCatchState)
        assert isinstance(reward, jax.Array)
        assert terminated.shape == () and terminated.dtype == jnp.bool_
        assert truncated.shape == () and truncated.dtype == jnp.bool_
        assert isinstance(info, dict)

    def test_step_increments_timestep(self, key: jax.Array) -> None:
        """step() must increment timestep by 1."""
        env = DancingCatch()
        obs, state = env.reset(key)
        assert state.timestep == 0

        _, state, _, _, _, _ = env.step(key, state, 1)
        assert state.timestep == 1

        _, state, _, _, _, _ = env.step(key, state, 1)
        assert state.timestep == 2

    def test_step_info_is_empty(self, key: jax.Array) -> None:
        """step() info dict must always be empty."""
        env = DancingCatch()
        obs, state = env.reset(key)
        _, _, _, _, _, info = env.step(key, state, 1)
        assert info == {}

    @pytest.mark.parametrize("action", range(NUM_ACTIONS))
    def test_step_terminated_truncated_dtypes(self, key: jax.Array, action: int) -> None:
        """step() terminated and truncated must be scalar bool arrays."""
        env = DancingCatch()
        obs, state = env.reset(key)
        _, _, _, terminated, truncated, _ = env.step(key, state, action)

        assert terminated.shape == ()
        assert terminated.dtype == jnp.bool_
        assert truncated.shape == ()
        assert truncated.dtype == jnp.bool_


class TestTermination:
    """Tests for episode termination signals."""

    def test_never_terminates_long_rollout(self, key: jax.Array) -> None:
        """DancingCatch is continuing: terminated stays False over a 200-step rollout."""
        env = DancingCatch()
        obs, state = env.reset(key)

        for step_i in range(200):
            _, state, _, terminated, _, _ = env.step(key, state, 1)
            assert not bool(terminated), (
                f"terminated must be False (continuing env), but got True at step {step_i}"
            )

    def test_truncates_at_max_steps_in_episode(self, key: jax.Array) -> None:
        """step() truncates exactly when timestep >= max_steps_in_episode."""
        env = DancingCatch()
        max_steps = 3
        obs, state = env.reset(key)
        params = DancingCatchParams(max_steps_in_episode=max_steps)

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
