"""Tests for Catch environment dynamics and invariants."""

from __future__ import annotations

import jax
import jax.lax
import jax.numpy as jnp
import pytest

from catch_jax.catch import Catch, CatchParams, CatchState, NUM_ACTIONS


@pytest.fixture
def key() -> jax.Array:
    """PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def deterministic_key() -> jax.Array:
    """A fixed PRNG key for statistical tests (guaranteed reproducibility)."""
    return jax.random.PRNGKey(42)


class TestPaddleMovement:
    """Tests for paddle left/stay/right movement and clipping."""

    def test_left_action_moves_paddle_left(self, key: jax.Array) -> None:
        """LEFT action (0) moves the paddle by -1."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)
        # Place paddle at column 2 by hand
        state = state._replace(paddle_x=jnp.asarray(2, dtype=jnp.int32))

        _, next_state, _, _, _, _ = env.step(key, state, 0)  # action LEFT
        assert next_state.paddle_x == 1

    def test_stay_action_keeps_paddle(self, key: jax.Array) -> None:
        """STAY action (1) leaves the paddle in its current column."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(2, dtype=jnp.int32))

        _, next_state, _, _, _, _ = env.step(key, state, 1)  # action STAY
        assert next_state.paddle_x == 2

    def test_right_action_moves_paddle_right(self, key: jax.Array) -> None:
        """RIGHT action (2) moves the paddle by +1."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(2, dtype=jnp.int32))

        _, next_state, _, _, _, _ = env.step(key, state, 2)  # action RIGHT
        assert next_state.paddle_x == 3

    def test_left_clips_at_zero(self, key: jax.Array) -> None:
        """Paddle clips at column 0 under repeated LEFT."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(0, dtype=jnp.int32))

        for _ in range(5):  # Push left 5 times
            _, state, _, _, _, _ = env.step(key, state, 0)  # LEFT
            assert state.paddle_x == 0

    def test_right_clips_at_boundary(self, key: jax.Array) -> None:
        """Paddle clips at column (columns - 1) under repeated RIGHT."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)
        # Rightmost column for a 5-column board is 4
        state = state._replace(paddle_x=jnp.asarray(4, dtype=jnp.int32))

        for _ in range(5):  # Push right 5 times
            _, state, _, _, _, _ = env.step(key, state, 2)  # RIGHT
            assert state.paddle_x == 4


class TestBallDescent:
    """Tests for ball movement and no-spawn behavior."""

    def test_ball_descends_one_row_per_step(self, key: jax.Array) -> None:
        """With spawn_probability=0, the ball descends one row per step until it resolves at the paddle row."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)

        # Disable spawning so the initial ball is the only one
        params = CatchParams(spawn_probability=0.0)

        # Track which row the ball is in
        ball_rows = []
        for step_i in range(10):
            # Find the row with the ball
            ball_row = jnp.where(state.ball_mask)[0]
            if len(ball_row) > 0:
                ball_rows.append(int(ball_row[0]))

            _, state, _, _, _, _ = env.step(key, state, 1, params)

        # The ball should start at row 0 and descend: 0, 1, 2, 3, 4
        # At row 4 (paddle row), it is resolved and removed, so no row 5
        assert ball_rows[0] == 0
        for i in range(1, len(ball_rows)):
            assert ball_rows[i] == ball_rows[i - 1] + 1

    def test_no_new_ball_with_zero_spawn_probability(self, key: jax.Array) -> None:
        """With spawn_probability=0, no new ball spawns after the initial one resolves."""
        env = Catch(rows=5, columns=5)
        obs, state = env.reset(key)
        params = CatchParams(spawn_probability=0.0)

        # Step until the initial ball resolves (reaches paddle row and is removed)
        # With 5 rows, it takes 4 steps to descend from row 0 to row 4
        for _ in range(4):
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        # Now step many more times; no new ball should ever appear
        for _ in range(20):
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            assert jnp.sum(state.ball_mask) == 0, (
                "Board must be empty with spawn_probability=0 after initial ball resolves"
            )


class TestRewardCorrectness:
    """Tests for reward calculation on catch and miss."""

    def test_reward_positive_on_catch(self, key: jax.Array) -> None:
        """Reward is +1.0 when a ball lands in the paddle's column."""
        env = Catch(rows=5, columns=5)
        # Manually construct a state with a ball at row 3 (one above paddle row 4)
        # in column 2, and paddle at column 2
        state = CatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 0, 2, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, False, True, False], dtype=jnp.bool_),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = CatchParams(spawn_probability=0.0)
        _, _, reward, _, _, _ = env.step(key, state, 1, params)

        assert float(reward) == 1.0

    def test_reward_negative_on_miss(self, key: jax.Array) -> None:
        """Reward is -1.0 when a ball lands outside the paddle's column."""
        env = Catch(rows=5, columns=5)
        state = CatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 0, 3, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, False, True, False], dtype=jnp.bool_),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = CatchParams(spawn_probability=0.0)
        _, _, reward, _, _, _ = env.step(key, state, 1, params)

        assert float(reward) == -1.0

    def test_reward_zero_no_resolution(self, key: jax.Array) -> None:
        """Reward is 0.0 when no ball reaches the paddle row."""
        env = Catch(rows=5, columns=5)
        state = CatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 2, 0, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, True, False, False], dtype=jnp.bool_),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = CatchParams(spawn_probability=0.0)
        _, _, reward, _, _, _ = env.step(key, state, 1, params)

        assert float(reward) == 0.0


class TestResolvedBallRemoval:
    """Tests verifying that resolved balls are removed before rendering."""

    def test_resolved_ball_not_in_observation(self, key: jax.Array) -> None:
        """A ball resolved this step does not appear in the observation (rendered after removal)."""
        env = Catch(rows=5, columns=5)
        # Ball at row 3 (one above paddle row 4), column 2
        state = CatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 0, 2, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, False, True, False], dtype=jnp.bool_),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = CatchParams(spawn_probability=0.0)
        obs, next_state, _, _, _, _ = env.step(key, state, 1, params)

        # After the step, the ball has been caught and removed
        # The paddle row (row 4) should not show the ball; it may show the paddle
        paddle_row_obs = obs[4]
        # The paddle is at column 2, so paddle_row_obs[2] should be 1 (paddle)
        # and no other cell in the paddle row should be 1 from a ball
        # (the ball was caught and removed)
        assert float(paddle_row_obs[2]) == 1.0
        # Other columns in the paddle row must be 0 (no ball, no paddle there)
        for col in range(5):
            if col != 2:
                assert float(paddle_row_obs[col]) == 0.0


class TestInvariants:
    """Tests for state invariants across long rollouts."""

    def test_invariant_at_most_one_ball_per_row(self, deterministic_key: jax.Array) -> None:
        """At most one ball occupies any row, over 300 steps at spawn_probability=1.0.

        This invariant is what makes the compact fixed-shape state representation possible.
        """
        env = Catch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = CatchParams(spawn_probability=1.0)

        for step_i in range(300):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            _, state, _, _, _, _ = env.step(key, state, action, params)

            # Check invariant: each row has at most one ball
            ball_count_per_row = jnp.sum(state.ball_mask)
            assert ball_count_per_row <= env.rows - 1, (
                f"At most {env.rows - 1} balls can exist (one per row except paddle row); "
                f"got {ball_count_per_row} at step {step_i}"
            )

    def test_invariant_paddle_row_never_holds_ball(self, deterministic_key: jax.Array) -> None:
        """The paddle row never holds a ball after a step; resolved balls are removed immediately."""
        env = Catch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = CatchParams(spawn_probability=1.0)

        for step_i in range(300):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            _, state, _, _, _, _ = env.step(key, state, action, params)

            # Paddle row is row (rows - 1)
            assert not bool(state.ball_mask[env.rows - 1]), (
                f"Paddle row (row {env.rows - 1}) must never hold a ball after step; "
                f"violated at step {step_i}"
            )

    def test_steady_state_at_spawn_probability_one(self, deterministic_key: jax.Array) -> None:
        """At spawn_probability=1.0, the board reaches a steady state of rows-1 balls (one per non-paddle row)."""
        env = Catch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = CatchParams(spawn_probability=1.0)

        # Warm up for a few steps to let the state settle
        for _ in range(20):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        # Now check steady state over many steps
        for step_i in range(100):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            ball_count = jnp.sum(state.ball_mask)
            assert ball_count == (env.rows - 1), (
                f"At steady state with p=1.0, must hold {env.rows - 1} balls; "
                f"got {ball_count} at step {step_i}"
            )


class TestSpawnProbability:
    """Tests for spawn probability behavior."""

    def test_spawn_probability_one_always_spawns(self, deterministic_key: jax.Array) -> None:
        """With spawn_probability=1.0, row 0 holds a ball after every step."""
        env = Catch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = CatchParams(spawn_probability=1.0)

        for step_i in range(10):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)  # STAY
            # After each step with p=1.0, row 0 must have a ball
            assert bool(state.ball_mask[0]), (
                f"Row 0 must have a ball after every step with spawn_probability=1.0; "
                f"violated at step {step_i}"
            )

    def test_spawn_probability_zero_never_spawns(self, deterministic_key: jax.Array) -> None:
        """With spawn_probability=0.0, no ball spawns after the initial one resolves."""
        env = Catch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = CatchParams(spawn_probability=0.0)

        # Let the initial ball resolve
        for _ in range(5):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        # Now verify no new ball ever spawns
        for step_i in range(50):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            assert jnp.sum(state.ball_mask) == 0, (
                f"No new ball must spawn with spawn_probability=0.0; "
                f"violated at step {step_i}"
            )


class TestStatisticalRNG:
    """Tests for random spawn probability and column distribution."""

    def _rollout_spawn_events(self, env: Catch, key: jax.Array, params: CatchParams, num_steps: int):
        """Run ``num_steps`` STAY actions via jitted ``lax.scan``, returning per-step
        ``(ball_mask[0], ball_cols[0])``.

        Compiling the rollout once avoids per-call dispatch overhead, which matters
        at the ~20,000-step sample sizes the statistical tests below need.
        """
        _, state = env.reset(key)
        keys = jax.random.split(key, num_steps)

        def body(state, step_key):
            _, next_state, _, _, _, _ = env.step(step_key, state, 1, params)
            return next_state, (next_state.ball_mask[0], next_state.ball_cols[0])

        _, (spawned, columns) = jax.jit(lambda s, ks: jax.lax.scan(body, s, ks))(state, keys)
        return spawned, columns

    def test_spawn_rate_matches_probability(self, deterministic_key: jax.Array) -> None:
        """Empirical spawn rate over ~20,000 steps matches spawn_probability=0.3 within ±0.02."""
        env = Catch(rows=10, columns=5)
        params = CatchParams(spawn_probability=0.3)
        spawned, _ = self._rollout_spawn_events(env, deterministic_key, params, num_steps=20000)

        empirical_rate = float(jnp.mean(spawned))
        tolerance = 0.02
        assert abs(empirical_rate - 0.3) < tolerance, (
            f"Spawn rate {empirical_rate} deviates from 0.3 by more than {tolerance}"
        )

    def test_spawn_columns_uniform(self, deterministic_key: jax.Array) -> None:
        """Spawned ball columns are uniform: each column's frequency over ~20,000 steps
        is within ±0.03 of 1/columns.
        """
        env = Catch(rows=10, columns=5)
        params = CatchParams(spawn_probability=0.3)
        spawned, columns = self._rollout_spawn_events(
            env, deterministic_key, params, num_steps=20000
        )

        spawn_columns = columns[spawned]
        num_spawns = int(jnp.sum(spawned))
        for col in range(env.columns):
            col_count = int(jnp.sum(spawn_columns == col))
            col_frequency = col_count / num_spawns if num_spawns else 0.0
            expected_frequency = 1.0 / env.columns
            tolerance = 0.03
            assert abs(col_frequency - expected_frequency) < tolerance, (
                f"Column {col} frequency {col_frequency} deviates from {expected_frequency} "
                f"by more than {tolerance}"
            )


class TestNonDefaultBoardSizes:
    """Tests for correct behavior at non-default board dimensions."""

    def test_board_size_6x3(self, key: jax.Array) -> None:
        """6x3 board works without shape errors (csuite's bug (b) case)."""
        env = Catch(rows=6, columns=3)
        obs, state = env.reset(key)

        assert obs.shape == (6, 3)
        for _ in range(50):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            obs, state, _, _, _, _ = env.step(key, state, action)
            assert obs.shape == (6, 3)

    def test_board_size_2x2(self, key: jax.Array) -> None:
        """2x2 is the minimum viable board: one paddle row, one spawn row."""
        env = Catch(rows=2, columns=2)
        obs, state = env.reset(key)

        assert obs.shape == (2, 2)
        for _ in range(50):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            obs, state, _, _, _, _ = env.step(key, state, action)
            assert obs.shape == (2, 2)

    def test_board_size_20x7(self, key: jax.Array) -> None:
        """Large 20x7 board must work correctly."""
        env = Catch(rows=20, columns=7)
        obs, state = env.reset(key)

        assert obs.shape == (20, 7)
        for _ in range(100):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            obs, state, _, _, _, _ = env.step(key, state, action)
            assert obs.shape == (20, 7)


class TestJITAndVmap:
    """Smoke tests for JAX transformations (jit, vmap, scan)."""

    def test_jit_step(self, key: jax.Array) -> None:
        """jax.jit wrapping step must work without errors."""
        env = Catch()
        obs, state = env.reset(key)

        jitted_step = jax.jit(env.step)
        obs, state, reward, terminated, truncated, info = jitted_step(key, state, 1)

        assert obs.shape == (10, 5)
        assert isinstance(state, CatchState)

    def test_vmap_reset_over_keys(self, key: jax.Array) -> None:
        """vmap over reset keys produces batched observations and states."""
        env = Catch()
        keys = jax.random.split(key, 8)

        vmapped_reset = jax.vmap(env.reset)
        obs_batch, state_batch = vmapped_reset(keys)

        assert obs_batch.shape == (8, 10, 5)
        assert state_batch.paddle_x.shape == (8,)

    def test_vmap_step_over_params(self, key: jax.Array) -> None:
        """vmap over CatchParams (varying spawn_probability) works without retracing."""
        env = Catch()
        obs, state = env.reset(key)

        # Batch of spawn probabilities to sweep under vmap.
        spawn_probs = jnp.array([0.0, 0.1, 0.5, 1.0])

        # vmap step over the params batch
        def step_with_params(spawn_prob):
            params = CatchParams(spawn_probability=spawn_prob)
            _, next_state, reward, _, _, _ = env.step(key, state, 1, params)
            return next_state, reward

        vmapped_step = jax.vmap(step_with_params)
        next_states_batch, rewards_batch = vmapped_step(spawn_probs)

        assert rewards_batch.shape == (4,)
        assert next_states_batch.paddle_x.shape == (4,)

    def test_lax_scan_rollout(self, key: jax.Array) -> None:
        """lax.scan over step works: the standard jitted-rollout pattern."""
        env = Catch()
        obs, state = env.reset(key)

        def step_fn(state, key):
            action = 1  # STAY
            obs, next_state, reward, terminated, truncated, info = env.step(
                key, state, action
            )
            return next_state, (reward, obs)

        keys = jax.random.split(key, 50)
        final_state, (rewards, obs_seq) = jax.lax.scan(step_fn, state, keys)

        assert rewards.shape == (50,)
        assert obs_seq.shape == (50, 10, 5)
        assert isinstance(final_state, CatchState)
