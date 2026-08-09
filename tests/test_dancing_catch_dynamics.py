"""Tests for DancingCatch environment dynamics and invariants."""

from __future__ import annotations

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import pytest

from catch_jax.dancing_catch import (
    DancingCatch,
    DancingCatchParams,
    DancingCatchState,
    NUM_ACTIONS,
)


@pytest.fixture
def key() -> jax.Array:
    """PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def deterministic_key() -> jax.Array:
    """A fixed PRNG key for statistical tests (guaranteed reproducibility)."""
    return jax.random.PRNGKey(42)


# ============================================================================
# Section 1: Dynamics Inherited from Catch
# ============================================================================
# These tests verify that the core Catch dynamics are preserved despite the
# observation permutation. Tests access state.ball_cols, state.ball_mask, and
# state.paddle_x directly rather than reading the permuted observation.


class TestInheritedPaddleMovement:
    """Tests for paddle left/stay/right movement and clipping."""

    def test_left_action_moves_paddle_left(self, key: jax.Array) -> None:
        """LEFT action (0) moves the paddle by -1."""
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(2, dtype=jnp.int32))

        _, next_state, _, _, _, _ = env.step(key, state, 0)  # action LEFT
        assert next_state.paddle_x == 1

    def test_stay_action_keeps_paddle(self, key: jax.Array) -> None:
        """STAY action (1) leaves the paddle in its current column."""
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(2, dtype=jnp.int32))

        _, next_state, _, _, _, _ = env.step(key, state, 1)  # action STAY
        assert next_state.paddle_x == 2

    def test_right_action_moves_paddle_right(self, key: jax.Array) -> None:
        """RIGHT action (2) moves the paddle by +1."""
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(2, dtype=jnp.int32))

        _, next_state, _, _, _, _ = env.step(key, state, 2)  # action RIGHT
        assert next_state.paddle_x == 3

    def test_left_clips_at_zero(self, key: jax.Array) -> None:
        """Paddle clips at column 0 under repeated LEFT."""
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(0, dtype=jnp.int32))

        for _ in range(5):
            _, state, _, _, _, _ = env.step(key, state, 0)  # LEFT
            assert state.paddle_x == 0

    def test_right_clips_at_boundary(self, key: jax.Array) -> None:
        """Paddle clips at column (columns - 1) under repeated RIGHT."""
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        state = state._replace(paddle_x=jnp.asarray(4, dtype=jnp.int32))

        for _ in range(5):
            _, state, _, _, _, _ = env.step(key, state, 2)  # RIGHT
            assert state.paddle_x == 4


class TestInheritedBallDescent:
    """Tests for ball movement and no-spawn behavior."""

    def test_ball_descends_one_row_per_step(self, key: jax.Array) -> None:
        """With spawn_probability=0, the ball descends one row per step."""
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0)

        ball_rows = []
        for step_i in range(10):
            ball_row = jnp.where(state.ball_mask)[0]
            if len(ball_row) > 0:
                ball_rows.append(int(ball_row[0]))

            _, state, _, _, _, _ = env.step(key, state, 1, params)

        assert ball_rows[0] == 0
        for i in range(1, len(ball_rows)):
            assert ball_rows[i] == ball_rows[i - 1] + 1

    def test_no_new_ball_with_zero_spawn_probability(
        self, key: jax.Array
    ) -> None:
        """With spawn_probability=0, no new ball spawns after the initial
        one resolves.
        """
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0)

        for _ in range(4):
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        for _ in range(20):
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            assert jnp.sum(state.ball_mask) == 0, (
                "Board must be empty with spawn_probability=0 after "
                "initial ball resolves"
            )


class TestInheritedRewardCorrectness:
    """Tests for reward calculation on catch and miss."""

    def test_reward_positive_on_catch(self, key: jax.Array) -> None:
        """Reward is +1.0 when a ball lands in the paddle's column."""
        env = DancingCatch(rows=5, columns=5)
        state = DancingCatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 0, 2, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, False, True, False], dtype=jnp.bool_),
            shuffle_idx=jnp.arange(25, dtype=jnp.int32),
            time_since_swap=jnp.asarray(0, dtype=jnp.int32),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = DancingCatchParams(spawn_probability=0.0)
        _, _, reward, _, _, _ = env.step(key, state, 1, params)

        assert float(reward) == 1.0

    def test_reward_negative_on_miss(self, key: jax.Array) -> None:
        """Reward is -1.0 when a ball lands outside the paddle's column."""
        env = DancingCatch(rows=5, columns=5)
        state = DancingCatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 0, 3, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, False, True, False], dtype=jnp.bool_),
            shuffle_idx=jnp.arange(25, dtype=jnp.int32),
            time_since_swap=jnp.asarray(0, dtype=jnp.int32),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = DancingCatchParams(spawn_probability=0.0)
        _, _, reward, _, _, _ = env.step(key, state, 1, params)

        assert float(reward) == -1.0

    def test_reward_zero_no_resolution(self, key: jax.Array) -> None:
        """Reward is 0.0 when no ball reaches the paddle row."""
        env = DancingCatch(rows=5, columns=5)
        state = DancingCatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 2, 0, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, True, False, False], dtype=jnp.bool_),
            shuffle_idx=jnp.arange(25, dtype=jnp.int32),
            time_since_swap=jnp.asarray(0, dtype=jnp.int32),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = DancingCatchParams(spawn_probability=0.0)
        _, _, reward, _, _, _ = env.step(key, state, 1, params)

        assert float(reward) == 0.0


class TestInheritedResolvedBallRemoval:
    """Tests verifying that resolved balls are removed before rendering."""

    def test_resolved_ball_not_in_state(self, key: jax.Array) -> None:
        """A ball resolved this step does not appear in the next state."""
        env = DancingCatch(rows=5, columns=5)
        state = DancingCatchState(
            paddle_x=jnp.asarray(2, dtype=jnp.int32),
            ball_cols=jnp.array([0, 0, 0, 2, 0], dtype=jnp.int32),
            ball_mask=jnp.array([False, False, False, True, False], dtype=jnp.bool_),
            shuffle_idx=jnp.arange(25, dtype=jnp.int32),
            time_since_swap=jnp.asarray(0, dtype=jnp.int32),
            timestep=jnp.asarray(0, dtype=jnp.int32),
        )

        params = DancingCatchParams(spawn_probability=0.0)
        obs, next_state, _, _, _, _ = env.step(key, state, 1, params)

        assert not bool(next_state.ball_mask[env.rows - 1]), (
            "Paddle row must not hold a ball after resolution"
        )


class TestInheritedInvariants:
    """Tests for state invariants across long rollouts."""

    def test_invariant_at_most_one_ball_per_row(
        self, deterministic_key: jax.Array
    ) -> None:
        """At most one ball occupies any row, over 300 steps at
        spawn_probability=1.0.
        """
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=1.0)

        for step_i in range(300):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            _, state, _, _, _, _ = env.step(key, state, action, params)

            ball_count_per_row = jnp.sum(state.ball_mask)
            assert ball_count_per_row <= env.rows - 1, (
                f"At most {env.rows - 1} balls can exist; "
                f"got {ball_count_per_row} at step {step_i}"
            )

    def test_invariant_paddle_row_never_holds_ball(
        self, deterministic_key: jax.Array
    ) -> None:
        """The paddle row never holds a ball after a step."""
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=1.0)

        for step_i in range(300):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            _, state, _, _, _, _ = env.step(key, state, action, params)

            assert not bool(state.ball_mask[env.rows - 1]), (
                f"Paddle row (row {env.rows - 1}) must never hold a ball after step; "
                f"violated at step {step_i}"
            )

    def test_steady_state_at_spawn_probability_one(
        self, deterministic_key: jax.Array
    ) -> None:
        """At spawn_probability=1.0, the board reaches steady state of
        rows-1 balls.
        """
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=1.0)

        for _ in range(20):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        for step_i in range(100):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            ball_count = jnp.sum(state.ball_mask)
            assert ball_count == (env.rows - 1), (
                f"At steady state with p=1.0, must hold {env.rows - 1} balls; "
                f"got {ball_count} at step {step_i}"
            )


class TestInheritedSpawnProbability:
    """Tests for spawn probability behavior."""

    def test_spawn_probability_one_always_spawns(
        self, deterministic_key: jax.Array
    ) -> None:
        """With spawn_probability=1.0, row 0 holds a ball after every step."""
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=1.0)

        for step_i in range(10):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            assert bool(state.ball_mask[0]), (
                f"Row 0 must have a ball after every step with spawn_probability=1.0; "
                f"violated at step {step_i}"
            )

    def test_spawn_probability_zero_never_spawns(
        self, deterministic_key: jax.Array
    ) -> None:
        """With spawn_probability=0.0, no ball spawns after the initial
        one resolves.
        """
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0)

        for _ in range(5):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        for step_i in range(50):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)
            assert jnp.sum(state.ball_mask) == 0, (
                f"No new ball must spawn with spawn_probability=0.0; "
                f"violated at step {step_i}"
            )


# ============================================================================
# Section 2: The Permutation - Specific to DancingCatch
# ============================================================================
# These tests verify that the observation permutation works correctly, that
# swaps occur at the right time and magnitude, and that the observation
# reflects the permuted board state.


class TestPermutationInvariance:
    """Tests for the shuffle_idx permutation invariant."""

    def test_shuffle_idx_is_permutation_long_rollout(
        self, deterministic_key: jax.Array
    ) -> None:
        """shuffle_idx must be a permutation of arange(rows*columns) at every step."""
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.1, swap_every=10)

        for step_i in range(200):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            _, state, _, _, _, _ = env.step(key, state, action, params)

            sorted_shuffle = jnp.sort(state.shuffle_idx)
            expected = jnp.arange(25, dtype=jnp.int32)
            assert jnp.array_equal(sorted_shuffle, expected), (
                f"shuffle_idx must be a permutation at step {step_i}"
            )

    def test_shuffle_idx_stays_identity_when_swap_disabled(
        self, key: jax.Array
    ) -> None:
        """With swap_every larger than rollout, shuffle_idx stays the
        identity.
        """
        env = DancingCatch(rows=5, columns=5)
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0, swap_every=10000)

        for _ in range(50):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        expected_shuffle_idx = jnp.arange(25, dtype=jnp.int32)
        assert jnp.array_equal(state.shuffle_idx, expected_shuffle_idx)


class TestObservationConsistency:
    """Tests for observation-to-board consistency."""

    def test_observation_equals_permuted_board(
        self, deterministic_key: jax.Array
    ) -> None:
        """At every step, obs == expected_board.flatten()[shuffle_idx].

        Runs a 200-step rollout with swap_every=1 and spawn_probability=0.3 so
        swaps and spawns both fire constantly.
        """
        env = DancingCatch(rows=4, columns=3)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.3, swap_every=1)

        for step_i in range(200):
            # Reconstruct the expected board from state fields
            board_np = np.zeros((env.rows, env.columns), dtype=np.float32)

            # Mark balls
            for row_idx in range(env.rows):
                if bool(state.ball_mask[row_idx]):
                    col = int(state.ball_cols[row_idx])
                    board_np[row_idx, col] = 1.0

            # Mark paddle (row rows-1)
            paddle_col = int(state.paddle_x)
            board_np[env.rows - 1, paddle_col] = 1.0

            # Flatten and permute
            expected_obs = board_np.flatten()[np.array(state.shuffle_idx)]

            # Compare with actual observation
            assert jnp.allclose(obs, expected_obs), (
                f"Observation mismatch at step {step_i}"
            )

            # Step
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            obs, state, _, _, _, _ = env.step(key, state, action, params)

    def test_observation_equals_flat_board_without_swap(self, key: jax.Array) -> None:
        """With swap_every large, observation equals the plain flattened board."""
        env = DancingCatch(rows=6, columns=3)
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0, swap_every=100000)

        for _ in range(20):
            # Reconstruct plain flattened board
            board_np = np.zeros((env.rows, env.columns), dtype=np.float32)
            for row_idx in range(env.rows):
                if bool(state.ball_mask[row_idx]):
                    col = int(state.ball_cols[row_idx])
                    board_np[row_idx, col] = 1.0
            board_np[env.rows - 1, int(state.paddle_x)] = 1.0
            expected_flat = board_np.flatten()

            assert jnp.allclose(obs, expected_flat), (
                "Without swaps, observation should equal plain flattened board"
            )

            key, subkey = jax.random.split(key)
            obs, state, _, _, _, _ = env.step(key, state, 1, params)


class TestSwapTiming:
    """Tests for when swaps occur and time_since_swap behavior."""

    @pytest.mark.parametrize("swap_every", [1, 2, 5, 17])
    def test_time_since_swap_cycles(
        self, deterministic_key: jax.Array, swap_every: int
    ) -> None:
        """With swap_every=k, time_since_swap cycles 1,2,...,k-1,0 and shuffle_idx
        changes only when time_since_swap returns to 0.
        """
        env = DancingCatch(rows=4, columns=3)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0, swap_every=swap_every)

        prev_shuffle_idx = state.shuffle_idx.copy()

        for step_i in range(50):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

            # time_since_swap should cycle in [0, 1, ..., swap_every-1]
            assert 0 <= int(state.time_since_swap) < swap_every, (
                f"time_since_swap out of range at step {step_i}: "
                f"{state.time_since_swap}"
            )

            # If time_since_swap == 0, a swap just occurred (shuffle_idx may differ)
            # If time_since_swap != 0, shuffle_idx should be unchanged
            if int(state.time_since_swap) != 0:
                assert jnp.array_equal(state.shuffle_idx, prev_shuffle_idx), (
                    f"shuffle_idx must not change except when time_since_swap == 0; "
                    f"changed at step {step_i}"
                )

            prev_shuffle_idx = state.shuffle_idx.copy()


class TestSwapMagnitude:
    """Tests for swap magnitude - how many positions change per swap."""

    def test_swap_magnitude_is_zero_or_two(
        self, deterministic_key: jax.Array
    ) -> None:
        """On a swap step, shuffle_idx differs from its previous value in exactly
        0 or 2 positions, never 1 and never more than 2.
        """
        env = DancingCatch(rows=4, columns=3)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0, swap_every=1)

        prev_shuffle_idx = state.shuffle_idx.copy()
        swap_count = 0

        for step_i in range(100):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

            # Since swap_every=1, a swap happens every step
            diffs = jnp.sum(state.shuffle_idx != prev_shuffle_idx)
            assert int(diffs) in [0, 2], (
                f"Swap must change exactly 0 or 2 positions; "
                f"got {int(diffs)} at step {step_i}"
            )

            if int(diffs) == 2:
                swap_count += 1

            prev_shuffle_idx = state.shuffle_idx.copy()

        assert swap_count > 0, "Should observe at least some swaps with swap_every=1"


class TestPermutationDrift:
    """Tests for permutation drifting away from identity over time."""

    def test_permutation_drifts_from_identity(
        self, deterministic_key: jax.Array
    ) -> None:
        """Over a long rollout with swap_every=1 and fixed seed, shuffle_idx
        eventually differs from the identity.
        """
        env = DancingCatch(rows=4, columns=3)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0, swap_every=1)

        drifted = False
        for step_i in range(200):
            identity = jnp.arange(12, dtype=jnp.int32)
            if not jnp.array_equal(state.shuffle_idx, identity):
                drifted = True
                break

            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

        assert drifted, "Permutation should drift from identity with swap_every=1"


class TestSwapStatistics:
    """Tests for uniformity of swap positions over many steps."""

    def test_swap_positions_roughly_uniform(self, deterministic_key: jax.Array) -> None:
        """With swap_every=1 over thousands of steps and fixed seed, the swapped
        positions are roughly uniform over [0, rows*columns). Uses generous
        tolerance and fixed seed for reproducibility.
        """
        env = DancingCatch(rows=5, columns=5)
        key = deterministic_key
        obs, state = env.reset(key)
        params = DancingCatchParams(spawn_probability=0.0, swap_every=1)

        # Track which positions appear in swaps
        position_counts = np.zeros(25, dtype=np.int32)
        prev_shuffle_idx = state.shuffle_idx.copy()

        for _ in range(5000):
            key, subkey = jax.random.split(key)
            _, state, _, _, _, _ = env.step(key, state, 1, params)

            # Find which positions changed
            changed_mask = state.shuffle_idx != prev_shuffle_idx
            changed_positions = np.where(np.array(changed_mask))[0]

            for pos in changed_positions:
                position_counts[pos] += 1

            prev_shuffle_idx = state.shuffle_idx.copy()

        # Check that position counts are roughly uniform
        # With 5000 steps and swap_every=1, expect ~2 changes per step =
        # ~10000 total changes
        # Distributed over 25 positions: ~400 per position
        # Allow large tolerance since this is statistical
        expected_count = float(np.sum(position_counts)) / 25
        max_tolerance = expected_count * 0.3  # Allow ±30%

        for i, count in enumerate(position_counts):
            assert abs(count - expected_count) < max_tolerance, (
                f"Position {i} frequency {count} deviates too much "
                f"from expected {expected_count}"
            )


class TestNonDefaultBoardSizes:
    """Tests for correct behavior at non-default board dimensions."""

    @pytest.mark.parametrize("rows,columns", [(6, 3), (2, 2), (20, 7)])
    def test_board_sizes_work(self, key: jax.Array, rows: int, columns: int) -> None:
        """Non-default board sizes work, including observation_dim scaling."""
        env = DancingCatch(rows=rows, columns=columns)
        obs, state = env.reset(key)

        expected_obs_dim = rows * columns
        assert obs.shape == (expected_obs_dim,)
        assert state.shuffle_idx.shape == (expected_obs_dim,)

        params = DancingCatchParams(spawn_probability=0.1, swap_every=10)

        for _ in range(50):
            key, subkey = jax.random.split(key)
            action = int(jax.random.randint(subkey, (), 0, NUM_ACTIONS))
            obs, state, _, _, _, _ = env.step(key, state, action, params)

            assert obs.shape == (expected_obs_dim,)
            assert state.shuffle_idx.shape == (expected_obs_dim,)


class TestJAXTransformations:
    """Smoke tests for JAX transformations (jit, vmap, scan)."""

    def test_jit_step(self, key: jax.Array) -> None:
        """jax.jit wrapping step must work without errors."""
        env = DancingCatch()
        obs, state = env.reset(key)

        jitted_step = jax.jit(env.step)
        obs, state, reward, terminated, truncated, info = jitted_step(key, state, 1)

        assert obs.shape == (50,)
        assert isinstance(state, DancingCatchState)

    def test_vmap_reset_over_keys(self, key: jax.Array) -> None:
        """vmap over reset keys produces batched observations and states."""
        env = DancingCatch()
        keys = jax.random.split(key, 8)

        vmapped_reset = jax.vmap(env.reset)
        obs_batch, state_batch = vmapped_reset(keys)

        assert obs_batch.shape == (8, 50)
        assert state_batch.paddle_x.shape == (8,)

    def test_vmap_step_over_params(self, key: jax.Array) -> None:
        """vmap over DancingCatchParams (varying swap_every and spawn_probability)."""
        env = DancingCatch()
        obs, state = env.reset(key)

        spawn_probs = jnp.array([0.0, 0.1, 0.5, 1.0])

        def step_with_params(spawn_prob):
            params = DancingCatchParams(spawn_probability=spawn_prob)
            _, next_state, reward, _, _, _ = env.step(key, state, 1, params)
            return next_state, reward

        vmapped_step = jax.vmap(step_with_params)
        next_states_batch, rewards_batch = vmapped_step(spawn_probs)

        assert rewards_batch.shape == (4,)
        assert next_states_batch.paddle_x.shape == (4,)

    def test_lax_scan_rollout(self, key: jax.Array) -> None:
        """lax.scan over step works: the standard jitted-rollout pattern."""
        env = DancingCatch()
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
        assert obs_seq.shape == (50, 50)
        assert isinstance(final_state, DancingCatchState)


class TestRender:
    """Tests for render() output and correctness."""

    def test_render_returns_uint8_rgb(self, key: jax.Array) -> None:
        """render(state) returns uint8[rows, columns, 3]."""
        env = DancingCatch(rows=6, columns=4)
        obs, state = env.reset(key)
        rgb = env.render(state)

        assert rgb.shape == (6, 4, 3)
        assert rgb.dtype == jnp.uint8

    def test_render_values_in_range(self, key: jax.Array) -> None:
        """render() values must be in {0, 255}."""
        env = DancingCatch()
        obs, state = env.reset(key)
        rgb = env.render(state)

        unique_values = jnp.unique(rgb)
        for val in unique_values:
            assert int(val) in [0, 255], f"Got unexpected value {int(val)}"

    def test_render_matches_permuted_observation(self, key: jax.Array) -> None:
        """render() output must match the permuted observation reshaped and scaled."""
        env = DancingCatch(rows=5, columns=4)
        obs, state = env.reset(key)
        rgb = env.render(state)

        # Reconstruct expected render from observation
        obs_uint8 = (obs.astype(jnp.uint8) * 255)
        obs_board = obs_uint8.reshape(env.rows, env.columns)
        expected_rgb = jnp.tile(obs_board[..., None], (1, 1, 3))

        assert jnp.array_equal(rgb, expected_rgb), (
            "render() must return the permuted observation reshaped and scaled"
        )
