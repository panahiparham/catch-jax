"""Parity tests: JAX DancingCatch must match the vendored numpy oracle.

Testing strategy:

(a) Deterministic parity (spawn_probability=0.0, swap_every > rollout length):
    Both environments are fully deterministic given the initial ball column
    and permutation. We seed the oracle with the exact column that JAX's
    reset drew, roll >= 300 random actions, and assert exact agreement on
    observation, reward, and shuffle_idx at every step.

(b) Event-replay parity (spawn_probability in {0.1, 0.5, 1.0}, swap_every in {1, 7}):
    The two RNG streams cannot be aligned (csuite draws the spawn column only
    when the Bernoulli succeeds, while JAX splits and draws everything
    unconditionally). We roll the JAX env once, extracting spawn and swap
    events as follows:
    - Spawn events come directly from state: row 0 is always freshly written
      by step, so (bool(state.ball_mask[0]), int(state.ball_cols[0])) unambiguously
      encodes the spawn event.
    - Swap events are recovered by diffing shuffle_idx across steps. When a swap
      fires, exactly 2 positions change (or 0 if the two drawn indices coincide,
      making it a no-op). We extract the pair of changed positions as the event.
    Then we replay these events through the oracle and assert exact agreement.

(c) Permutation parity: Assert that the oracle's shuffle_idx equals the JAX
    state's shuffle_idx at every step. This independently validates the
    permutation logic.

(d) Render parity: DancingCatch.render(state) matches the oracle's
    binary_board_to_rgb() byte for byte. Use spawn_probability=0.0 with a
    small swap_every so swaps do fire, replaying the recovered swap events
    into the oracle to stay synchronized.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from catch_jax.dancing_catch import DancingCatch, DancingCatchParams
from _reference_dancing_catch import DancingCatch as DancingCatchReference


# =========================================================================== #
# (a) Deterministic parity: spawn_probability = 0.0, swap_every > rollout
# =========================================================================== #

DETERMINISTIC_BOARD_SIZES = [
    (10, 5),  # default
    (6, 3),   # non-default
]
DETERMINISTIC_SEEDS = [0, 1, 2]
DETERMINISTIC_STEPS = 300


@pytest.mark.parametrize("rows,columns", DETERMINISTIC_BOARD_SIZES)
@pytest.mark.parametrize("seed", DETERMINISTIC_SEEDS)
def test_deterministic_parity(rows: int, columns: int, seed: int) -> None:
    """JAX and reference agree exactly over 300+ random actions with
    spawn_probability=0.0 and swap_every > rollout length.
    """
    # Create and reset the JAX environment.
    jax_env = DancingCatch(rows=rows, columns=columns)
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    # Extract the initial ball column from JAX's state.
    initial_ball_col = int(state_jax.ball_cols[0])

    # Create and reset the reference with spawn_probability=0.0
    # and swap_every > rollout (so no swaps occur).
    ref = DancingCatchReference(
        rows=rows,
        columns=columns,
        spawn_probability=0.0,
        swap_every=DETERMINISTIC_STEPS + 1,
    )
    ref.reset(initial_ball_column=initial_ball_col)

    # Generate random actions.
    action_key = jax.random.PRNGKey(seed + 1000)
    actions = jax.random.randint(action_key, (DETERMINISTIC_STEPS,), 0, 3)
    actions = np.array(actions)

    # Roll both environments through the same actions.
    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Step the JAX environment.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, reward_jax, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, DancingCatchParams(spawn_probability=0.0)
        )

        # Step the reference.
        ref.step(action_int)
        obs_ref = ref.get_observation()
        reward_ref = ref.get_reward()
        shuffle_idx_ref = ref.get_state().shuffle_idx

        # Construct the expected JAX observation for comparison.
        obs_jax_int = obs_jax.astype(np.int32)

        # Assert exact agreement on observation.
        np.testing.assert_array_equal(
            obs_jax_int,
            obs_ref,
            err_msg=(
                f"Observation mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, seed={seed})"
            ),
        )

        # Assert exact agreement on reward.
        np.testing.assert_equal(
            float(reward_jax),
            float(reward_ref),
            err_msg=(
                f"Reward mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, seed={seed})"
            ),
        )

        # Assert exact agreement on shuffle_idx (permutation parity).
        np.testing.assert_array_equal(
            np.array(state_jax.shuffle_idx),
            shuffle_idx_ref,
            err_msg=(
                f"Shuffle index mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, seed={seed})"
            ),
        )


# =========================================================================== #
# (b) & (c) Event-replay parity and permutation parity
# =========================================================================== #

EVENT_REPLAY_BOARD_SIZES = [
    (10, 5),  # default
    (6, 3),   # non-default
]
# Pair (spawn_probability, swap_every)
EVENT_REPLAY_CONFIG = [
    (0.1, 1),
    (0.1, 7),
    (0.5, 1),
    (0.5, 7),
    (1.0, 1),
    (1.0, 7),
]
EVENT_REPLAY_SEEDS = [0, 1]
EVENT_REPLAY_STEPS = 500


@pytest.mark.parametrize("rows,columns", EVENT_REPLAY_BOARD_SIZES)
@pytest.mark.parametrize("spawn_prob,swap_every", EVENT_REPLAY_CONFIG)
@pytest.mark.parametrize("seed", EVENT_REPLAY_SEEDS)
def test_event_replay_parity(
    rows: int, columns: int, spawn_prob: float, swap_every: int, seed: int
) -> None:
    """JAX and reference agree when spawn and swap events are replayed.
    Also assert permutation parity (shuffle_idx agreement).
    """
    # Create and reset the JAX environment.
    jax_env = DancingCatch(rows=rows, columns=columns)
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    # Extract the initial ball column from JAX's state.
    initial_ball_col = int(state_jax.ball_cols[0])

    # Collect spawn and swap events as we roll the JAX environment.
    spawn_events = []
    swap_events = []

    # Generate random actions.
    action_key = jax.random.PRNGKey(seed + 1000)
    actions = jax.random.randint(action_key, (EVENT_REPLAY_STEPS,), 0, 3)
    actions = np.array(actions)

    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Remember the pre-step shuffle_idx for swap event extraction.
        shuffle_idx_pre = np.array(state_jax.shuffle_idx)

        # Step the JAX environment with a fresh key.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, reward_jax, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, DancingCatchParams(
                spawn_probability=spawn_prob, swap_every=swap_every
            )
        )

        # Extract the spawn event: row 0 is always freshly written by step(),
        # so ball_mask[0] and ball_cols[0] unambiguously indicate spawn status
        # and column.
        spawned = bool(state_jax.ball_mask[0])
        spawn_col = int(state_jax.ball_cols[0]) if spawned else 0
        spawn_events.append((spawned, spawn_col))

        # Extract the swap event by diffing shuffle_idx. The oracle consumes one
        # event per step where a swap fires, so record an event only on those
        # steps; time_since_swap returning to 0 marks them.
        shuffle_idx_post = np.array(state_jax.shuffle_idx)
        diff_positions = np.where(shuffle_idx_pre != shuffle_idx_post)[0]
        swap_fired = int(state_jax.time_since_swap) == 0

        # A swap of two distinct indices changes exactly 2 positions, and a swap
        # of one index with itself changes 0. Steps without a swap change none.
        expected_diffs = (0, 2) if swap_fired else (0,)
        assert len(diff_positions) in expected_diffs, (
            f"Step {step_idx}: expected {expected_diffs} differing positions, "
            f"got {len(diff_positions)}"
        )

        if swap_fired:
            if len(diff_positions) == 2:
                swap_events.append((int(diff_positions[0]), int(diff_positions[1])))
            else:
                # No-op swap: both drawn indices were the same.
                swap_events.append((0, 0))

    # Now create a reference with the collected spawn and swap events.
    ref = DancingCatchReference(
        rows=rows,
        columns=columns,
        spawn_probability=spawn_prob,  # unused when events are injected
        spawn_events=spawn_events,
        swap_events=swap_events,
        swap_every=swap_every,
    )
    ref.reset(initial_ball_column=initial_ball_col)

    # Reset JAX environment again and roll through the same actions.
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Step the JAX environment.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, reward_jax, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, DancingCatchParams(
                spawn_probability=spawn_prob, swap_every=swap_every
            )
        )

        # Step the reference.
        ref.step(action_int)
        obs_ref = ref.get_observation()
        reward_ref = ref.get_reward()
        shuffle_idx_ref = ref.get_state().shuffle_idx

        # Construct the expected JAX observation for comparison.
        obs_jax_int = obs_jax.astype(np.int32)

        # Assert exact agreement on observation.
        np.testing.assert_array_equal(
            obs_jax_int,
            obs_ref,
            err_msg=(
                f"Observation mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, p={spawn_prob}, "
                f"swap_every={swap_every}, seed={seed})"
            ),
        )

        # Assert exact agreement on reward.
        np.testing.assert_equal(
            float(reward_jax),
            float(reward_ref),
            err_msg=(
                f"Reward mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, p={spawn_prob}, "
                f"swap_every={swap_every}, seed={seed})"
            ),
        )

        # Assert exact agreement on shuffle_idx (permutation parity).
        np.testing.assert_array_equal(
            np.array(state_jax.shuffle_idx),
            shuffle_idx_ref,
            err_msg=(
                f"Shuffle index mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, p={spawn_prob}, "
                f"swap_every={swap_every}, seed={seed})"
            ),
        )


# =========================================================================== #
# (d) Render parity
# =========================================================================== #


@pytest.mark.parametrize("rows,columns", [(10, 5), (6, 3)])
@pytest.mark.parametrize("seed", [0, 1])
def test_render_parity(rows: int, columns: int, seed: int) -> None:
    """JAX render() matches the reference's binary_board_to_rgb() byte-for-byte.

    Uses spawn_probability=0.0 and swap_every=1 so swaps fire at every step,
    but we inject them into the oracle to stay synchronized.
    """
    # Create and reset the JAX environment.
    jax_env = DancingCatch(rows=rows, columns=columns)
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    # Create and reset the reference with the same initial ball column.
    initial_ball_col = int(state_jax.ball_cols[0])
    ref = DancingCatchReference(
        rows=rows,
        columns=columns,
        spawn_probability=0.0,
        swap_every=1,
    )
    ref.reset(initial_ball_column=initial_ball_col)

    # Render both and compare at reset.
    render_jax = np.array(jax_env.render(state_jax))
    render_ref = ref.binary_board_to_rgb()

    np.testing.assert_array_equal(
        render_jax,
        render_ref,
        err_msg=(
            f"Render mismatch at reset (rows={rows}, columns={columns}, seed={seed})"
        ),
    )

    # Roll a few steps, collecting swap events, and check render parity at each step.
    action_key = jax.random.PRNGKey(seed + 1000)
    actions = jax.random.randint(action_key, (50,), 0, 3)
    actions = np.array(actions)

    swap_events = []

    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Remember the pre-step shuffle_idx.
        shuffle_idx_pre = np.array(state_jax.shuffle_idx)

        # Step JAX.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, _, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, DancingCatchParams(
                spawn_probability=0.0, swap_every=1
            )
        )

        # Extract the swap event, recording one per step where a swap fires.
        shuffle_idx_post = np.array(state_jax.shuffle_idx)
        diff_positions = np.where(shuffle_idx_pre != shuffle_idx_post)[0]

        if int(state_jax.time_since_swap) == 0:
            if len(diff_positions) == 2:
                swap_events.append((int(diff_positions[0]), int(diff_positions[1])))
            else:
                swap_events.append((0, 0))

    # Now rebuild the reference with collected swap events.
    ref = DancingCatchReference(
        rows=rows,
        columns=columns,
        spawn_probability=0.0,
        swap_events=swap_events,
        swap_every=1,
    )
    ref.reset(initial_ball_column=initial_ball_col)

    # Reset JAX and re-roll, comparing render at each step.
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Step both.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, _, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, DancingCatchParams(
                spawn_probability=0.0, swap_every=1
            )
        )
        ref.step(action_int)

        # Render and compare.
        render_jax = np.array(jax_env.render(state_jax))
        render_ref = ref.binary_board_to_rgb()

        np.testing.assert_array_equal(
            render_jax,
            render_ref,
            err_msg=(
                f"Render mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, seed={seed})"
            ),
        )
