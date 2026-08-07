"""Parity tests: JAX Catch must match the vendored numpy oracle.

Two testing modes:

(a) Deterministic parity (spawn_probability=0.0): Both environments are fully
    deterministic given the initial ball column. We seed the reference with the
    exact column that JAX's reset drew, then roll >= 300 random actions and assert
    exact agreement on board and reward at every step.

(b) Event-replay parity (spawn_probability in {0.1, 0.5, 1.0}): We roll the JAX
    env normally, extracting spawn events directly from the state (ball_mask[0]
    and ball_cols[0] post-step), and replay those events through the reference.
    Since the reference becomes fully deterministic once its spawn events are
    fixed, this validates the entire transition function while letting the two
    RNGs differ freely.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from catch_jax.catch import Catch, CatchParams
from _reference_catch import Catch as CatchReference


# =========================================================================== #
# (a) Deterministic parity: spawn_probability = 0.0
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
    """JAX and reference agree with spawn_probability=0.0 over 300+ random actions.

    Both environments are fully deterministic given the initial ball column.
    We extract that column from JAX's reset, seed the reference with it, then
    roll both environments through the same action sequence and assert exact
    agreement on board and reward at every step.
    """
    # Create and reset the JAX environment.
    jax_env = Catch(rows=rows, columns=columns)
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    # Extract the initial ball column from JAX's state.
    initial_ball_col = int(state_jax.ball_cols[0])

    # Create and reset the reference, seeding it with the same initial column
    # and spawn_probability=0.0 (so it never spawns).
    ref = CatchReference(
        rows=rows,
        columns=columns,
        spawn_probability=0.0,
        spawn_events=None,
    )
    ref.reset(initial_ball_column=initial_ball_col)

    # Generate a deterministic sequence of random actions.
    action_key = jax.random.PRNGKey(seed + 1000)
    actions = jax.random.randint(action_key, (DETERMINISTIC_STEPS,), 0, 3)
    actions = np.array(actions)

    # Roll both environments through the same actions.
    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Step the JAX environment.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, reward_jax, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, CatchParams(spawn_probability=0.0)
        )

        # Step the reference.
        ref.step(action_int)
        board_ref = ref.get_board()
        reward_ref = ref.get_reward()

        # Construct the expected JAX board for comparison.
        board_jax = obs_jax.astype(np.int32)

        # Assert exact agreement on board.
        np.testing.assert_array_equal(
            board_jax,
            board_ref,
            err_msg=f"Board mismatch at step {step_idx} (rows={rows}, columns={columns}, seed={seed})",
        )

        # Assert exact agreement on reward.
        np.testing.assert_equal(
            float(reward_jax),
            float(reward_ref),
            err_msg=f"Reward mismatch at step {step_idx} (rows={rows}, columns={columns}, seed={seed})",
        )


# =========================================================================== #
# (b) Event-replay parity: spawn_probability in {0.1, 0.5, 1.0}
# =========================================================================== #

EVENT_REPLAY_BOARD_SIZES = [
    (10, 5),  # default
    (6, 3),   # non-default
]
EVENT_REPLAY_SPAWN_PROBS = [0.1, 0.5, 1.0]
EVENT_REPLAY_SEEDS = [0, 1]
EVENT_REPLAY_STEPS = 500


@pytest.mark.parametrize("rows,columns", EVENT_REPLAY_BOARD_SIZES)
@pytest.mark.parametrize("spawn_probability", EVENT_REPLAY_SPAWN_PROBS)
@pytest.mark.parametrize("seed", EVENT_REPLAY_SEEDS)
def test_event_replay_parity(rows: int, columns: int, spawn_probability: float, seed: int) -> None:
    """JAX and reference agree when spawn events are replayed.

    We roll the JAX environment normally, extracting spawn events directly from
    the state post-step (ball_mask[0] and ball_cols[0]). Row 0 is always freshly
    written by the step function (after descent clears it), so it unambiguously
    indicates whether a new ball spawned this step and which column.

    We then replay those exact events through the reference, which becomes fully
    deterministic. This validates the entire transition function while letting
    the RNGs differ.
    """
    # Create and reset the JAX environment.
    jax_env = Catch(rows=rows, columns=columns)
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    # Extract the initial ball column from JAX's state.
    initial_ball_col = int(state_jax.ball_cols[0])

    # Collect spawn events as we roll the JAX environment.
    spawn_events = []

    # Generate random actions and step keys.
    action_key = jax.random.PRNGKey(seed + 1000)
    actions = jax.random.randint(action_key, (EVENT_REPLAY_STEPS,), 0, 3)
    actions = np.array(actions)

    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Step the JAX environment with a fresh key.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, reward_jax, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, CatchParams(spawn_probability=spawn_probability)
        )

        # Extract the spawn event: row 0 is always freshly written by step(),
        # so ball_mask[0] and ball_cols[0] unambiguously indicate spawn status and column.
        spawned = bool(state_jax.ball_mask[0])
        spawn_col = int(state_jax.ball_cols[0]) if spawned else 0
        spawn_events.append((spawned, spawn_col))

    # Now create a reference with the collected spawn events.
    ref = CatchReference(
        rows=rows,
        columns=columns,
        spawn_probability=spawn_probability,  # unused when events are injected
        spawn_events=spawn_events,
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
            step_key, state_jax, action_int, CatchParams(spawn_probability=spawn_probability)
        )

        # Step the reference.
        ref.step(action_int)
        board_ref = ref.get_board()
        reward_ref = ref.get_reward()

        # Construct the expected JAX board for comparison.
        board_jax = obs_jax.astype(np.int32)

        # Assert exact agreement on board.
        np.testing.assert_array_equal(
            board_jax,
            board_ref,
            err_msg=(
                f"Board mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, p={spawn_probability}, seed={seed})"
            ),
        )

        # Assert exact agreement on reward.
        np.testing.assert_equal(
            float(reward_jax),
            float(reward_ref),
            err_msg=(
                f"Reward mismatch at step {step_idx} "
                f"(rows={rows}, columns={columns}, p={spawn_probability}, seed={seed})"
            ),
        )


# =========================================================================== #
# Render parity: render() matches the reference's binary_board_to_rgb()
# =========================================================================== #


@pytest.mark.parametrize("rows,columns", [(10, 5), (6, 3)])
@pytest.mark.parametrize("seed", [0, 1])
def test_render_parity(rows: int, columns: int, seed: int) -> None:
    """JAX render() matches the reference's binary_board_to_rgb() byte-for-byte.

    Uses spawn_probability=0.0 to ensure both environments remain synchronized
    without needing event injection.
    """
    # Create and reset the JAX environment.
    jax_env = Catch(rows=rows, columns=columns)
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    # Create and reset the reference with the same initial ball column and no spawns.
    initial_ball_col = int(state_jax.ball_cols[0])
    ref = CatchReference(rows=rows, columns=columns, spawn_probability=0.0)
    ref.reset(initial_ball_column=initial_ball_col)

    # Render both and compare.
    render_jax = np.array(jax_env.render(state_jax))
    render_ref = ref.binary_board_to_rgb()

    np.testing.assert_array_equal(
        render_jax,
        render_ref,
        err_msg=f"Render mismatch at reset (rows={rows}, columns={columns}, seed={seed})",
    )

    # Roll a few steps and check render parity at each step with spawn_probability=0.0.
    action_key = jax.random.PRNGKey(seed + 1000)
    actions = jax.random.randint(action_key, (50,), 0, 3)
    actions = np.array(actions)

    for step_idx, action in enumerate(actions):
        action_int = int(action)

        # Step both with spawn_probability=0.0.
        step_key = jax.random.PRNGKey(seed + step_idx + 10000)
        obs_jax, state_jax, _, _, _, _ = jax_env.step(
            step_key, state_jax, action_int, CatchParams(spawn_probability=0.0)
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
