"""JAX implementation of the continuing Catch environment with non-stationary observation permutations.

DancingCatch is identical to Catch except the observation is flattened to shape
(rows*columns,) and gathered through a permutation shuffle_idx. Every swap_every
steps, two uniformly-random indices of shuffle_idx are swapped, making the
observation mapping non-stationary.

The dynamics, reward, and probabilistic spawn are identical to Catch. The
environment maintains the same at-most-one-ball-per-row invariant as Catch,
allowing a compact fixed-shape state representation compatible with JAX array
operations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from catch_jax.constants import (
    DEFAULT_COLUMNS,
    DEFAULT_MAX_STEPS_IN_EPISODE,
    DEFAULT_ROWS,
    DEFAULT_SPAWN_PROBABILITY,
    NUM_ACTIONS,
)
from catch_jax.gym_env import DiscreteActionSpace, ObservationSpace

DEFAULT_SWAP_EVERY = 10_000


class DancingCatchParams(NamedTuple):
    spawn_probability: float = DEFAULT_SPAWN_PROBABILITY
    swap_every: int = DEFAULT_SWAP_EVERY
    max_steps_in_episode: int = DEFAULT_MAX_STEPS_IN_EPISODE


class DancingCatchState(NamedTuple):
    paddle_x: jax.Array
    ball_cols: jax.Array
    ball_mask: jax.Array
    shuffle_idx: jax.Array
    time_since_swap: jax.Array
    timestep: jax.Array


class _DancingCatchObservationSpace:
    """Observation space for DancingCatch environment."""

    def __init__(self, observation_dim: int) -> None:
        self._observation_dim = observation_dim

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._observation_dim,)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32


class _DancingCatchActionSpace:
    """Action space for DancingCatch environment."""

    @property
    def n(self) -> int:
        return NUM_ACTIONS


class DancingCatch:
    """DancingCatch environment adhering to the ``GymEnv`` protocol.

    A paddle moves left/stay/right along the bottom row of a grid, catching
    balls that fall from the top. The observation is a flattened binary board
    whose entries are read through a permutation that is periodically perturbed
    by a random transposition.

    :param rows: Height of the grid (must be >= 2). Default is 10.
    :param columns: Width of the grid (must be >= 1). Default is 5.
    """

    def __init__(self, rows: int = DEFAULT_ROWS, columns: int = DEFAULT_COLUMNS) -> None:
        if rows < 2:
            raise ValueError(
                f"rows must be >= 2 (got {rows}); row {rows-1} is the paddle row "
                "and at least one row above it is needed for a ball to fall through"
            )
        if columns < 1:
            raise ValueError(f"columns must be >= 1 (got {columns})")
        self.rows = rows
        self.columns = columns
        self.observation_dim = rows * columns

    # -- spaces ------------------------------------------------------------- #

    def observation_space(self, params: DancingCatchParams | None = None) -> ObservationSpace:
        del params
        return _DancingCatchObservationSpace(self.observation_dim)

    def action_space(self, params: DancingCatchParams | None = None) -> DiscreteActionSpace:
        del params
        return _DancingCatchActionSpace()

    # -- protocol ----------------------------------------------------------- #

    def reset(
        self,
        key: jax.Array,
        params: DancingCatchParams | None = None,
    ) -> tuple[jax.Array, DancingCatchState]:
        """Reset the environment to an initial state.

        Paddle is centred at ``columns // 2``. Exactly one ball appears at row 0
        in a uniformly random column. The observation permutation is initialized
        to the identity.

        :param key: JAX random key.
        :param params: Environment parameters (unused).
        :return: Tuple of (observation, state).
        """
        del params

        paddle_x = jnp.asarray(self.columns // 2, dtype=jnp.int32)
        ball_col = jax.random.randint(key, (), 0, self.columns)
        ball_col = jnp.asarray(ball_col, dtype=jnp.int32)

        ball_cols = jnp.zeros(self.rows, dtype=jnp.int32)
        ball_cols = ball_cols.at[0].set(ball_col)

        ball_mask = jnp.zeros(self.rows, dtype=jnp.bool_)
        ball_mask = ball_mask.at[0].set(True)

        shuffle_idx = jnp.arange(self.observation_dim, dtype=jnp.int32)
        time_since_swap = jnp.asarray(0, dtype=jnp.int32)
        timestep = jnp.asarray(0, dtype=jnp.int32)

        state = DancingCatchState(
            paddle_x=paddle_x,
            ball_cols=ball_cols,
            ball_mask=ball_mask,
            shuffle_idx=shuffle_idx,
            time_since_swap=time_since_swap,
            timestep=timestep,
        )
        obs = self._get_observation(paddle_x, ball_cols, ball_mask, shuffle_idx)

        return obs, state

    def step(
        self,
        key: jax.Array,
        state: DancingCatchState,
        action: jax.Array,
        params: DancingCatchParams | None = None,
    ) -> tuple[jax.Array, DancingCatchState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """Execute one environment step.

        Step order (critical for csuite parity):
        1. Move the paddle left/stay/right.
        2. Descend all balls by one row.
        3. Resolve any ball reaching the paddle row (reward, remove).
        4. Spawn a new ball with the configured probability.
        5. Advance the swap counter and, on a swap step, transpose two entries
           of the observation permutation.
        6. Return the observation built from the updated permutation.

        :param key: JAX random key (consumed for spawn and swap draws).
        :param state: Current environment state.
        :param action: Action index (0=LEFT, 1=STAY, 2=RIGHT).
        :param params: Environment parameters.
        :return: Tuple of (obs, next_state, reward, terminated, truncated, info).
        """
        params = params if params is not None else DancingCatchParams()

        # 1. Move paddle: LEFT=0 -> dx=-1, STAY=1 -> dx=0, RIGHT=2 -> dx=+1
        paddle_x = jnp.clip(state.paddle_x + (action - 1), 0, self.columns - 1)

        # 2. Descend balls: roll down and clear row 0
        cols = jnp.roll(state.ball_cols, 1).at[0].set(0)
        mask = jnp.roll(state.ball_mask, 1).at[0].set(False)

        # 3. Resolve ball at paddle row
        landed = mask[self.rows - 1]
        reward = jnp.where(
            landed,
            jnp.where(cols[self.rows - 1] == paddle_x, 1.0, -1.0),
            0.0,
        )
        reward = reward.astype(jnp.float32)
        mask = mask.at[self.rows - 1].set(False)

        # 4. Spawn
        spawn_key, col_key, swap_key = jax.random.split(key, 3)
        spawn = jax.random.uniform(spawn_key) < params.spawn_probability
        new_col = jax.random.randint(col_key, (), 0, self.columns)
        new_col = new_col.astype(jnp.int32)
        cols = cols.at[0].set(jnp.where(spawn, new_col, cols[0]))
        mask = mask.at[0].set(spawn)

        # 5. Update time_since_swap and conditionally swap observation permutation
        time_since_swap = state.time_since_swap + 1

        do_swap = time_since_swap >= params.swap_every

        # Drawn unconditionally, since JAX cannot draw inside a branch. The two
        # indices may coincide, in which case the transposition is a no-op.
        swap_idx = jax.random.randint(swap_key, (2,), 0, self.observation_dim)

        i1, i2 = swap_idx[0], swap_idx[1]
        swapped = state.shuffle_idx.at[i1].set(state.shuffle_idx[i2]).at[i2].set(state.shuffle_idx[i1])
        shuffle_idx = jnp.where(do_swap, swapped, state.shuffle_idx)
        time_since_swap = jnp.where(do_swap, 0, time_since_swap).astype(jnp.int32)

        # 6. Update timestep
        timestep = state.timestep + 1
        next_state = DancingCatchState(
            paddle_x=paddle_x,
            ball_cols=cols,
            ball_mask=mask,
            shuffle_idx=shuffle_idx,
            time_since_swap=time_since_swap,
            timestep=timestep,
        )

        # Termination signals
        terminated = jnp.asarray(False)
        truncated = timestep >= params.max_steps_in_episode
        info: dict[str, jax.Array] = {}

        # Built from the post-swap permutation, so a swap takes effect on the
        # observation returned by the step that performed it.
        obs = self._get_observation(paddle_x, cols, mask, shuffle_idx)

        return obs, next_state, reward, terminated, truncated, info

    def _get_observation(
        self,
        paddle_x: jax.Array,
        ball_cols: jax.Array,
        ball_mask: jax.Array,
        shuffle_idx: jax.Array,
    ) -> jax.Array:
        """Construct the flattened and permuted binary board observation.

        Builds a 2-D binary board (identical to Catch._get_observation), then
        flattens it and gathers entries via the shuffle_idx permutation.

        Uses one-hot encoding (not scatter) so that an unoccupied row's
        placeholder column index cannot accidentally overwrite the paddle cell.

        :param paddle_x: Paddle column position.
        :param ball_cols: Column index for each row (padded with 0 for empty rows).
        :param ball_mask: Boolean mask indicating which rows hold a ball.
        :param shuffle_idx: Permutation indices for the flattened observation.
        :return: Permuted flat board of shape (rows*columns,) with dtype float32.
        """
        balls = jax.nn.one_hot(ball_cols, self.columns, dtype=jnp.float32) * ball_mask[:, None]
        paddle_row = (jnp.arange(self.rows) == self.rows - 1).astype(jnp.float32)
        paddle = jax.nn.one_hot(paddle_x, self.columns, dtype=jnp.float32) * paddle_row[:, None]
        board = jnp.maximum(balls, paddle)
        return board.reshape(-1)[shuffle_idx]

    def render(self, state: DancingCatchState) -> jax.Array:
        """Render the environment state as an RGB image.

        Returns a uint8 array matching csuite's binary_board_to_rgb: 0 (black)
        for empty cells, 255 (white) for occupied cells. Unlike Catch.render,
        this shows the permuted board (reshaped from the shuffled observation),
        matching csuite's behavior.

        :param state: Environment state.
        :return: RGB image of shape (rows, columns, 3) with dtype uint8.
        """
        obs = self._get_observation(state.paddle_x, state.ball_cols, state.ball_mask, state.shuffle_idx)
        board = obs.reshape(self.rows, self.columns)
        board_uint8 = (board.astype(jnp.uint8) * 255)
        # Expand channel axis and tile to 3 channels
        return jnp.tile(board_uint8[..., None], (1, 1, 3))
