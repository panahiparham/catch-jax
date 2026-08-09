"""JAX implementation of the continuing Catch environment from csuite.

The Catch environment is a breakout-like game where a paddle at the bottom of a
grid catches falling balls. ``reset`` and ``step`` are pure functions and are
fully JIT- and vmap-able, enabling large-scale distributed training.

Implementation note on state representation: since all balls descend in lockstep
and at most one ball spawns per timestep, the environment maintains an invariant
that at most one ball occupies any given row. This permits a compact,
fixed-shape state representation using a column index per row and a mask,
rather than a variable-length ball list, making the state compatible with JAX's
array operations without truncation.
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


class CatchParams(NamedTuple):
    spawn_probability: float = DEFAULT_SPAWN_PROBABILITY
    max_steps_in_episode: int = DEFAULT_MAX_STEPS_IN_EPISODE


class CatchState(NamedTuple):
    paddle_x: jax.Array  # int32 scalar
    ball_cols: jax.Array  # int32[rows], column of the ball occupying each row
    ball_mask: jax.Array  # bool[rows], which rows currently hold a ball
    timestep: jax.Array  # int32 scalar


class _CatchObservationSpace:
    """Observation space for Catch environment."""

    def __init__(self, rows: int, columns: int) -> None:
        self._rows = rows
        self._columns = columns

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._rows, self._columns)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32


class _CatchActionSpace:
    """Action space for Catch environment."""

    @property
    def n(self) -> int:
        return NUM_ACTIONS


class Catch:
    """Catch environment adhering to the ``GymEnv`` protocol.

    A paddle moves left/stay/right along the bottom row of a grid, catching
    balls that fall from the top. The observation is a binary board indicating
    ball and paddle positions.

    Args:
        rows: Height of the grid (must be >= 2). Default is 10.
        columns: Width of the grid (must be >= 1). Default is 5.
    """

    def __init__(
        self, rows: int = DEFAULT_ROWS, columns: int = DEFAULT_COLUMNS
    ) -> None:
        if rows < 2:
            raise ValueError(
                f"rows must be >= 2 (got {rows}); row {rows-1} is the paddle row "
                "and at least one row above it is needed for a ball to fall through"
            )
        if columns < 1:
            raise ValueError(f"columns must be >= 1 (got {columns})")
        self.rows = rows
        self.columns = columns

    # -- spaces ------------------------------------------------------------- #

    def observation_space(self, params: CatchParams | None = None) -> ObservationSpace:
        del params
        return _CatchObservationSpace(self.rows, self.columns)

    def action_space(self, params: CatchParams | None = None) -> DiscreteActionSpace:
        del params
        return _CatchActionSpace()

    # -- protocol ----------------------------------------------------------- #

    def reset(
        self,
        key: jax.Array,
        params: CatchParams | None = None,
    ) -> tuple[jax.Array, CatchState]:
        """Reset the environment to an initial state.

        Paddle is centred at ``columns // 2``. Exactly one ball appears at row 0
        in a uniformly random column.

        Args:
            key: JAX random key.
            params: Environment parameters (unused).

        Returns:
            Tuple of (observation, state).
        """
        del params

        paddle_x = jnp.asarray(self.columns // 2, dtype=jnp.int32)
        ball_col = jax.random.randint(key, (), 0, self.columns)
        ball_col = jnp.asarray(ball_col, dtype=jnp.int32)

        ball_cols = jnp.zeros(self.rows, dtype=jnp.int32)
        ball_cols = ball_cols.at[0].set(ball_col)

        ball_mask = jnp.zeros(self.rows, dtype=jnp.bool_)
        ball_mask = ball_mask.at[0].set(True)

        timestep = jnp.asarray(0, dtype=jnp.int32)

        state = CatchState(
            paddle_x=paddle_x,
            ball_cols=ball_cols,
            ball_mask=ball_mask,
            timestep=timestep,
        )
        obs = self._get_observation(paddle_x, ball_cols, ball_mask)

        return obs, state

    def step(
        self,
        key: jax.Array,
        state: CatchState,
        action: jax.Array,
        params: CatchParams | None = None,
    ) -> tuple[
        jax.Array, CatchState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        """Execute one environment step.

        Step order matters for csuite parity: move the paddle, descend the
        balls, resolve any ball landing on the paddle row, then maybe spawn a
        new one (see the numbered comments in the body).

        Args:
            key: JAX random key (consumed for spawn draws).
            state: Current environment state.
            action: Action index (0=LEFT, 1=STAY, 2=RIGHT).
            params: Environment parameters.

        Returns:
            Tuple of (obs, next_state, reward, terminated, truncated, info).
        """
        params = params if params is not None else CatchParams()

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
        spawn_key, col_key = jax.random.split(key)
        spawn = jax.random.uniform(spawn_key) < params.spawn_probability
        new_col = jax.random.randint(col_key, (), 0, self.columns)
        new_col = new_col.astype(jnp.int32)
        cols = cols.at[0].set(jnp.where(spawn, new_col, cols[0]))
        mask = mask.at[0].set(spawn)

        # 5. Update timestep
        timestep = state.timestep + 1
        next_state = CatchState(
            paddle_x=paddle_x, ball_cols=cols, ball_mask=mask, timestep=timestep
        )

        # Termination signals
        terminated = jnp.asarray(False)
        truncated = timestep >= params.max_steps_in_episode
        info: dict[str, jax.Array] = {}

        # Observation
        obs = self._get_observation(paddle_x, cols, mask)

        return obs, next_state, reward, terminated, truncated, info

    def _get_observation(
        self, paddle_x: jax.Array, ball_cols: jax.Array, ball_mask: jax.Array
    ) -> jax.Array:
        """Construct the binary board observation.

        Args:
            paddle_x: Paddle column position.
            ball_cols: Column index for each row (padded with 0 for empty rows).
            ball_mask: Boolean mask indicating which rows hold a ball.

        Returns:
            Binary board of shape (rows, columns) with dtype float32.
        """
        # One-hot encoding (not scatter) so an unoccupied row's placeholder
        # column index cannot accidentally overwrite the paddle cell.
        one_hot_cols = jax.nn.one_hot(ball_cols, self.columns, dtype=jnp.float32)
        balls = one_hot_cols * ball_mask[:, None]
        paddle_row = (jnp.arange(self.rows) == self.rows - 1).astype(jnp.float32)
        one_hot_paddle = jax.nn.one_hot(paddle_x, self.columns, dtype=jnp.float32)
        paddle = one_hot_paddle * paddle_row[:, None]
        return jnp.maximum(balls, paddle)

    def render(self, state: CatchState) -> jax.Array:
        """Render the environment state as an RGB image.

        Returns a uint8 array matching csuite's binary_board_to_rgb: 0 (black)
        for empty cells, 255 (white) for occupied cells.

        Args:
            state: Environment state.

        Returns:
            RGB image of shape (rows, columns, 3) with dtype uint8.
        """
        board = self._get_observation(state.paddle_x, state.ball_cols, state.ball_mask)
        board_uint8 = (board.astype(jnp.uint8) * 255)
        # Expand channel axis and tile to 3 channels
        return jnp.tile(board_uint8[..., None], (1, 1, 3))
