"""Vendored numpy reference Catch environment, used ONLY as a test oracle.

This is a verbatim-derived copy of the original numpy Catch implementation
from google-deepmind/csuite (https://github.com/google-deepmind/csuite/blob/main/csuite/environments/catch.py),
adapted for test-only use as a parity oracle against catch-jax.

The parity tests roll out both the JAX env and this reference side-by-side and
assert exact agreement on board state and reward at every step.

Minor modifications are marked with comments:
- Removed dm_env and csuite dependencies; stands alone with only numpy and stdlib.
- Fixed bug: allocate board from self._params.rows/columns instead of module constants (§3b).
- Added injectable spawn source: optional list of (spawn: bool, column: int) events
  for test-driven determinism. When exhausted or None, falls back to RNG.
"""

# Apache License Version 2.0, January 2004
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Optional, Sequence
import copy

import numpy as np


class Action(IntEnum):
    """Discrete action space: 0=LEFT, 1=STAY, 2=RIGHT."""
    LEFT = 0
    STAY = 1
    RIGHT = 2


@dataclass
class Params:
    """Catch environment parameters."""
    rows: int = 10
    columns: int = 5
    spawn_probability: float = 0.1


@dataclass
class State:
    """Catch environment state."""
    paddle_x: int  # Column position of the paddle (0 to columns-1)
    ball_cols: np.ndarray  # shape (rows,); ball column index per row
    ball_mask: np.ndarray  # shape (rows,) bool; which rows hold a ball
    timestep: int


class Catch:
    """Catch environment: a paddle catches falling balls on a grid.

    A ball descends one row per step, moving in the same column. The paddle
    is at the bottom row and can move LEFT/STAY/RIGHT. Reward is +1 when
    the ball reaches the paddle row in the paddle's column, -1 otherwise,
    0 if no ball reached the paddle row.

    This is a continuing environment (no termination).
    """

    def __init__(
        self,
        rows: int = 10,
        columns: int = 5,
        spawn_probability: float = 0.1,
        spawn_events: Optional[Sequence[tuple[bool, int]]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the Catch environment.

        Args:
            rows: Grid height (must be >= 2).
            columns: Grid width (must be >= 1).
            spawn_probability: Probability of spawning a ball each step.
            spawn_events: Optional sequence of (spawn: bool, column: int) tuples
                to inject as spawn events, one per step() call, overriding RNG.
                When exhausted or None, falls back to RNG-driven behavior.
            seed: Seed for the internal RNG, used only when spawn_events is not
                provided (or is exhausted). Matches upstream csuite's constructor.
        """
        if rows < 2:
            raise ValueError(
                f"rows must be >= 2 (got {rows}); row {rows-1} is the paddle row "
                "and at least one row above it is needed for a ball to fall through"
            )
        if columns < 1:
            raise ValueError(f"columns must be >= 1 (got {columns})")

        self._params = Params(rows=rows, columns=columns, spawn_probability=spawn_probability)
        self._rng = np.random.default_rng(seed)

        # Event injection for testing: a queue of (spawn: bool, column: int) tuples.
        # When provided, these override the RNG draws.
        if spawn_events is not None:
            self._spawn_events = iter(spawn_events)
        else:
            self._spawn_events = None

        self._state: Optional[State] = None
        self._last_reward = 0.0

    def reset(self, initial_ball_column: Optional[int] = None) -> State:
        """Reset the environment to an initial state.

        Paddle is centred at columns // 2. Exactly one ball appears at row 0
        in the specified column (or uniformly random if not specified).

        Args:
            initial_ball_column: Column index for the initial ball (0 to columns-1).
                If None, drawn uniformly at random.

        Returns:
            Initial State.
        """
        paddle_x = self._params.columns // 2

        if initial_ball_column is None:
            ball_col = self._rng.integers(0, self._params.columns)
        else:
            ball_col = initial_ball_column

        ball_cols = np.zeros(self._params.rows, dtype=np.int32)
        ball_cols[0] = ball_col

        ball_mask = np.zeros(self._params.rows, dtype=np.bool_)
        ball_mask[0] = True

        self._state = State(
            paddle_x=paddle_x,
            ball_cols=ball_cols,
            ball_mask=ball_mask,
            timestep=0,
        )
        self._last_reward = 0.0
        return self._state

    def step(self, action: int) -> State:
        """Execute one environment step.

        Step order (critical for csuite parity):
        1. Move the paddle left/stay/right.
        2. Descend all balls by one row.
        3. Resolve any ball reaching the paddle row (reward, remove).
        4. Spawn a new ball with the configured probability.
        5. Return the updated state.

        Args:
            action: Action index (0=LEFT, 1=STAY, 2=RIGHT).

        Returns:
            Updated State after the step.
        """
        assert self._state is not None, "Must call reset() before step()"

        # 1. Move paddle: LEFT=0 -> dx=-1, STAY=1 -> dx=0, RIGHT=2 -> dx=+1
        paddle_x = np.clip(self._state.paddle_x + (action - 1), 0, self._params.columns - 1)

        # 2. Descend balls: roll down and clear row 0
        cols = np.roll(self._state.ball_cols, 1)
        cols[0] = 0
        mask = np.roll(self._state.ball_mask, 1)
        mask[0] = False

        # 3. Resolve ball at paddle row
        landed = mask[self._params.rows - 1]
        if landed:
            if cols[self._params.rows - 1] == paddle_x:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        mask[self._params.rows - 1] = False

        # 4. Spawn: either from injected events or RNG.
        #
        # The RNG fallback deliberately mirrors upstream csuite's short-circuit
        # draw (`if rng.random() < p: rng.integers(columns)`) rather than JAX's
        # unconditional draw of both values: the column is only drawn when the
        # spawn check succeeds. This matters only for standalone (undriven) use
        # of this reference; every parity test below either injects events or
        # uses spawn_probability=0, so it never touches this fallback.
        if self._spawn_events is not None:
            try:
                spawn, new_col = next(self._spawn_events)
            except StopIteration:
                spawn = bool(self._rng.uniform() < self._params.spawn_probability)
                new_col = int(self._rng.integers(0, self._params.columns)) if spawn else 0
        else:
            spawn = bool(self._rng.uniform() < self._params.spawn_probability)
            new_col = int(self._rng.integers(0, self._params.columns)) if spawn else 0

        if spawn:
            cols[0] = new_col
            mask[0] = True

        # 5. Update state and timestep
        self._state = State(
            paddle_x=paddle_x,
            ball_cols=cols,
            ball_mask=mask,
            timestep=self._state.timestep + 1,
        )
        self._last_reward = reward
        return self._state

    def get_board(self) -> np.ndarray:
        """Get the binary board representation.

        Returns a (rows, columns) int array with 1 where a ball or paddle sits, 0 elsewhere.
        The paddle is always at row (rows-1).

        NOTE: Bug fix from csuite: allocate board from self._params.rows/columns
        instead of module constants, so non-default board sizes work correctly.
        At the default 10x5 size, this is a no-op.

        Returns:
            int array of shape (rows, columns).
        """
        assert self._state is not None
        # Allocate from actual params, not hardcoded module constants (bug fix).
        board = np.zeros((self._params.rows, self._params.columns), dtype=np.int32)

        # Place balls.
        for row, (col, has_ball) in enumerate(zip(self._state.ball_cols, self._state.ball_mask)):
            if has_ball:
                board[row, col] = 1

        # Place paddle at the bottom row.
        board[self._params.rows - 1, self._state.paddle_x] = 1

        return board

    def get_reward(self) -> float:
        """Get the reward from the last step."""
        return self._last_reward

    def get_state(self) -> State:
        """Get the current state."""
        assert self._state is not None
        return copy.copy(self._state)

    def binary_board_to_rgb(self) -> np.ndarray:
        """Render the board as an RGB image.

        Matches csuite's binary_board_to_rgb: 0 (black) for empty cells,
        255 (white) for occupied cells, tiled to 3 channels.

        Returns:
            uint8 array of shape (rows, columns, 3).
        """
        board = self.get_board().astype(np.uint8)
        board = board * 255
        # Expand to 3 channels.
        return np.tile(board[:, :, np.newaxis], (1, 1, 3))
