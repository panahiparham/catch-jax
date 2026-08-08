"""Vendored numpy reference DancingCatch environment, used ONLY as a test oracle.

This is a verbatim-derived copy of the original numpy DancingCatch implementation
from google-deepmind/csuite (https://github.com/google-deepmind/csuite/blob/main/csuite/environments/dancing_catch.py),
adapted for test-only use as a parity oracle against catch-jax.

The parity tests roll out both the JAX env and this reference side-by-side and
assert exact agreement on observation and reward at every step, and on the
permutation structure (shuffle_idx) as well.

Minor modifications are marked with comments:
- Removed dm_env and csuite dependencies; stands alone with only numpy and stdlib.
- Fixed bug: allocate board from self._params.rows/columns instead of module constants (same as _reference_catch.py).
- Added injectable spawn source: optional list of (spawn: bool, column: int) events
  for test-driven determinism. When exhausted or None, falls back to RNG.
- Added injectable swap source: optional list of (idx_1: int, idx_2: int) events
  for test-driven determinism. When exhausted or None, falls back to RNG.
- Modified constructor to match reset() style: rows, columns, spawn_probability, seed,
  swap_every, and optional event sources are all constructor parameters, not class config.
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
    """DancingCatch environment parameters."""
    rows: int = 10
    columns: int = 5
    observation_dim: int = 50
    spawn_probability: float = 0.1
    swap_every: int = 10000


@dataclass
class State:
    """DancingCatch environment state.

    Attributes:
        paddle_x: An integer denoting the x-coordinate of the paddle.
        paddle_y: An integer denoting the y-coordinate of the paddle.
        balls: A list of (x, y) coordinates representing the present balls.
        shuffle_idx: Indices for performing the observation shuffle as a result of
            the random swaps.
        time_since_swap: An integer denoting how many timesteps have elapsed since
            the last swap.
        rng: Internal NumPy pseudo-random number generator.
    """
    paddle_x: int
    paddle_y: int
    balls: list[tuple[int, int]]
    shuffle_idx: np.ndarray
    time_since_swap: int
    rng: np.random.Generator


class DancingCatch:
    """DancingCatch environment: a paddle catches falling balls on a grid.

    The observation is a flattened binary board that is permuted at random intervals.
    At each swap_every steps, two entries in the observation are transposed,
    making the observation mapping non-stationary. This tests whether a learner can
    track objects despite observation scrambling.
    """

    def __init__(
        self,
        rows: int = 10,
        columns: int = 5,
        spawn_probability: float = 0.1,
        spawn_events: Optional[Sequence[tuple[bool, int]]] = None,
        swap_events: Optional[Sequence[tuple[int, int]]] = None,
        seed: Optional[int] = None,
        swap_every: int = 10000,
    ):
        """Initialize the DancingCatch environment.

        Args:
            rows: Grid height (must be >= 2).
            columns: Grid width (must be >= 1).
            spawn_probability: Probability of spawning a ball each step.
            spawn_events: Optional sequence of (spawn: bool, column: int) tuples
                to inject as spawn events, one per step() call, overriding RNG.
                When exhausted or None, falls back to RNG-driven behavior.
            swap_events: Optional sequence of (idx_1: int, idx_2: int) tuples
                to inject as swap events, one per step where a swap fires,
                overriding RNG. When exhausted or None, falls back to RNG.
            seed: Seed for the internal RNG, used when events are not provided
                (or are exhausted).
            swap_every: Interval (in steps) at which a swap occurs.
        """
        if rows < 2:
            raise ValueError(
                f"rows must be >= 2 (got {rows}); row {rows-1} is the paddle row "
                "and at least one row above it is needed for a ball to fall through"
            )
        if columns < 1:
            raise ValueError(f"columns must be >= 1 (got {columns})")

        self._params = Params(
            rows=rows,
            columns=columns,
            observation_dim=rows * columns,
            spawn_probability=spawn_probability,
            swap_every=swap_every,
        )
        self._rng = np.random.default_rng(seed)

        # Event injection for testing: queues of tuples.
        # When provided, these override the RNG draws.
        if spawn_events is not None:
            self._spawn_events: Iterator[tuple[bool, int]] | None = iter(spawn_events)
        else:
            self._spawn_events = None

        if swap_events is not None:
            self._swap_events: Iterator[tuple[int, int]] | None = iter(swap_events)
        else:
            self._swap_events = None

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

        # csuite stores balls as a list of (x, y) tuples, not as arrays.
        # This is the key difference from the plain Catch oracle: we must
        # keep this representation verbatim so that we validate the JAX
        # environment's fixed-shape encoding against csuite's list encoding.
        balls = [(ball_col, 0)]

        self._state = State(
            paddle_x=paddle_x,
            paddle_y=self._params.rows - 1,
            balls=balls,
            shuffle_idx=np.arange(self._params.observation_dim, dtype=np.int32),
            time_since_swap=0,
            rng=self._rng,
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
        5. Increment time_since_swap.
        6. Conditionally swap observation permutation.
        7. Return the updated state.

        Args:
            action: Action index (0=LEFT, 1=STAY, 2=RIGHT).

        Returns:
            Updated State after the step.
        """
        assert self._state is not None, "Must call reset() before step()"

        # 1. Move paddle: LEFT=0 -> dx=-1, STAY=1 -> dx=0, RIGHT=2 -> dx=+1
        paddle_x = int(np.clip(
            self._state.paddle_x + (action - 1), 0, self._params.columns - 1
        ))

        # 2. Descend balls: move all balls down by one row.
        balls = [(x, y + 1) for x, y in self._state.balls]

        # 3. Resolve ball at paddle row: check if oldest ball reached the bottom.
        reward = 0.0
        if balls and balls[0][1] == self._state.paddle_y:
            if balls[0][0] == paddle_x:
                reward = 1.0
            else:
                reward = -1.0
            # Remove the ball that reached the paddle row.
            balls = balls[1:]

        # 4. Spawn: either from injected events or RNG.
        # The RNG fallback mirrors csuite's short-circuit draw: the column is
        # only drawn when the spawn check succeeds.
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
            balls.append((new_col, 0))

        # 5. Update time since last swap.
        time_since_swap = self._state.time_since_swap + 1

        # 6. Update the observation permutation indices by swapping two indices,
        # at the given interval. Keep csuite's check verbatim.
        shuffle_idx = self._state.shuffle_idx.copy()
        if time_since_swap % self._params.swap_every == 0:
            if self._swap_events is not None:
                try:
                    idx_1, idx_2 = next(self._swap_events)
                except StopIteration:
                    idx_1, idx_2 = self._rng.integers(
                        self._params.observation_dim, size=2
                    )
            else:
                idx_1, idx_2 = self._rng.integers(
                    self._params.observation_dim, size=2
                )
            # Both reads are taken from the pre-swap array, so i1 == i2 is a no-op.
            shuffle_idx[[idx_1, idx_2]] = shuffle_idx[[idx_2, idx_1]]
            time_since_swap = 0

        # 7. Update state
        self._state = State(
            paddle_x=paddle_x,
            paddle_y=self._state.paddle_y,
            balls=balls,
            shuffle_idx=shuffle_idx,
            time_since_swap=time_since_swap,
            rng=self._rng,
        )
        self._last_reward = reward
        return self._state

    def get_observation(self) -> np.ndarray:
        """Get the permuted flat observation.

        Constructs a binary board (rows, columns) with 1 where a ball or paddle
        sits, 0 elsewhere, flattens it to (rows*columns,), and gathers entries
        through the shuffle_idx permutation.

        NOTE: Bug fix from csuite: allocate board from self._params.rows/columns
        instead of module constants, so non-default board sizes work correctly.
        At the default 10x5 size, this is a no-op.

        Returns:
            int array of shape (rows*columns,).
        """
        assert self._state is not None
        # Allocate from actual params, not hardcoded module constants (bug fix).
        board = np.zeros((self._params.rows, self._params.columns), dtype=np.int32)

        # Place balls.
        for x, y in self._state.balls:
            board[y, x] = 1

        # Place paddle at the bottom row.
        board[self._state.paddle_y, self._state.paddle_x] = 1

        # Flatten and permute.
        flat = board.flatten()
        return flat[self._state.shuffle_idx]

    def get_reward(self) -> float:
        """Get the reward from the last step."""
        return self._last_reward

    def get_state(self) -> State:
        """Get the current state."""
        assert self._state is not None
        return copy.deepcopy(self._state)

    def binary_board_to_rgb(self) -> np.ndarray:
        """Render the permuted board as an RGB image.

        Matches csuite's render(), which reshapes the shuffled observation back
        to (rows, columns) and passes it straight to binary_board_to_rgb, so the
        image shows the permuted board: 0 (black) for empty cells, 255 (white)
        for occupied cells, tiled to 3 channels.

        NOTE: Bug fix from csuite: reshape to self._params.rows/columns instead
        of module constants, so non-default board sizes work correctly.

        Returns:
            uint8 array of shape (rows, columns, 3).
        """
        assert self._state is not None
        board = self.get_observation().reshape(self._params.rows, self._params.columns)
        board_uint8 = board.astype(np.uint8) * 255
        # Expand to 3 channels.
        return np.tile(board_uint8[:, :, np.newaxis], (1, 1, 3))
