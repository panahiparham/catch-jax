"""Generate an animated GIF of a random-policy rollout on Catch.

Renders the environment by building frames from state.paddle_x, ball_cols,
and ball_mask with distinct colors for the paddle and balls, upscales with
nearest-neighbor to make the grid legible, and draws grid lines. Seed 0
produces the committed GIF: 5 catches and 11 misses over 120 steps.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from catch_jax import Catch, CatchParams

SEED = 0
NUM_STEPS = 120
ROWS = 10
COLS = 5
CELL_SIZE = 40
FRAME_DURATION_MS = 200

BACKGROUND = (0, 0, 0)
BALL_COLOR = (255, 255, 255)
PADDLE_COLOR = (0, 255, 255)
GRID_COLOR = (50, 50, 50)


def build_frame(
    paddle_x: int, ball_cols: jnp.ndarray, ball_mask: jnp.ndarray
) -> Image.Image:
    """Render a single frame as a PIL Image."""
    img_h = ROWS * CELL_SIZE
    img_w = COLS * CELL_SIZE

    canvas = np.full((img_h, img_w, 3), BACKGROUND, dtype=np.uint8)

    for r in range(ROWS):
        for c in range(COLS):
            is_paddle = (r == ROWS - 1) and (c == int(paddle_x))
            is_ball = bool(ball_mask[r]) and (int(ball_cols[r]) == c)

            if is_ball:
                color = BALL_COLOR
            elif is_paddle:
                color = PADDLE_COLOR
            else:
                color = BACKGROUND

            r1, r2 = r * CELL_SIZE, (r + 1) * CELL_SIZE
            c1, c2 = c * CELL_SIZE, (c + 1) * CELL_SIZE
            canvas[r1:r2, c1:c2] = color

    for r in range(1, ROWS):
        canvas[r * CELL_SIZE] = GRID_COLOR

    for c in range(1, COLS):
        canvas[:, c * CELL_SIZE] = GRID_COLOR

    return Image.fromarray(canvas, "RGB")


def main() -> None:
    env = Catch(rows=ROWS, columns=COLS)
    params = CatchParams(spawn_probability=0.1)

    key = jax.random.PRNGKey(SEED)
    reset_key, rollout_key = jax.random.split(key)

    _, state = env.reset(reset_key)

    frames = []
    catches = 0
    misses = 0

    frame = build_frame(state.paddle_x, state.ball_cols, state.ball_mask)
    frames.append(frame)

    keys = jax.random.split(rollout_key, NUM_STEPS)
    for step_idx in range(NUM_STEPS):
        action_key, step_key = jax.random.split(keys[step_idx])
        action = jax.random.randint(action_key, (), 0, 3)
        _, state, reward, _, _, _ = env.step(step_key, state, action, params)

        if reward > 0:
            catches += 1
        elif reward < 0:
            misses += 1

        frame = build_frame(state.paddle_x, state.ball_cols, state.ball_mask)
        frames.append(frame)

    output_path = Path("catch_random_policy.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
    )

    file_size_kb = output_path.stat().st_size / 1024

    print(f"Output: {output_path}")
    print(f"Frames: {len(frames)}")
    print(f"Size: {output_path.stat().st_size} bytes ({file_size_kb:.1f} KB)")
    print(f"Catches: {catches}, Misses: {misses}")


if __name__ == "__main__":
    main()
