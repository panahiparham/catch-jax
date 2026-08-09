"""Minimal Catch usage example: a jitted ``lax.scan`` rollout.

Run with::

    uv run python example.py
"""

import jax
import jax.numpy as jnp

from catch_jax import Catch, CatchParams

NUM_STEPS = 10


def main() -> None:
    env = Catch()  # rows and columns are optional; defaults: 10, 5
    params = CatchParams(spawn_probability=0.1, max_steps_in_episode=1000)

    key = jax.random.PRNGKey(42)
    reset_key, action_key, rollout_key = jax.random.split(key, 3)

    obs, state = env.reset(reset_key)
    actions = jax.random.randint(action_key, (NUM_STEPS,), 0, 3)

    @jax.jit
    def rollout(key, state, actions):
        def step(carry, action):
            key, state = carry
            key, subkey = jax.random.split(key)
            obs, state, reward, terminated, truncated, info = env.step(
                subkey, state, action, params
            )
            return (key, state), (obs, reward, state.timestep)

        (_, final_state), (obs_seq, rewards, timesteps) = jax.lax.scan(
            step, (key, state), actions
        )
        return final_state, obs_seq, rewards, timesteps

    final_state, obs_seq, rewards, timesteps = rollout(rollout_key, state, actions)

    # The paddle row is never occupied by a ball post-step (a landing ball is
    # resolved and removed the same step), so `obs_seq[t].sum() - 1` (minus the
    # paddle's own cell) is the number of balls currently falling.
    print(f"start obs shape: {obs.shape}, dtype: {obs.dtype}")
    for t in range(NUM_STEPS):
        r = float(rewards[t])
        num_balls_falling = int(obs_seq[t].sum()) - 1
        print(
            f"step {t:2d}  action={int(actions[t])}  "
            f"reward={r:+.0f}  "
            f"balls_falling={num_balls_falling}  "
            f"cumulative_reward={float(jnp.sum(rewards[:t+1])):+.0f}"
        )

    rgb_image = env.render(final_state)
    print(f"rendered RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")


if __name__ == "__main__":
    main()
