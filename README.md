# catch-jax

A JAX implementation of the continuing Catch environment from
[google-deepmind/csuite](https://github.com/google-deepmind/csuite). Unlike
bsuite's episodic Catch variant, this port targets the **continuing** environment
with its own reward structure and dynamics. `reset` and `step` are pure functions
and are fully JIT- and vmap-able, enabling distributed training and large-scale
experimentation.

## References

- **csuite repository:** google-deepmind/csuite. [https://github.com/google-deepmind/csuite](https://github.com/google-deepmind/csuite)
  Specifically, the Catch environment implementation at [csuite/environments/catch.py](https://github.com/google-deepmind/csuite/blob/main/csuite/environments/catch.py).

## Installation

Add it to your project with [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/panahiparham/catch-jax
```

## Usage

```python
import jax
import jax.numpy as jnp

from catch_jax import Catch, CatchParams

NUM_STEPS = 10

def main():
    # Create environment with default board size (10 rows x 5 columns)
    env = Catch()  # rows and columns are optional; defaults: 10, 5
    
    # Customize parameters as needed
    params = CatchParams(spawn_probability=0.1, max_steps_in_episode=1000)

    # Initialize random key
    key = jax.random.PRNGKey(42)
    reset_key, action_key, rollout_key = jax.random.split(key, 3)

    # Reset environment: returns (observation, state)
    obs, state = env.reset(reset_key)
    print(f"Initial board shape: {obs.shape}, dtype: {obs.dtype}")
    
    # Actions: 0 = LEFT, 1 = STAY, 2 = RIGHT
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

    # Print metrics for each step. The paddle row is never occupied by a ball
    # post-step (a landing ball is resolved and removed the same step), so
    # `obs_seq[t].sum() - 1` (minus the paddle's own cell) is the number of
    # balls currently falling.
    print(f"\nRollout over {NUM_STEPS} steps:")
    for t in range(NUM_STEPS):
        r = float(rewards[t])
        num_balls_falling = int(obs_seq[t].sum()) - 1
        print(f"  step {t:2d}  action={int(actions[t])}  "
              f"reward={r:+.0f}  "
              f"balls_falling={num_balls_falling}  "
              f"cumulative_reward={float(jnp.sum(rewards[:t+1])):+.0f}")

    # Render the final state as an RGB image
    rgb_image = env.render(final_state)
    print(f"\nRendered RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

if __name__ == "__main__":
    main()
```

**Observation:** The `observation` returned by `reset` and `step` is a float32 array of shape
`(rows, columns)`, with 1.0 indicating a ball or paddle, and 0.0 indicating an empty cell.

**Reward:** At each step, the reward is:
- `+1.0` when a ball reaches the paddle row in the paddle's column (a catch)
- `-1.0` when a ball reaches the paddle row outside the paddle's column (a miss)
- `0.0` otherwise

**Termination:** The environment is continuing; `terminated` is always `False`. An episode
may be truncated if `timestep >= params.max_steps_in_episode` (configurable).

## Benchmark

Below is a benchmark comparing a small DQN agent against a uniform-random baseline
on the default 10×5 Catch environment over 50,000 timesteps, repeated across
30 random seeds.

![DQN vs. random agent on Catch](benchmark_dqn.png)

(vector version: [`benchmark_dqn.pdf`](benchmark_dqn.pdf))

Run the benchmark with:

```sh
uv run --group benchmark python benchmark_dqn.py
```

### Hyperparameters

The hyperparameters are derived from the NeverEndingRL suite
(`position_catch/Catch50k/DQN.json`):

| Hyperparameter | Value |
| --- | --- |
| Total timesteps | 50,000 |
| Seeds | 30 |
| Replay buffer size | 100,000 (uniform, iid sampling) |
| Batch size | 32 |
| Update frequency | Every 4 environment steps |
| Target network refresh | Hard copy every 128 steps |
| Warm-up | Updates begin once buffer ≥ batch size (32 transitions) |
| Epsilon | 0.01 (constant epsilon-greedy) |
| Optimizer | Adam: learning rate 0.01, β₁=0.9, β₂=0.999, ε=1e-8 |
| Network | 2 hidden layers × 32 units, ReLU activation |
| Discount (γ) | 0.9 |

### Metrics

Two exponential moving averages (EMA) are tracked with β=0.99:

1. **Reward EMA:** Updated every step with the raw reward (−1, 0, or +1). Since
   Catch is continuing (no episode termination), the "return" is not well-defined
   in the episodic sense; instead, we track the moving average of step rewards
   to assess convergence of the agent's behavior.

2. **Catch-rate EMA:** Event-gated, updated *only* on steps where a ball is resolved
   (`reward ≠ 0`). The value is 1.0 for a catch and 0.0 for a miss. On steps where
   no ball resolves (`reward = 0`), this EMA is unchanged. This metric isolates the
   agent's accuracy when catching is possible.

Both EMAs are bias-corrected at each step using the standard correction:
`ema_corrected = ema / (1 − β^n)`, where n is the count of updates (different
for each EMA).

### Results

Final metrics (mean across 30 seeds):

- **DQN:** Reward EMA ≈ 0.084, Catch-rate EMA ≈ 0.903
- **Random:** Reward EMA ≈ −0.058, Catch-rate EMA ≈ 0.203

The random agent achieves a catch rate around 0.2 (uniform random action over
5 columns) and negative expected reward. DQN learns to catch effectively,
reaching near-perfect performance.

## Differences from csuite

This port makes several deliberate changes and fixes relative to the original
csuite implementation:

| Aspect | csuite | catch-jax | Reason |
| --- | --- | --- | --- |
| **State representation** | Mutable list of balls | Fixed-shape arrays: `ball_cols[rows]` (column per row) + `ball_mask[rows]` (occupancy) | JAX requires static shapes; the ≤1-ball-per-row invariant enables exact representation. |
| **RNG handling** | Stored `np.random.Generator` in state | Pure `key` argument to `reset`/`step` | Functional purity; `step` consumes the key. |
| **Observation dtype** | `int` (0 or 1) | `float32` | Consistency with `pinball-jax` and neural network conventions. |
| **Board size in `_get_observation`** | Hard-coded `_ROWS`, `_COLUMNS` module constants | Respects configured `self.rows`, `self.columns` | csuite's bug (b): non-default boards silently produce wrong-shape observations or index errors. |
| **Spawn semantics** | Docstring claims balls always respawn; code only spawns probabilistically | Matches code (probabilistic-only) | csuite's bug (a): the docstring is misleading; we implement the actual behavior. |
| **Minimum board height** | No validation; `rows=1` breaks silently | Validates `rows >= 2` with an explicit error | With 1 row, the spawn row and paddle row coincide; balls never resolve. |
| **Protocol interface** | `start()`/`step()` + dm_env specs | `reset()`/`step()` + `observation_space()`/`action_space()` | Conform to the `GymEnv` protocol shared with `pinball-jax`. |
| **Episode termination** | No termination flag | `terminated` is always `False`; `truncated` on reaching `max_steps_in_episode` | The protocol requires both flags; Catch is continuing. |
| **Rendering** | `binary_board_to_rgb()` as a standalone function | `render(state)` method returning uint8 RGB | Added convenience; pinball-jax has no render method. |

The RNG stream cannot be perfectly aligned with numpy (csuite draws the spawn column
*only* when the Bernoulli succeeds; JAX splits and draws both unconditionally), but
behavioral fidelity is verified via event-replay parity testing.

## Related work

This project mirrors the structure of [`pinball-jax`](https://github.com/panahiparham/pinball-jax),
a JAX implementation of a different control environment by the same author.
