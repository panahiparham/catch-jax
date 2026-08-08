# catch-jax

A JAX implementation of the continuing Catch environment and its non-stationary
Dancing Catch variant, both from [google-deepmind/csuite](https://github.com/google-deepmind/csuite).
`reset` and `step` are pure functions and are fully JIT- and vmap-able, enabling
distributed training and large-scale experimentation.

## References

- **csuite repository:** google-deepmind/csuite. [https://github.com/google-deepmind/csuite](https://github.com/google-deepmind/csuite)

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

## Dancing Catch

Dancing Catch is a non-stationary variant of Catch where the observation is
permuted in a time-varying manner. The paddle, balls, spawn, reward, and
termination are identical to Catch. The difference is the observation: it is
a flattened 1-D array of length `rows * columns` (50 elements at default size),
and the entries are gathered through a permutation `shuffle_idx`. Every
`swap_every` steps (default 10,000), two uniformly-random indices of
`shuffle_idx` are swapped (transposed), drifting the observation's input-to-meaning
mapping.

```python
import jax
import jax.numpy as jnp

from catch_jax import DancingCatch, DancingCatchParams

NUM_STEPS = 25_000

def main():
    # Create environment with default board size (10 rows x 5 columns)
    env = DancingCatch()

    # Use a small swap_every so swaps happen during the snippet
    params = DancingCatchParams(spawn_probability=0.1, swap_every=5_000)

    # Initialize random key
    key = jax.random.PRNGKey(42)
    reset_key, rollout_key = jax.random.split(key)

    # Reset environment: returns (observation, state)
    obs, state = env.reset(reset_key, params)
    print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Initial shuffle_idx (first 10): {state.shuffle_idx[:10]}")

    # Actions: 0 = LEFT, 1 = STAY, 2 = RIGHT
    actions = jax.random.randint(rollout_key, (NUM_STEPS,), 0, 3)

    @jax.jit
    def rollout(key, state, actions):
        def step(carry, action):
            key, state = carry
            key, subkey = jax.random.split(key)
            obs, state, reward, terminated, truncated, info = env.step(
                subkey, state, action, params
            )
            # time_since_swap returns to 0 on the step a swap fires
            return (key, state), (reward, state.time_since_swap)

        (_, final_state), (rewards, times_since_swap) = jax.lax.scan(
            step, (key, state), actions
        )
        return final_state, rewards, times_since_swap

    final_state, rewards, times_since_swap = rollout(rollout_key, state, actions)

    identity = jnp.arange(final_state.shuffle_idx.shape[0])
    print(f"\nAfter {NUM_STEPS} steps:")
    print(f"  swaps fired: {int(jnp.sum(times_since_swap == 0))}")
    print(f"  entries displaced from identity: {int(jnp.sum(final_state.shuffle_idx != identity))}")
    print(f"  cumulative reward: {float(jnp.sum(rewards)):+.0f}")

    # Render the final state as an RGB image of the permuted board
    rgb_image = env.render(final_state)
    print(f"  rendered RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

if __name__ == "__main__":
    main()
```

**Observation:** The observation returned by `reset` and `step` is a float32 array
of shape `(rows * columns,)`. It is the flattened binary board gathered through
the permutation: `board.reshape(-1)[shuffle_idx]`. A value of 1.0 indicates a
ball or paddle, and 0.0 indicates an empty cell.

**Reward:** Identical to Catch:
- `+1.0` when a ball reaches the paddle row in the paddle's column
- `-1.0` when a ball reaches the paddle row outside the paddle's column
- `0.0` otherwise

**Termination:** Identical to Catch. The environment is continuing;
`terminated` is always `False`. An episode may be truncated if
`timestep >= params.max_steps_in_episode` (configurable).

**Rendering:** `render(state)` returns a `uint8` array of shape
`(rows, columns, 3)` built from the permuted observation reshaped back to the
board. The image therefore shows the scrambled board rather than the true one.

### Benchmark

Below is a benchmark comparing DQN against a uniform-random baseline on
Dancing Catch over 500,000 timesteps, repeated across 30 random seeds.
The benchmark includes 50 observation swaps (one every 10,000 steps).

![DQN vs. random agent on Dancing Catch](benchmark_dancing_catch.png)

Run the benchmark with:

```sh
uv run --group benchmark python benchmark_dancing_catch.py
```

**Setup:** 500,000 timesteps, 30 seeds, default 10x5 board, `spawn_probability=0.1`,
`swap_every=10_000` (50 swaps total). Every DQN hyperparameter is identical to the
Catch benchmark, enabling direct comparison.

## Related work

This project mirrors the structure of [`pinball-jax`](https://github.com/panahiparham/pinball-jax),
a JAX implementation of a different control environment by the same author.
